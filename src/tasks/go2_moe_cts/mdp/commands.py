"""Reference velocity command (go2_rl_gym ``LeggedRobot._resample_commands``).

Also owns the per-env bookkeeping that the reference keeps next to the commands and
the terrain curriculum reads: commanded xy accumulation and max distance from origin.
Heading commands are disabled in the reference config and not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from .terrains import column_terrain_ids

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

Range = tuple[float, float]


@dataclass(frozen=True)
class CommandRanges:
  lin_vel_x: Range
  lin_vel_y: Range
  ang_vel_yaw: Range


def _schedule(start_iter: int, end_iter: int, start_value: float, end_value: float, it: int) -> float:
  """Reference ``get_current_scale``: linear interpolation clamped to [start, end]."""
  p = min(max((it - start_iter) / (end_iter - start_iter), 0.0), 1.0)
  return (1.0 - p) * start_value + p * end_value


@dataclass(kw_only=True)
class RefVelocityCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  resampling_time_range: tuple[float, float] = (5.0, 5.0)  # Only [0] is used.
  ranges: CommandRanges = CommandRanges((-0.5, 0.5), (-0.5, 0.5), (-1.0, 1.0))
  range_curriculum: tuple[tuple[int, CommandRanges], ...] = (
    (20000, CommandRanges((-1.0, 1.0), (-1.0, 1.0), (-1.5, 1.5))),
    (50000, CommandRanges((-2.0, 2.0), (-1.0, 1.0), (-2.0, 2.0))),
  )
  # Indexed by reference terrain id (0 wave .. 8 flat).
  terrain_max_ranges: tuple[CommandRanges, ...] = field(
    default_factory=lambda: (
      *(CommandRanges((-1.5, 1.5), (-1.0, 1.0), (-1.5, 1.5)),) * 3,
      *(CommandRanges((-1.0, 1.0), (-1.0, 1.0), (-1.5, 1.5)),) * 5,
      CommandRanges((-2.0, 2.0), (-1.0, 1.0), (-2.0, 2.0)),
    )
  )
  dynamic_resample: bool = True
  limit_vel_prob: float = 0.2
  limit_vel: tuple[tuple[int, ...], ...] = ((-1, 1), (-1, 1), (-1, 0, 1))
  limit_vel_invert_when_continuous: bool = True
  zero_command_curriculum: tuple[int, int, float, float] = (0, 1500, 0.0, 0.1)
  limit_ang_vel_at_zero_command_prob: float = 0.2
  terrain_length: float = 8.0
  num_steps_per_env: int = 24

  def build(self, env: ManagerBasedRlEnv) -> RefVelocityCommand:
    return RefVelocityCommand(self, env)


class RefVelocityCommand(CommandTerm):
  cfg: RefVelocityCommandCfg

  def __init__(self, cfg: RefVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.entity_name]
    n, dev = self.num_envs, self.device
    self.vel_command = torch.zeros(n, 3, device=dev)
    self.resampling_step = torch.zeros(n, device=dev)
    self.xy_accumulation = torch.zeros(n, 2, device=dev)
    self.max_move_distance = torch.zeros(n, device=dev)
    self.last_is_limit_vel = torch.zeros(n, dtype=torch.bool, device=dev)
    self.zero_command_proba = 0.0
    self.limit_vel_comb = torch.tensor(list(product(*cfg.limit_vel)), dtype=torch.float, device=dev)
    self._pending = sorted(cfg.range_curriculum, key=lambda x: x[0])
    terrain = env.scene.terrain
    if terrain is not None and terrain.terrain_origins is not None:
      col2id = torch.tensor(column_terrain_ids(terrain.terrain_origins.shape[1]), device=dev)
      self.terrain_ids: torch.Tensor | None = col2id[terrain.terrain_types]
    else:
      self.terrain_ids = None
    self._set_ranges(cfg.ranges)

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command

  @property
  def iteration(self) -> int:
    return self._env.common_step_counter // self.cfg.num_steps_per_env

  def _set_ranges(self, ranges: CommandRanges) -> None:
    """``command_ranges`` plus ``_update_env_command_ranges``: clip to the terrain maxima."""
    self.ranges = ranges
    self.max_lin_vel = max(abs(v) for v in (*ranges.lin_vel_x, *ranges.lin_vel_y))
    env_ranges = {}
    for key in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw"):
      r = torch.tensor(getattr(ranges, key), device=self.device).repeat(self.num_envs, 1)
      if self.terrain_ids is not None:
        for tid, tmax in enumerate(self.cfg.terrain_max_ranges):
          m = self.terrain_ids == tid
          lo, hi = getattr(tmax, key)
          r[m, 0] = max(lo, getattr(ranges, key)[0])
          r[m, 1] = min(hi, getattr(ranges, key)[1])
      env_ranges[key] = r
    self.env_ranges = env_ranges

  # -- mjlab hooks -----------------------------------------------------------

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    self.max_move_distance[env_ids] = 0.0
    self.xy_accumulation[env_ids] = 0.0
    # mjlab clears episode_length_buf after the command reset; the reference resamples
    # with the cleared length.
    self._resample_ref(env_ids, torch.zeros_like(env_ids))
    return {"max_command_x": self.ranges.lin_vel_x[1]}

  def compute(self, dt: float) -> None:
    # Envs reset in this step were already handled by ``reset``; the reference only
    # starts counting on their next step.
    ep = self._env.episode_length_buf
    running = ep > 0
    dist = torch.norm(self.robot.data.root_link_pos_w[:, :2] - self._env.scene.env_origins[:, :2], dim=1)
    self.max_move_distance = torch.where(running, torch.maximum(self.max_move_distance, dist), self.max_move_distance)
    self.resampling_step -= running.float()
    ids = ((self.resampling_step <= 0) & (ep < self._env.max_episode_length - 1)).nonzero().flatten()
    self._resample_ref(ids, ep[ids])

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    raise NotImplementedError("Resampling is driven by ``_resample_ref``.")

  def _update_command(self) -> None:
    pass

  # -- reference resampling ----------------------------------------------------

  def _resample_ref(self, env_ids: torch.Tensor, ep_len: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    cfg, dev, dt = self.cfg, self.device, self._env.step_dt
    while self._pending and self.iteration >= self._pending[0][0]:
      self._set_ranges(self._pending.pop(0)[1])
    max_len = self._env.max_episode_length
    period = cfg.resampling_time_range[0]
    n = len(env_ids)
    cmd = self.vel_command
    rx, ry, ryaw = (self.env_ranges[k][env_ids] for k in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw"))

    remaining = torch.clip(0.625 * cfg.terrain_length - torch.norm(self.xy_accumulation[env_ids], dim=1) * period, min=0.0)
    self.resampling_step[env_ids] = period / dt
    if cfg.dynamic_resample:
      low = torch.clip(remaining / ((max_len - ep_len + 1e-9) * dt), min=0.0)
      cmd[env_ids, 0] = _sample_disjoint(low, rx[:, 0], rx[:, 1])
      cmd[env_ids, 1] = _sample_disjoint(low, ry[:, 0], ry[:, 1])
      cmd[env_ids, 2] = ryaw[:, 0] + torch.rand(n, device=dev) * (ryaw[:, 1] - ryaw[:, 0])
    else:
      for i, r in enumerate((rx, ry, ryaw)):
        cmd[env_ids, i] = r[:, 0] + torch.rand(n, device=dev) * (r[:, 1] - r[:, 0])
      cmd[env_ids, :2] *= (torch.norm(cmd[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    u = torch.rand(n, device=dev)
    lo = 0.0
    if cfg.limit_vel_prob > 0.0:
      lim = (u >= lo) & (u < lo + cfg.limit_vel_prob)
      lim_ids = env_ids[lim]
      change = lim
      if cfg.limit_vel_invert_when_continuous:
        was = lim & self.last_is_limit_vel[env_ids]
        cmd[env_ids[was], :3] *= -1.0
        change = lim & ~was
      ch_ids = env_ids[change]
      if len(ch_ids) > 0:
        comb = self.limit_vel_comb[torch.randint(0, len(self.limit_vel_comb), (len(ch_ids),), device=dev)]
        for i, r in enumerate((rx[change], ry[change], ryaw[change])):
          v = torch.where(comb[:, i] == -1, r[:, 0], r[:, 1])
          cmd[ch_ids, i] = torch.where(comb[:, i] == 0, torch.zeros_like(v), v)
      self.last_is_limit_vel[env_ids] = False
      self.last_is_limit_vel[lim_ids] = True
      lo += cfg.limit_vel_prob

    self.zero_command_proba = _schedule(*cfg.zero_command_curriculum, self.iteration)
    if self.zero_command_proba > 0.0:
      next_step = torch.clip(max_len - ep_len - remaining / (0.8 * self.max_lin_vel * dt + 1e-9), min=0.0, max=period / dt)
      zero = (u >= lo) & (u < lo + self.zero_command_proba) & (next_step > 0.0)
      z_ids = env_ids[zero]
      if len(z_ids) > 0:
        cmd[z_ids, :2] = 0.0
        self.resampling_step[z_ids] = next_step[zero]
        if cfg.limit_ang_vel_at_zero_command_prob > 0.0:
          add = torch.rand(len(z_ids), device=dev) < cfg.limit_ang_vel_at_zero_command_prob
          a_ids = z_ids[add]
          if len(a_ids) > 0:
            r = ryaw[zero][add]
            cmd[a_ids, 2] = torch.where(torch.rand(len(a_ids), device=dev) < 0.5, r[:, 0], r[:, 1])

    self.xy_accumulation[env_ids] += cmd[env_ids, :2]


def _sample_disjoint(bound: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
  """Uniform on [lo, -bound] U [bound, hi] (reference ``sample_disjoint_intervals``)."""
  w_neg = torch.relu(-bound - lo)
  w_pos = torch.relu(hi - bound)
  u = torch.rand_like(bound) * (w_neg + w_pos + 1e-6)
  return torch.where(u < w_neg, lo + u, hi - w_pos + (u - w_neg))
