"""Reference joint PD action (go2_rl_gym ``LeggedRobot.step`` / ``_compute_torques``).

Torques are computed every physics substep and applied through motor actuators:

  k ~ U{0..decimation} per policy step; substeps i < k still use the previous action
  tau = clip(kp*m_p*(a*scale + q0 - q + offset) - kd*m_d*qd, -limit, limit) * strength

PD multipliers, zero offsets and motor strengths are resampled per env on reset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class RefPdActionCfg(ActionTermCfg):
  entity_name: str = "robot"
  scale: float = 0.25
  stiffness: float = 20.0
  damping: float = 0.5
  torque_limits: dict[str, float] = field(
    default_factory=lambda: {".*_hip_joint": 23.7, ".*_thigh_joint": 23.7, ".*_calf_joint": 35.55}
  )
  randomize_delay: bool = True
  kp_range: tuple[float, float] = (0.9, 1.1)
  kd_range: tuple[float, float] = (0.9, 1.1)
  zero_offset_range: tuple[float, float] = (-0.035, 0.035)
  strength_range: tuple[float, float] = (0.8, 1.2)

  def build(self, env: ManagerBasedRlEnv) -> RefPdAction:
    return RefPdAction(self, env)


class RefPdAction(ActionTerm):
  cfg: RefPdActionCfg

  def __init__(self, cfg: RefPdActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    names = self._entity.joint_names
    _, _, limits = resolve_matching_names_values(cfg.torque_limits, names)
    self.torque_limits = torch.tensor(limits, device=self.device)
    self._decimation = env.cfg.decimation
    n, d = self.num_envs, len(names)
    zeros = lambda: torch.zeros(n, d, device=self.device)  # noqa: E731
    self._actions, self._last_actions, self.torques = zeros(), zeros(), zeros()
    self.kp_mult, self.kd_mult, self.zero_offset = zeros() + 1.0, zeros() + 1.0, zeros()
    self.strength = zeros() + 1.0
    self._delay = torch.zeros(n, 1, dtype=torch.long, device=self.device)
    self._substep = 0

  @property
  def action_dim(self) -> int:
    return self._actions.shape[1]

  @property
  def raw_action(self) -> torch.Tensor:
    return self._actions

  def process_actions(self, actions: torch.Tensor) -> None:
    self._last_actions[:] = self._actions
    self._actions[:] = actions
    if self.cfg.randomize_delay:
      self._delay = torch.randint(0, self._decimation + 1, (self.num_envs, 1), device=self.device)
    self._substep = 0

  def apply_actions(self) -> None:
    a = self._actions
    if self.cfg.randomize_delay:
      a = torch.where(self._substep >= self._delay, self._actions, self._last_actions)
    data = self._entity.data
    target = a * self.cfg.scale + data.default_joint_pos - data.joint_pos + self.zero_offset
    tau = self.cfg.stiffness * self.kp_mult * target - self.cfg.damping * self.kd_mult * data.joint_vel
    self.torques[:] = torch.clamp(tau, -self.torque_limits, self.torque_limits) * self.strength
    self._entity.set_joint_effort_target(self.torques)
    self._substep += 1

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._actions[env_ids] = 0.0
    self._last_actions[env_ids] = 0.0
    for buf, (lo, hi) in (
      (self.strength, self.cfg.strength_range),
      (self.zero_offset, self.cfg.zero_offset_range),
      (self.kp_mult, self.cfg.kp_range),
      (self.kd_mult, self.cfg.kd_range),
    ):
      buf[env_ids] = torch.empty_like(buf[env_ids]).uniform_(lo, hi)
