"""Reference curricula: terrain levels (``_update_terrain_curriculum``) and reward
weight schedules (``rewards.curriculum_rewards``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .commands import _schedule

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


TERRAIN_NAMES = {0: "wave", 1: "slope", 2: "rough_slope", 3: "stairs_up", 4: "stairs_down", 5: "obstacles", 8: "flat"}


def terrain_levels(env: ManagerBasedRlEnv, env_ids: torch.Tensor, terrain_length: float = 8.0) -> dict[str, torch.Tensor]:
  """Up when the robot got farther than half a tile from its origin; down when it covered
  less than half of the commanded distance (non-zero-command share)."""
  terrain = env.scene.terrain
  assert terrain is not None
  if env.common_step_counter > 0:  # Reference skips the call before ``init_done``.
    cmd = env.command_manager.get_term("twist")
    robot = env.scene["robot"]
    # The reference updates the max distance before resets, include this step's position.
    dist = torch.norm(robot.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    dist = torch.maximum(cmd.max_move_distance[env_ids], dist)  # type: ignore[attr-defined]
    move_up = dist > terrain_length / 2
    commanded = torch.norm(cmd.xy_accumulation[env_ids], dim=1) * cmd.cfg.resampling_time_range[0]  # type: ignore[attr-defined]
    move_down = (dist < commanded * (1 - cmd.zero_command_proba) * 0.5) & ~move_up  # type: ignore[attr-defined]
    terrain.update_env_origins(env_ids, move_up, move_down)
  # Logged as ``Curriculum/terrain_levels/<key>``, per type like the reference ``terrain_level_<name>``.
  levels = terrain.terrain_levels.float()
  out = {"all": levels.mean()}
  ids = env.command_manager.get_term("twist").terrain_ids  # type: ignore[attr-defined]
  for tid, name in TERRAIN_NAMES.items():
    m = ids == tid
    if m.any():
      out[name] = levels[m].mean()
  return out


class reward_weight_schedule:
  """Multiply a reward weight by a value interpolated over training iterations."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    name = cfg.params["term_name"]
    self.base = env.reward_manager.get_term_cfg(name).weight

  def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, term_name: str, start_iter: int, end_iter: int,
               start_value: float, end_value: float, num_steps_per_env: int = 24) -> torch.Tensor:
    scale = _schedule(start_iter, end_iter, start_value, end_value, env.common_step_counter // num_steps_per_env)
    env.reward_manager.get_term_cfg(term_name).weight = self.base * scale
    return torch.tensor(scale)
