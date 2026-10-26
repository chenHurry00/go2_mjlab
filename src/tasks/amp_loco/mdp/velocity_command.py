"""Velocity commands with dedicated, symmetric in-place turning samples."""

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class TurningVelocityCommand(UniformVelocityCommand):
  """Sample mutually exclusive standing, in-place turning, and walking commands."""

  cfg: "TurningVelocityCommandCfg"

  def __init__(self, cfg: "TurningVelocityCommandCfg", env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.is_turning_env = torch.zeros_like(self.is_standing_env)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

    # One categorical draw keeps the standing and turning fractions disjoint.
    mode = torch.rand(len(env_ids), device=self.device)
    standing = mode < self.cfg.rel_standing_envs
    turning = (mode >= self.cfg.rel_standing_envs) & (
      mode < self.cfg.rel_standing_envs + self.cfg.rel_turning_envs
    )
    self.is_standing_env[env_ids] = standing
    self.is_turning_env[env_ids] = turning
    self.is_heading_env[env_ids] = False
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = (
        torch.rand(len(env_ids), device=self.device) < self.cfg.rel_heading_envs
      ) & ~(standing | turning)

    turning_ids = env_ids[turning]
    speed = torch.empty(len(turning_ids), device=self.device).uniform_(
      *self.cfg.turning_velocity_range
    )
    direction = torch.where(
      torch.rand(len(turning_ids), device=self.device) < 0.5, -1.0, 1.0
    )
    self.vel_command_b[turning_ids, :2] = 0.0
    self.vel_command_b[turning_ids, 2] = speed * direction
    self.vel_command_b[env_ids[standing]] = 0.0

    # Apply optional initial velocities only after selecting the command mode.
    init_vel_ids = env_ids[
      torch.rand(len(env_ids), device=self.device) < self.cfg.init_velocity_prob
    ]
    if len(init_vel_ids) > 0:
      root_pos = self.robot.data.root_link_pos_w[init_vel_ids]
      root_quat = self.robot.data.root_link_quat_w[init_vel_ids]
      lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_ids].clone()
      lin_vel_b[:, :2] = self.vel_command_b[init_vel_ids, :2]
      root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
      root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_ids].clone()
      root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_ids, 2]
      root_state = torch.cat(
        [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
      )
      self.robot.write_root_state_to_sim(root_state, init_vel_ids)


@dataclass(kw_only=True)
class TurningVelocityCommandCfg(UniformVelocityCommandCfg):
  rel_turning_envs: float = 0.25
  """Fraction of all resampled commands dedicated to in-place turning."""
  turning_velocity_range: tuple[float, float] = (0.2, 1.0)
  """Positive yaw-rate magnitudes in rad/s; left/right signs are equiprobable."""

  def __post_init__(self):
    super().__post_init__()
    if not (
      0.0 <= self.rel_standing_envs <= 1.0
      and 0.0 <= self.rel_turning_envs <= 1.0
      and self.rel_standing_envs + self.rel_turning_envs <= 1.0
    ):
      raise ValueError("Standing and turning fractions must be nonnegative and sum to at most 1.")
    lower, upper = self.turning_velocity_range
    if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 < lower <= upper):
      raise ValueError("turning_velocity_range must contain finite positive ordered magnitudes.")
    if self.rel_turning_envs > 0.0 and upper > min(
      -self.ranges.ang_vel_z[0], self.ranges.ang_vel_z[1]
    ):
      raise ValueError("Turning magnitudes must fit both signs of ranges.ang_vel_z.")

  def build(self, env: "ManagerBasedRlEnv") -> TurningVelocityCommand:
    return TurningVelocityCommand(self, env)
