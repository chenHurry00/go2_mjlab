"""Reference resets and pushes (go2_rl_gym ``_reset_root_states``, ``_reset_dofs``, ``_push_robots``)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _uniform(shape, lo: float, hi: float, device) -> torch.Tensor:
  return torch.empty(shape, device=device).uniform_(lo, hi)


def reset_root_state(env: ManagerBasedRlEnv, env_ids: torch.Tensor, xy_range: float = 1.0, vel_range: float = 0.5) -> None:
  """Spawn at the terrain origin + init pos, xy within +-1 m, random yaw, velocities U(-0.5, 0.5)."""
  robot = env.scene["robot"]
  n, dev = len(env_ids), env.device
  state = torch.zeros(n, 13, device=dev)
  state[:, :3] = robot.data.default_root_state[env_ids, :3] + env.scene.env_origins[env_ids]
  state[:, :2] += _uniform((n, 2), -xy_range, xy_range, dev)
  yaw = _uniform((n,), -math.pi, math.pi, dev)
  state[:, 3] = torch.cos(yaw / 2)
  state[:, 6] = torch.sin(yaw / 2)
  state[:, 7:13] = _uniform((n, 6), -vel_range, vel_range, dev)
  robot.write_root_state_to_sim(state, env_ids=env_ids)


def reset_joints(env: ManagerBasedRlEnv, env_ids: torch.Tensor, scale_range: tuple[float, float] = (0.5, 1.5)) -> None:
  """``dof_pos = default * U(0.5, 1.5)``, ``dof_vel = 0``."""
  robot = env.scene["robot"]
  q0 = robot.data.default_joint_pos[env_ids]
  robot.write_joint_state_to_sim(q0 * _uniform(q0.shape, *scale_range, env.device), torch.zeros_like(q0), env_ids=env_ids)


def push_by_episode_step(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, interval_steps: int = 201,
                         max_vel_xy: float = 0.4, max_ang_vel: float = 0.6) -> None:
  """Every ``interval_steps`` of an episode (including right after reset), replace the base
  xy linear velocity and the angular velocity with uniform samples."""
  ids = (env.episode_length_buf % interval_steps == 0).nonzero().flatten()
  if len(ids) == 0:
    return
  robot = env.scene["robot"]
  vel = robot.data.root_link_vel_w[ids].clone()
  vel[:, :2] = _uniform((len(ids), 2), -max_vel_xy, max_vel_xy, env.device)
  vel[:, 3:] = _uniform((len(ids), 3), -max_ang_vel, max_ang_vel, env.device)
  robot.write_root_link_velocity_to_sim(vel, env_ids=ids)
