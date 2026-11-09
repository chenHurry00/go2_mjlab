"""Reference observations (go2_rl_gym ``Go2Robot.compute_observations``).

Every term returns already-scaled values; noise is configured on the scaled values
with the reference magnitudes (``noise_scale * obs_scale``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.observation_manager import ObservationTermCfg

from .terrains import measured_heights

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

LIN_VEL, ANG_VEL, DOF_VEL, HEIGHT = 2.0, 0.25, 0.05, 2.5
COMMAND_SCALE = (LIN_VEL, LIN_VEL, ANG_VEL)


def base_lin_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["robot"].data.root_com_lin_vel_b * LIN_VEL


def base_ang_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["robot"].data.root_link_ang_vel_b * ANG_VEL


def projected_gravity(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["robot"].data.projected_gravity_b


def velocity_command(env: ManagerBasedRlEnv, command_name: str = "twist") -> torch.Tensor:
  cmd = env.command_manager.get_command(command_name)
  return cmd * torch.tensor(COMMAND_SCALE, device=env.device)


def dof_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
  data = env.scene["robot"].data
  return data.joint_pos - data.default_joint_pos


def dof_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["robot"].data.joint_vel * DOF_VEL


def actions(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.action_manager.action


def foot_forces(env: ManagerBasedRlEnv, sensor_name: str = "feet_contact") -> torch.Tensor:
  force = env.scene[sensor_name].data.force
  assert force is not None
  return torch.norm(force, dim=-1) * 1e-3


def normalized_torques(env: ManagerBasedRlEnv) -> torch.Tensor:
  term = env.action_manager.get_term("joint_pd")
  return term.torques / term.torque_limits  # type: ignore[attr-defined]


class dof_acc:
  """``(last_dof_vel - dof_vel) / dt * 1e-4``; ``last_dof_vel`` is cleared on reset.

  Advances once per env step, so extra ``compute()`` calls in the same step (e.g.
  ``get_observations``) return the same value.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    self.last = torch.zeros_like(env.scene["robot"].data.joint_vel)
    self.out = torch.zeros_like(self.last)
    self.step = -1

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    if env.common_step_counter != self.step:
      vel = env.scene["robot"].data.joint_vel
      self.out = (self.last - vel) / env.step_dt * 1e-4
      self.last = vel.clone()
      self.step = env.common_step_counter
    return self.out

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.last[slice(None) if env_ids is None else env_ids] = 0.0


def height_scan(env: ManagerBasedRlEnv) -> torch.Tensor:
  z = env.scene["robot"].data.root_link_pos_w[:, 2:3]
  return torch.clip(z - 0.5 - measured_heights(env), -1.0, 1.0) * HEIGHT
