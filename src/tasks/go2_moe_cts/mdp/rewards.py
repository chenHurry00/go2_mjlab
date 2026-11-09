"""Reference rewards (go2_rl_gym ``LeggedRobot._reward_*`` and ``Go2Robot``).

Raw terms only; weights (scaled by dt in the reward manager) live in the env config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.reward_manager import RewardTermCfg

from .terrains import base_scan_mask, measured_heights

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

FEET = ("FL_foot_collision", "FR_foot_collision", "RL_foot_collision", "RR_foot_collision")
HIP_IDS = (0, 3, 6, 9)
BASE_HEIGHT_TARGET = 0.38
TRACKING_SIGMA = 0.25
# ``rewards.dynamic_sigma``; max_sigma is indexed by reference terrain id.
MAX_SIGMA = (0.4166666666666667, 0.25, 0.25, 0.5, 0.5, 0.75, 1.0, 1.0, 0.25)
LIN_SIGMA_RANGE, ANG_SIGMA_RANGE = (0.5, 1.5), (1.0, 2.0)


def _robot(env: ManagerBasedRlEnv):
  return env.scene["robot"].data


def _cmd(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.command_manager.get_command("twist")


def _torques(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.action_manager.get_term("joint_pd").torques  # type: ignore[attr-defined]


def _dynamic_sigma(env: ManagerBasedRlEnv, target_abs: torch.Tensor, v_range: tuple[float, float]) -> torch.Tensor:
  """``_get_dynamic_sigma``: widen sigma for fast commands on hard terrain."""
  term = env.command_manager.get_term("twist")
  terrain = env.scene.terrain
  if term.terrain_ids is None or terrain is None:  # type: ignore[attr-defined]
    return torch.full_like(target_abs, TRACKING_SIGMA)
  v_min, v_max = v_range
  target = torch.tensor(MAX_SIGMA, device=env.device)[term.terrain_ids]  # type: ignore[attr-defined]
  ratio = ((target_abs - v_min) / (v_max - v_min)).clamp(0.0, 1.0)
  sigma = torch.where(target_abs >= v_min, TRACKING_SIGMA + ratio * (target - TRACKING_SIGMA), torch.full_like(target_abs, TRACKING_SIGMA))
  level_scale = torch.clamp(torch.exp((terrain.terrain_levels.float() + 1.0) / 10.0) - 1.0, max=1.0)
  return TRACKING_SIGMA + level_scale * (sigma - TRACKING_SIGMA)


def tracking_lin_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
  cmd = _cmd(env)
  err = torch.square(cmd[:, :2] - _robot(env).root_com_lin_vel_b[:, :2])
  sx = _dynamic_sigma(env, cmd[:, 0].abs(), LIN_SIGMA_RANGE)
  sy = _dynamic_sigma(env, cmd[:, 1].abs(), LIN_SIGMA_RANGE)
  return torch.exp(-(err[:, 0] / sx + err[:, 1] / sy))


def tracking_ang_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
  cmd = _cmd(env)
  err = torch.square(cmd[:, 2] - _robot(env).root_link_ang_vel_b[:, 2])
  return torch.exp(-err / _dynamic_sigma(env, cmd[:, 2].abs(), ANG_SIGMA_RANGE))


def lin_vel_z(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.square(_robot(env).root_com_lin_vel_b[:, 2])


def ang_vel_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.sum(torch.square(_robot(env).root_link_ang_vel_b[:, :2]), dim=1)


class dof_acc:
  """``sum(((last_dof_vel - dof_vel) / dt) ** 2)``; ``last_dof_vel`` is cleared on reset."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.last = torch.zeros_like(_robot(env).joint_vel)

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    vel = _robot(env).joint_vel
    out = torch.sum(torch.square((self.last - vel) / env.step_dt), dim=1)
    self.last = vel.clone()
    return out

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.last[slice(None) if env_ids is None else env_ids] = 0.0


def dof_power(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.sum(torch.abs(_torques(env) * _robot(env).joint_vel), dim=1)


def torques(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.sum(torch.square(_torques(env)), dim=1)


def _base_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Base z minus the mean terrain height of the 15 scan points under the trunk."""
  mask = base_scan_mask(env.device)
  ground = measured_heights(env)[:, mask].mean(dim=1)
  return _robot(env).root_link_pos_w[:, 2] - ground


def correct_base_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.square(_base_height(env) - BASE_HEIGHT_TARGET)


def action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  am = env.action_manager
  return torch.sum(torch.square(am.prev_action - am.action), dim=1)


def action_smoothness(env: ManagerBasedRlEnv) -> torch.Tensor:
  am = env.action_manager
  return torch.sum(torch.square(am.action - 2 * am.prev_action + am.prev_prev_action), dim=1)


def collision(env: ManagerBasedRlEnv, thigh_sensor: str = "thigh_contact", calf_sensor: str = "calf_contact") -> torch.Tensor:
  """Count of thigh and calf bodies with net contact force > 0.1 N (feet excluded)."""
  thigh = env.scene[thigh_sensor].data.force
  calf = env.scene[calf_sensor].data.force
  assert thigh is not None and calf is not None
  calf = calf.view(calf.shape[0], 4, -1, 3).sum(dim=2)  # calf1 + calf2 per leg.
  return (torch.norm(thigh, dim=-1) > 0.1).sum(dim=1) + (torch.norm(calf, dim=-1) > 0.1).sum(dim=1)


def dof_pos_limits(env: ManagerBasedRlEnv) -> torch.Tensor:
  data = _robot(env)
  lim = data.soft_joint_pos_limits
  out = -(data.joint_pos - lim[..., 0]).clip(max=0.0) + (data.joint_pos - lim[..., 1]).clip(min=0.0)
  return torch.sum(out, dim=1)


def feet_regulation(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalise foot xy speed near the ground; uses the base-height scan estimate.

  As in the reference, the world-frame foot offset is projected on the body-frame gravity.
  """
  data = _robot(env)
  ids = env.scene["robot"].find_geoms(FEET, preserve_order=True)[0]
  feet_pos = data.geom_pos_w[:, ids]
  feet_xy_vel = data.geom_lin_vel_w[:, ids, :2]
  feet2base = ((feet_pos - data.root_link_pos_w.unsqueeze(1)) * data.projected_gravity_b.unsqueeze(1)).sum(-1)
  feet_height = torch.clamp(_base_height(env).unsqueeze(1) - feet2base, min=0.0)
  return (feet_xy_vel.pow(2).sum(-1) * torch.exp(-feet_height / (0.025 * BASE_HEIGHT_TARGET))).sum(-1)


def hip_to_default(env: ManagerBasedRlEnv) -> torch.Tensor:
  data = _robot(env)
  ids = list(HIP_IDS)
  return torch.sum(torch.abs(data.joint_pos[:, ids] - data.default_joint_pos[:, ids]), dim=1)
