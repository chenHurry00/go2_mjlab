"""Reference terminations (go2_rl_gym ``check_termination``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def base_contact(env: ManagerBasedRlEnv, sensor_name: str = "base_contact", threshold: float = 1.0) -> torch.Tensor:
  force = env.scene[sensor_name].data.force
  assert force is not None
  return torch.any(torch.norm(force, dim=-1) > threshold, dim=1)


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  """``episode_length_buf > max_episode_length`` (one step later than mjlab's ``>=``)."""
  return env.episode_length_buf > env.max_episode_length
