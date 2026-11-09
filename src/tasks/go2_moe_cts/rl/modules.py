"""MoE-CTS networks (rsl_rl ``modules/utils.py`` and ``actor_critic_moe_cts.py``).

Module and parameter names match the reference so its checkpoints load with
``strict=True``. Unlike the reference, the model keeps no history buffer: callers
pass the frame-major history (oldest -> newest, newest = current observation).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

_ACTIVATIONS = {"elu": nn.ELU, "selu": nn.SELU, "relu": nn.ReLU, "lrelu": nn.LeakyReLU, "tanh": nn.Tanh}


class L2Norm(nn.Module):
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2.0, dim=-1)


class MLP(nn.Module):
  def __init__(self, dims: list[int], activation: str = "elu", last_activation: bool = False):
    super().__init__()
    layers: list[nn.Module] = []
    for i in range(len(dims) - 2):
      layers += [nn.Linear(dims[i], dims[i + 1]), _ACTIVATIONS[activation]()]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if last_activation:
      layers.append(_ACTIVATIONS[activation]())
    self.network = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.network(x)


class Experts(nn.Module):
  """Shared backbone, then one 1x1 grouped conv head per expert."""

  def __init__(self, expert_num: int, input_dim: int, backbone_hidden_dims: list[int], expert_hidden_dim: int,
               output_dim: int, activation: str = "elu"):
    super().__init__()
    self.expert_num, self.output_dim = expert_num, output_dim
    self.backbone = MLP([input_dim, *backbone_hidden_dims, expert_num * expert_hidden_dim], activation, last_activation=True)
    self.experts = nn.Conv1d(expert_num * expert_hidden_dim, expert_num * output_dim, kernel_size=1, groups=expert_num)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.experts(self.backbone(x).unsqueeze(-1)).squeeze(-1)
    return out.reshape(-1, self.expert_num, self.output_dim)


class MoE(nn.Module):
  def __init__(self, expert_num: int, input_dim: int, hidden_dims: list[int], output_dim: int, activation: str = "elu"):
    super().__init__()
    self.experts = Experts(expert_num, input_dim, hidden_dims[:-1], hidden_dims[-1], output_dim, activation)
    self.gating_network = nn.Sequential(MLP([input_dim, *hidden_dims[:-1], expert_num], activation), nn.Softmax(dim=-1))

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weights = self.gating_network(x)
    return torch.sum(weights.unsqueeze(-1) * self.experts(x), dim=1), weights


class StudentMoEEncoder(nn.Module):
  def __init__(self, expert_num: int, input_dim: int, hidden_dims: list[int], output_dim: int, activation: str = "elu"):
    super().__init__()
    self.norm_layer = L2Norm()
    self.moe = MoE(expert_num, input_dim, hidden_dims, output_dim, activation)

  def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    latent, weights = self.moe(history)
    return self.norm_layer(latent), weights


class ActorCriticMoECTS(nn.Module):
  def __init__(
    self,
    num_obs: int,
    num_critic_obs: int,
    num_actions: int,
    history_length: int,
    actor_hidden_dims: list[int],
    critic_hidden_dims: list[int],
    teacher_encoder_hidden_dims: list[int],
    student_encoder_hidden_dims: list[int],
    expert_num: int = 8,
    activation: str = "elu",
    init_noise_std: float = 1.0,
    latent_dim: int = 32,
  ):
    super().__init__()
    self.history_length = history_length
    self.teacher_encoder = nn.Sequential(MLP([num_critic_obs, *teacher_encoder_hidden_dims, latent_dim], activation), L2Norm())
    self.student_moe_encoder = StudentMoEEncoder(expert_num, num_obs * history_length, list(student_encoder_hidden_dims), latent_dim, activation)
    self.actor = MLP([latent_dim + num_obs, *actor_hidden_dims, num_actions], activation)
    self.critic = MLP([latent_dim + num_critic_obs, *critic_hidden_dims, 1], activation)
    self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
    self.distribution: Normal | None = None

  def _latent(self, privileged_obs: torch.Tensor, history: torch.Tensor, is_teacher: bool) -> torch.Tensor:
    return self.teacher_encoder(privileged_obs) if is_teacher else self.student_moe_encoder(history)[0]

  def act(self, obs: torch.Tensor, privileged_obs: torch.Tensor, history: torch.Tensor, is_teacher: bool) -> torch.Tensor:
    if is_teacher:
      latent = self.teacher_encoder(privileged_obs)
    else:
      with torch.no_grad():  # The student encoder is trained only by the latent loss.
        latent = self.student_moe_encoder(history)[0]
    mean = self.actor(torch.cat([latent, obs], dim=1))
    self.distribution = Normal(mean, mean * 0.0 + self.std)
    return self.distribution.sample()

  def evaluate(self, privileged_obs: torch.Tensor, history: torch.Tensor, is_teacher: bool) -> torch.Tensor:
    latent = self._latent(privileged_obs, history, is_teacher)
    return self.critic(torch.cat([latent.detach(), privileged_obs], dim=1))

  def act_inference(self, obs: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    latent, _ = self.student_moe_encoder(history)
    return self.actor(torch.cat([latent, obs], dim=1))

  def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.log_prob(actions).sum(dim=-1)

  @property
  def action_mean(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.mean

  @property
  def action_std(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.stddev

  @property
  def entropy(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.entropy().sum(dim=-1)


class JitPolicy(nn.Module):
  """Reference ``_TorchPolicyExporter.forward_moe_cts``: single 45-d frame in, internal
  5-frame history, returns ``(action, (gating weights, latent))``. Batch size 1."""

  def __init__(self, model: ActorCriticMoECTS, num_obs: int):
    super().__init__()
    import copy

    self.student_moe_encoder = copy.deepcopy(model.student_moe_encoder).cpu()
    self.actor = copy.deepcopy(model.actor).cpu()
    self.normalizer = nn.Identity()
    self.history = torch.zeros(1, model.history_length, num_obs)

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    x = self.normalizer(x)
    self.history = torch.cat([self.history[:, 1:], x.unsqueeze(1)], dim=1)
    latent, weights = self.student_moe_encoder(self.history.flatten(1))
    return self.actor(torch.cat([latent, x], dim=1)), (weights, latent)

  @torch.jit.export
  def reset(self) -> None:
    self.history = torch.zeros_like(self.history)
