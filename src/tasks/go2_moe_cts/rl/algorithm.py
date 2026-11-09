"""MoE-CTS PPO (rsl_rl ``algorithms/cts.py``, ``algorithms/moe_cts.py``,
``storage/rollout_storage_cts.py``).

Envs with ``i % 4 == 0`` act through the student encoder, the rest through the teacher.
Storage keeps teacher envs first; every minibatch holds a teacher part and a student part.
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn as nn
import torch.optim as optim

from .modules import ActorCriticMoECTS


class RolloutStorage:
  def __init__(self, num_envs: int, num_teacher: int, steps: int, obs_dim: int, priv_dim: int, hist_dim: int,
               act_dim: int, device: str):
    self.num_envs, self.num_teacher, self.steps, self.device = num_envs, num_teacher, steps, device
    z = lambda *s: torch.zeros(steps, num_envs, *s, device=device)  # noqa: E731
    self.obs, self.priv, self.history = z(obs_dim), z(priv_dim), z(hist_dim)
    self.actions, self.mu, self.sigma = z(act_dim), z(act_dim), z(act_dim)
    self.rewards, self.dones, self.values, self.log_prob = z(1), z(1), z(1), z(1)
    self.returns, self.advantages = z(1), z(1)
    self.step = 0

  def add(self, **t: torch.Tensor) -> None:
    for k, v in t.items():
      getattr(self, k)[self.step].copy_(v.view_as(getattr(self, k)[self.step]))
    self.step += 1

  def clear(self) -> None:
    self.step = 0

  def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float) -> None:
    advantage = 0
    for t in reversed(range(self.steps)):
      next_values = last_values if t == self.steps - 1 else self.values[t + 1]
      not_done = 1.0 - self.dones[t]
      delta = self.rewards[t] + not_done * gamma * next_values - self.values[t]
      advantage = delta + not_done * gamma * lam * advantage
      self.returns[t] = advantage + self.values[t]
    adv = self.returns - self.values
    self.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

  def mini_batches(self, num_mini_batches: int, num_epochs: int):
    """One permutation for all epochs, as the reference; teacher samples first."""
    nt = self.num_teacher * self.steps
    ns = (self.num_envs - self.num_teacher) * self.steps
    bt, bs = nt // num_mini_batches, ns // num_mini_batches
    t_idx = torch.randperm(nt, device=self.device)
    s_idx = nt + torch.randperm(ns, device=self.device)
    flat = {k: getattr(self, k).transpose(0, 1).flatten(0, 1) for k in
            ("obs", "priv", "actions", "history", "values", "advantages", "returns", "log_prob", "mu", "sigma")}
    for _ in range(num_epochs):
      for i in range(num_mini_batches):
        idx = torch.cat([t_idx[i * bt:(i + 1) * bt], s_idx[i * bs:(i + 1) * bs]])
        yield {k: v[idx] for k, v in flat.items()}


class MoECTS:
  def __init__(
    self,
    model: ActorCriticMoECTS,
    num_envs: int,
    num_learning_epochs: int = 5,
    num_mini_batches: int = 4,
    clip_param: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.95,
    value_loss_coef: float = 1.0,
    entropy_coef: float = 0.01,
    load_balance_coef: float = 0.01,
    learning_rate: float = 1e-3,
    student_encoder_learning_rate: float = 1e-3,
    max_grad_norm: float = 1.0,
    use_clipped_value_loss: bool = True,
    schedule: str = "adaptive",
    desired_kl: float = 0.01,
    teacher_env_ratio: float = 0.75,
    device: str = "cpu",
  ):
    if not 0.0 < teacher_env_ratio < 1.0:
      raise ValueError("teacher_env_ratio must lie strictly between 0 and 1.")
    stride = round(1 / (1 - teacher_env_ratio))
    if stride < 2 or not math.isclose(teacher_env_ratio, 1 - 1 / stride, abs_tol=1e-9):
      raise ValueError("teacher_env_ratio must equal 1 - 1/n for an integer n >= 2 (for example, 0.75).")
    self.teacher_env_ratio = teacher_env_ratio
    self.model, self.device = model.to(device), device
    self.learning_rate, self.schedule, self.desired_kl = learning_rate, schedule, desired_kl
    self.num_learning_epochs, self.num_mini_batches = num_learning_epochs, num_mini_batches
    self.clip_param, self.gamma, self.lam = clip_param, gamma, lam
    self.value_loss_coef, self.entropy_coef, self.load_balance_coef = value_loss_coef, entropy_coef, load_balance_coef
    self.max_grad_norm, self.use_clipped_value_loss = max_grad_norm, use_clipped_value_loss
    self.optimizer1 = optim.Adam(
      [{"params": model.teacher_encoder.parameters()}, {"params": model.critic.parameters()},
       {"params": model.actor.parameters()}, {"params": model.std}],
      lr=learning_rate,
    )
    self.optimizer2 = optim.Adam(model.student_moe_encoder.parameters(), lr=student_encoder_learning_rate)
    ids = torch.arange(num_envs, device=device)
    self.teacher_ids, self.student_ids = ids[ids % stride != 0], ids[ids % stride == 0]
    self.storage: RolloutStorage | None = None

  def init_storage(self, steps: int, obs_dim: int, priv_dim: int, hist_dim: int, act_dim: int) -> None:
    n = len(self.teacher_ids) + len(self.student_ids)
    self.storage = RolloutStorage(n, len(self.teacher_ids), steps, obs_dim, priv_dim, hist_dim, act_dim, self.device)

  def validate_training(self) -> None:
    """Reject empty or inconsistent training batches while allowing single-env inference."""
    assert self.storage is not None
    nt, ns = len(self.teacher_ids), len(self.student_ids)
    if nt == 0 or ns == 0:
      raise ValueError("MoE-CTS training requires nonempty teacher and student groups; increase env.scene.num_envs.")
    if not math.isclose(nt / (nt + ns), self.teacher_env_ratio, abs_tol=1e-9):
      raise ValueError("env.scene.num_envs must preserve the configured teacher_env_ratio exactly.")
    if self.storage.steps <= 0 or self.num_mini_batches <= 0 or self.num_learning_epochs <= 0:
      raise ValueError("Rollout steps, minibatch count and learning epochs must be positive.")
    if min(nt, ns) * self.storage.steps < self.num_mini_batches:
      raise ValueError("Each minibatch must contain both teacher and student samples; increase rollout size or reduce minibatches.")

  def _split(self, x: torch.Tensor) -> torch.Tensor:
    """Env order -> storage order (teacher envs first)."""
    return torch.cat([x[self.teacher_ids], x[self.student_ids]], dim=0)

  def act(self, obs: torch.Tensor, priv: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    outs = []
    for ids, teacher in ((self.teacher_ids, True), (self.student_ids, False)):
      a = self.model.act(obs[ids], priv[ids], history[ids], teacher).detach()
      outs.append((a, self.model.evaluate(priv[ids], history[ids], teacher).detach(),
                   self.model.get_actions_log_prob(a).detach(), self.model.action_mean.detach(), self.model.action_std.detach()))
    actions, values, log_prob, mu, sigma = (torch.cat(x, dim=0) for x in zip(*outs))
    self._transition = dict(obs=self._split(obs), priv=self._split(priv), history=self._split(history),
                            actions=actions, values=values, log_prob=log_prob, mu=mu, sigma=sigma)
    real = torch.zeros_like(actions)
    nt = len(self.teacher_ids)
    real[self.teacher_ids], real[self.student_ids] = actions[:nt], actions[nt:]
    return real

  def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor, time_outs: torch.Tensor | None) -> None:
    assert self.storage is not None
    rewards = self._split(rewards).clone()
    if time_outs is not None:  # Bootstrap on time-outs.
      rewards += self.gamma * self._transition["values"].squeeze(1) * self._split(time_outs).float()
    self.storage.add(rewards=rewards, dones=self._split(dones).float(), **self._transition)

  def compute_returns(self, priv: torch.Tensor, history: torch.Tensor) -> None:
    assert self.storage is not None
    last = torch.cat([self.model.evaluate(priv[self.teacher_ids], history[self.teacher_ids], True),
                      self.model.evaluate(priv[self.student_ids], history[self.student_ids], False)]).detach()
    self.storage.compute_returns(last, self.gamma, self.lam)

  def update(self) -> dict[str, float]:
    self.validate_training()
    assert self.storage is not None
    batches = list(self.storage.mini_batches(self.num_mini_batches, self.num_learning_epochs))
    nt = len(self.teacher_ids) * self.storage.steps // self.num_mini_batches
    m = self.model
    stats = dict(value_function=0.0, surrogate=0.0, entropy=0.0, latent=0.0, load_balance=0.0)
    for b in batches:
      parts = []
      for sl, teacher in ((slice(0, nt), True), (slice(nt, None), False)):
        m.act(b["obs"][sl], b["priv"][sl], b["history"][sl], teacher)
        parts.append((m.get_actions_log_prob(b["actions"][sl]), m.evaluate(b["priv"][sl], b["history"][sl], teacher),
                      m.action_mean, m.action_std, m.entropy))
      log_prob, value, mu, sigma, entropy = (torch.cat(x, dim=0) for x in zip(*parts))

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = torch.sum(torch.log(sigma / b["sigma"] + 1.0e-5)
                         + (torch.square(b["sigma"]) + torch.square(b["mu"] - mu)) / (2.0 * torch.square(sigma)) - 0.5, dim=-1).mean()
          if kl > self.desired_kl * 2.0:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
          elif 0.0 < kl < self.desired_kl / 2.0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          for g in self.optimizer1.param_groups:
            g["lr"] = self.learning_rate

      adv = torch.squeeze(b["advantages"])
      ratio = torch.exp(log_prob - torch.squeeze(b["log_prob"]))
      surr = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param))
      surrogate = surr[:nt].mean() + surr[nt:].mean()
      if self.use_clipped_value_loss:
        clipped = b["values"] + (value - b["values"]).clamp(-self.clip_param, self.clip_param)
        value_loss = torch.max((value - b["returns"]).pow(2), (clipped - b["returns"]).pow(2)).mean()
      else:
        value_loss = (b["returns"] - value).pow(2).mean()
      loss = surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
      self.optimizer1.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(itertools.chain.from_iterable(g["params"] for g in self.optimizer1.param_groups), self.max_grad_norm)
      self.optimizer1.step()
      stats["value_function"] += value_loss.item()
      stats["surrogate"] += surrogate.item()
      stats["entropy"] += entropy.mean().item()

    for b in batches:  # Student encoder: regress the teacher latent, balance expert usage.
      student, weights = m.student_moe_encoder(b["history"][nt:])
      with torch.no_grad():
        teacher = m.teacher_encoder(b["priv"][nt:])
      latent_loss = (teacher - student).pow(2).mean()
      usage = weights.mean(dim=0)
      load_balance = torch.mean((usage - 1.0 / weights.shape[1]).pow(2))
      self.optimizer2.zero_grad()
      (latent_loss + self.load_balance_coef * load_balance).backward()
      nn.utils.clip_grad_norm_(m.student_moe_encoder.parameters(), self.max_grad_norm)
      self.optimizer2.step()
      stats["latent"] += latent_loss.item()
      stats["load_balance"] += load_balance.item()

    self.storage.clear()
    return {k: v / len(batches) for k, v in stats.items()}
