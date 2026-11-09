"""MoE-CTS on-policy runner (rsl_rl ``runners/on_policy_runner_cts.py``).

Keeps the frame-major 5-frame history of actor observations: cleared when an episode
ends, newest frame = current (noisy) observation. Checkpoints use the reference keys.
"""

from __future__ import annotations

import os
import statistics
import time
from collections import defaultdict, deque

import torch
from mjlab.rl import RslRlVecEnvWrapper
from torch.utils.tensorboard import SummaryWriter

from .algorithm import MoECTS
from .modules import ActorCriticMoECTS, JitPolicy


class MoECtsRunner:
  def __init__(self, env: RslRlVecEnvWrapper, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
      raise ValueError("MoE-CTS supports one process; select a single GPU with --gpu-ids.")
    self.env, self.cfg, self.log_dir, self.device = env, train_cfg, log_dir, device
    obs = env.get_observations()
    self.num_obs = obs[train_cfg["actor_obs_group"]].shape[1]
    num_priv = obs[train_cfg["privileged_obs_group"]].shape[1]
    hist = train_cfg["policy"]["history_length"]
    policy = {k: v for k, v in train_cfg["policy"].items() if k != "history_length"}
    self.model = ActorCriticMoECTS(self.num_obs, num_priv, env.num_actions, hist, **policy).to(device)
    self.alg = MoECTS(self.model, env.num_envs, device=device, **train_cfg["algorithm"])
    self.steps = train_cfg["num_steps_per_env"]
    self.alg.init_storage(self.steps, self.num_obs, num_priv, self.num_obs * hist, env.num_actions)
    self.history = torch.zeros(env.num_envs, hist, self.num_obs, device=device)
    self.current_learning_iteration = 0
    self.writer: SummaryWriter | None = None
    self._git_paths: list[str] = []

  def _obs(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
    return obs[self.cfg["actor_obs_group"]].to(self.device), obs[self.cfg["privileged_obs_group"]].to(self.device)

  def _push_history(self, obs: torch.Tensor, dones: torch.Tensor | None = None) -> None:
    if dones is not None:
      self.history[dones > 0] = 0.0
    self.history = torch.cat([self.history[:, 1:], obs.unsqueeze(1)], dim=1)

  def add_git_repo_to_log(self, path: str) -> None:
    self._git_paths.append(path)

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    self.alg.validate_training()
    from ..mdp.curriculums import reward_weight_schedule

    env_cfg = self.env.unwrapped.cfg
    for name, command_cfg in env_cfg.commands.items():
      if hasattr(command_cfg, "num_steps_per_env") and command_cfg.num_steps_per_env != self.steps:
        raise ValueError(f"Command '{name}' num_steps_per_env must match agent.num_steps_per_env ({self.steps}).")
    for name, curriculum_cfg in env_cfg.curriculum.items():
      if curriculum_cfg.func is reward_weight_schedule and curriculum_cfg.params.get("num_steps_per_env", 24) != self.steps:
        raise ValueError(f"Curriculum '{name}' num_steps_per_env must match agent.num_steps_per_env ({self.steps}).")
    if self.log_dir is not None and self.writer is None:
      self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
    start_it = self.current_learning_iteration
    # Resample commands and reward schedules at the restored iteration before the first observation.
    self.env.unwrapped.common_step_counter = start_it * self.steps
    if start_it > 0:
      self.env.reset()
      self.history.zero_()
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
    obs, priv = self._obs(self.env.get_observations())
    self._push_history(obs)
    self.model.train()
    t_ids = self.alg.teacher_ids
    is_teacher = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
    is_teacher[t_ids] = True
    buffers = {k: deque(maxlen=100) for k in ("teacher_rew", "teacher_len", "student_rew", "student_len")}
    cur_rew = torch.zeros(self.env.num_envs, device=self.device)
    cur_len = torch.zeros(self.env.num_envs, device=self.device)
    for it in range(start_it, start_it + num_learning_iterations):
      t0 = time.time()
      ep_logs: dict[str, list[float]] = defaultdict(list)
      with torch.inference_mode():
        for _ in range(self.steps):
          actions = self.alg.act(obs, priv, self.history.flatten(1))
          obs_dict, rewards, dones, extras = self.env.step(actions)
          obs, priv = self._obs(obs_dict)
          rewards, dones = rewards.to(self.device), dones.to(self.device)
          self._push_history(obs, dones)
          self.alg.process_env_step(rewards, dones, extras.get("time_outs"))
          for k, v in extras.get("log", {}).items():
            ep_logs[k].append(float(v))
          cur_rew += rewards
          cur_len += 1
          done = dones > 0
          if done.any():
            for tag, m in (("teacher", done & is_teacher), ("student", done & ~is_teacher)):
              buffers[f"{tag}_rew"].extend(cur_rew[m].tolist())
              buffers[f"{tag}_len"].extend(cur_len[m].tolist())
            cur_rew[done] = 0.0
            cur_len[done] = 0.0
        collect_time = time.time() - t0
        t1 = time.time()
        self.alg.compute_returns(priv, self.history.flatten(1))
      losses = self.alg.update()
      learn_time = time.time() - t1
      self.current_learning_iteration = it + 1
      self._log(it, losses, ep_logs, buffers, collect_time, learn_time)
      if self.log_dir is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
    if self.log_dir is not None:
      self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

  def _log(self, it, losses, ep_logs, buffers, collect_time, learn_time) -> None:
    fps = int(self.steps * self.env.num_envs / (collect_time + learn_time))
    scalars = {f"Loss/{k}": v for k, v in losses.items()}
    scalars["Loss/learning_rate"] = self.alg.learning_rate
    scalars["Policy/mean_noise_std"] = self.model.std.mean().item()
    scalars["Perf/total_fps"] = fps
    scalars["Perf/collection_time"] = collect_time
    scalars["Perf/learning_time"] = learn_time
    for tag in ("teacher", "student"):
      if buffers[f"{tag}_len"]:
        scalars[f"Train/mean_{tag}_reward"] = statistics.mean(buffers[f"{tag}_rew"])
        scalars[f"Train/mean_{tag}_episode_length"] = statistics.mean(buffers[f"{tag}_len"])
    for k, v in ep_logs.items():
      scalars[k if "/" in k else f"Episode/{k}"] = statistics.mean(v)
    if self.writer is not None:
      for k, v in scalars.items():
        self.writer.add_scalar(k, v, it)
    keys = ("Train/mean_teacher_episode_length", "Train/mean_student_episode_length", "Curriculum/terrain_levels/all",
            "Episode_Reward/tracking_lin_vel", "Loss/latent", "Loss/learning_rate")
    line = " ".join(f"{k.split('/')[-1]}={scalars[k]:.4g}" for k in keys if k in scalars)
    print(f"[it {it}] fps={fps} {line}", flush=True)

  def save(self, path: str, infos=None) -> None:
    torch.save({
      "model_state_dict": self.model.state_dict(),
      "optimizer1_state_dict": self.alg.optimizer1.state_dict(),
      "optimizer2_state_dict": self.alg.optimizer2.state_dict(),
      "iter": self.current_learning_iteration,
      "learning_rate": self.alg.learning_rate,
      "infos": infos,
    }, path)
    self.export_jit(os.path.join(os.path.dirname(path), "jit_models", os.path.basename(path).replace("model_", "policy_jit_")))

  def load(self, path: str, load_optimizer: bool = True) -> dict:
    ck = torch.load(path, map_location=self.device, weights_only=False)
    self.model.load_state_dict(ck["model_state_dict"])
    if load_optimizer and "optimizer1_state_dict" in ck:
      self.alg.optimizer1.load_state_dict(ck["optimizer1_state_dict"])
      self.alg.optimizer2.load_state_dict(ck["optimizer2_state_dict"])
      self.alg.learning_rate = ck.get("learning_rate", self.alg.learning_rate)
    self.current_learning_iteration = ck["iter"]
    return ck.get("infos") or {}

  def export_jit(self, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.jit.script(JitPolicy(self.model, self.num_obs)).save(path)

  def get_inference_policy(self, device: str | None = None):
    """Student policy for play: keeps its own history, cleared on ``reset(dones)``."""
    self.model.eval()
    history = torch.zeros_like(self.history)

    def policy(obs_dict) -> torch.Tensor:
      nonlocal history
      obs = obs_dict[self.cfg["actor_obs_group"]].to(self.device)
      history = torch.cat([history[:, 1:], obs.unsqueeze(1)], dim=1)
      return self.model.act_inference(obs, history.flatten(1))

    def reset(dones: torch.Tensor | None = None) -> None:
      if dones is None:
        history.zero_()
      else:
        history[dones > 0] = 0.0

    policy.reset = reset  # type: ignore[attr-defined]
    return policy
