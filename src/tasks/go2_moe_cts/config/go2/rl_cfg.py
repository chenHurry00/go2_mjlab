"""MoE-CTS runner configuration (reference ``GO2CfgMoECTS`` train config)."""

from dataclasses import dataclass, field

from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class MoECtsPolicyCfg:
  history_length: int = 5
  actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
  critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
  teacher_encoder_hidden_dims: tuple[int, ...] = (512, 256)
  student_encoder_hidden_dims: tuple[int, ...] = (512, 256, 256)
  expert_num: int = 8
  activation: str = "elu"
  init_noise_std: float = 1.0
  latent_dim: int = 32


@dataclass
class MoECtsAlgorithmCfg:
  clip_param: float = 0.2
  desired_kl: float = 0.01
  entropy_coef: float = 0.01
  gamma: float = 0.99
  lam: float = 0.95
  learning_rate: float = 1e-3
  student_encoder_learning_rate: float = 1e-3
  load_balance_coef: float = 0.01
  max_grad_norm: float = 1.0
  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  schedule: str = "adaptive"
  teacher_env_ratio: float = 0.75
  use_clipped_value_loss: bool = True
  value_loss_coef: float = 1.0


@dataclass
class MoECtsRunnerCfg(RslRlBaseRunnerCfg):
  actor_obs_group: str = "actor"
  privileged_obs_group: str = "privileged"
  policy: MoECtsPolicyCfg = field(default_factory=MoECtsPolicyCfg)
  algorithm: MoECtsAlgorithmCfg = field(default_factory=MoECtsAlgorithmCfg)


def go2_moe_cts_runner_cfg() -> MoECtsRunnerCfg:
  return MoECtsRunnerCfg(
    seed=42,
    num_steps_per_env=24,
    max_iterations=150_000,
    save_interval=500,
    experiment_name="go2_moe_cts",
    logger="tensorboard",
    clip_actions=100.0,
    upload_model=False,
  )
