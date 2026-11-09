from mjlab.tasks.registry import register_mjlab_task

from src.tasks.go2_moe_cts.rl import MoECtsRunner

from .env_cfgs import go2_moe_cts_env_cfg
from .rl_cfg import go2_moe_cts_runner_cfg

register_mjlab_task(
  task_id="Unitree-Go2-MoE-CTS",
  env_cfg=go2_moe_cts_env_cfg(),
  play_env_cfg=go2_moe_cts_env_cfg(play=True),
  rl_cfg=go2_moe_cts_runner_cfg(),
  runner_cls=MoECtsRunner,
)

# Stairs collide as boxes with vertical walls (reference mesh with ``slope_threshold`` 1.5).
# Validation task: BoxStairs plus the RoboGauge thigh range and contact solver (see docs/moe_cts/rewrite.md).
register_mjlab_task(
  task_id="Unitree-Go2-MoE-CTS-RGAlign",
  env_cfg=go2_moe_cts_env_cfg(box_kinds=("stairs_up", "stairs_down"), rg_align=True),
  play_env_cfg=go2_moe_cts_env_cfg(play=True, box_kinds=("stairs_up", "stairs_down"), rg_align=True),
  rl_cfg=go2_moe_cts_runner_cfg(),
  runner_cls=MoECtsRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-MoE-CTS-BoxStairs",
  env_cfg=go2_moe_cts_env_cfg(box_kinds=("stairs_up", "stairs_down")),
  play_env_cfg=go2_moe_cts_env_cfg(play=True, box_kinds=("stairs_up", "stairs_down")),
  rl_cfg=go2_moe_cts_runner_cfg(),
  runner_cls=MoECtsRunner,
)
