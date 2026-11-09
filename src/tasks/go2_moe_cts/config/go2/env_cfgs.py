"""Unitree Go2 MoE-CTS environment, ported 1:1 from go2_rl_gym ``go2_moe_cts``.

Spec: the resolved ``config.yaml`` of the reference run (E0.4). Known deviations are
listed in ``docs/moe_cts/rewrite.md``.
"""

from __future__ import annotations

from functools import partial

import mujoco
from mjlab.actuator import BuiltinMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig

from src.assets.robots.unitree_go2.go2_constants import get_spec as get_go2_spec
from src.tasks.go2_moe_cts import mdp
from src.tasks.go2_moe_cts.mdp.rewards import FEET

TORQUE_LIMITS = {".*_hip_joint": 23.7, ".*_thigh_joint": 23.7, ".*_calf_joint": 35.55}  # URDF effort
MAX_STRENGTH = 1.2
OBS_CLIP = (-100.0, 100.0)
LEGS = ("FL", "FR", "RL", "RR")


# RoboGauge ``go2.xml`` narrows the thigh range (URDF: front [-1.5708, 3.4907], rear [-0.5236, 4.5379]).
RG_THIGH_RANGE = {"F": (-1.5708, 1.5708), "R": (-0.5236, 1.5708)}


def _go2_spec(thigh_cap: bool = False) -> mujoco.MjSpec:
  spec = get_go2_spec()
  for leg in LEGS:  # URDF thigh box is 0.11 m long (MJCF: 0.213 m).
    spec.geom(f"{leg}_thigh_collision").size[0] = 0.055
    if thigh_cap:
      joint = spec.joint(f"{leg}_thigh_joint")
      joint.range = list(RG_THIGH_RANGE[leg[0]])
      joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
  return spec


def go2_robot_cfg(thigh_cap: bool = False) -> EntityCfg:
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.42),
      joint_pos={
        ".*L_hip_joint": 0.1, ".*R_hip_joint": -0.1,
        "F[LR]_thigh_joint": 0.8, "R[LR]_thigh_joint": 1.0,
        ".*_calf_joint": -1.5,
      },
      joint_vel={".*": 0.0},
    ),
    # Self-collisions on (reference ``self_collisions: 0``). ``priority=1`` makes contacts
    # use the robot friction, randomised to U(0.5, 1.5) = PhysX average with ground mu 1.
    collisions=(CollisionCfg(geom_names_expr=(".*_collision",), condim=3, priority=1, contype=1, conaffinity=1),),
    spec_fn=partial(_go2_spec, thigh_cap),
    articulation=EntityArticulationInfoCfg(
      # Torques are computed by the action term; limits only guard strength-scaled torques.
      actuators=tuple(
        BuiltinMotorActuatorCfg(target_names_expr=(expr,), effort_limit=lim * MAX_STRENGTH, armature=0.0, frictionloss=0.0)
        for expr, lim in TORQUE_LIMITS.items()
      ),
      soft_joint_pos_limit_factor=0.9,
    ),
  )


def _contact(name: str, geoms: tuple[str, ...]) -> ContactSensorCfg:
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(mode="geom", pattern=geoms, entity="robot"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )


def _obs(func, noise: float | None = None, **params) -> ObservationTermCfg:
  return ObservationTermCfg(func=func, params=params, noise=Unoise(n_min=-noise, n_max=noise) if noise else None, clip=OBS_CLIP)


def _rew(func, weight: float, **params) -> RewardTermCfg:
  return RewardTermCfg(func=func, weight=weight, params=params)


def go2_moe_cts_env_cfg(play: bool = False, box_kinds: tuple[str, ...] = (), rg_align: bool = False) -> ManagerBasedRlEnvCfg:
  """``rg_align``: thigh range and contact solver as in the RoboGauge evaluation (elliptic cone, impratio 100)."""
  o, r = mdp.observations, mdp.rewards
  # Noise on scaled values: noise_scale * obs_scale.
  actor_terms = {
    "base_ang_vel": _obs(o.base_ang_vel, 0.2 * o.ANG_VEL),
    "projected_gravity": _obs(o.projected_gravity, 0.05),
    "velocity_command": _obs(o.velocity_command, command_name="twist"),
    "dof_pos": _obs(o.dof_pos, 0.01),
    "dof_vel": _obs(o.dof_vel, 1.5 * o.DOF_VEL),
    "actions": _obs(o.actions),
  }
  privileged_terms = {
    "base_lin_vel": _obs(o.base_lin_vel),
    "base_ang_vel": _obs(o.base_ang_vel),
    "projected_gravity": _obs(o.projected_gravity),
    "velocity_command": _obs(o.velocity_command, command_name="twist"),
    "dof_pos": _obs(o.dof_pos),
    "dof_vel": _obs(o.dof_vel),
    "actions": _obs(o.actions),
    "foot_forces": _obs(o.foot_forces),
    "torques": _obs(o.normalized_torques),
    "dof_acc": _obs(o.dof_acc),
    "height_scan": _obs(o.height_scan),
  }
  thighs = tuple(f"{leg}_thigh_collision" for leg in LEGS)
  calves = tuple(f"{leg}_calf{i}_collision" for leg in LEGS for i in (1, 2))

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=mdp.terrain_generator_cfg(box_kinds=box_kinds),
        max_init_terrain_level=5,
      ),
      entities={"robot": go2_robot_cfg(thigh_cap=rg_align)},
      sensors=(
        _contact("feet_contact", FEET),
        _contact("thigh_contact", thighs),
        _contact("calf_contact", calves),
        _contact("base_contact", ("base1_collision",)),  # Trunk only; the URDF head is a separate link.
      ),
      num_envs=1,
      extent=2.0,
    ),
    observations={
      "actor": ObservationGroupCfg(terms=actor_terms, enable_corruption=not play),
      "privileged": ObservationGroupCfg(terms=privileged_terms, enable_corruption=False),
    },
    actions={"joint_pd": mdp.RefPdActionCfg(torque_limits=TORQUE_LIMITS)},
    commands={"twist": mdp.RefVelocityCommandCfg()},
    events={
      "foot_friction": EventTermCfg(
        func=dr.geom_friction, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", geom_names=(".*_collision",)), "ranges": (0.5, 1.5),
                "operation": "abs", "shared_random": True},
      ),
      "base_mass": EventTermCfg(
        func=dr.body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)), "ranges": (-1.0, 1.0), "operation": "add"},
      ),
      "link_mass": EventTermCfg(
        func=dr.body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=(".*_hip", ".*_thigh", ".*_calf")), "ranges": (0.9, 1.1),
                "operation": "scale"},
      ),
      "base_com": EventTermCfg(
        func=dr.body_com_offset, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)), "ranges": (-0.03, 0.03), "operation": "add"},
      ),
      "reset_root": EventTermCfg(func=mdp.events.reset_root_state, mode="reset"),
      "reset_joints": EventTermCfg(func=mdp.events.reset_joints, mode="reset"),
      "push": EventTermCfg(func=mdp.events.push_by_episode_step, mode="step"),
    },
    rewards={
      "tracking_lin_vel": _rew(r.tracking_lin_vel, 1.0),
      "tracking_ang_vel": _rew(r.tracking_ang_vel, 0.5),
      "lin_vel_z": _rew(r.lin_vel_z, -2.0),
      "ang_vel_xy": _rew(r.ang_vel_xy, -0.05),
      "dof_acc": _rew(r.dof_acc, -2.5e-7),
      "dof_power": _rew(r.dof_power, -2.0e-5),
      "torques": _rew(r.torques, -1.0e-4),
      "correct_base_height": _rew(r.correct_base_height, -1.0),
      "action_rate": _rew(r.action_rate, -0.01),
      "action_smoothness": _rew(r.action_smoothness, -0.01),
      "collision": _rew(r.collision, -1.0),
      "dof_pos_limits": _rew(r.dof_pos_limits, -2.0),
      "feet_regulation": _rew(r.feet_regulation, -0.05),
      "hip_to_default": _rew(r.hip_to_default, -0.05),
    },
    terminations={
      "time_out": TerminationTermCfg(func=mdp.terminations.time_out, time_out=True),
      "base_contact": TerminationTermCfg(func=mdp.terminations.base_contact),
    },
    curriculum={
      "terrain_levels": CurriculumTermCfg(func=mdp.curriculums.terrain_levels),
      "lin_vel_z": CurriculumTermCfg(
        func=mdp.curriculums.reward_weight_schedule,
        params={"term_name": "lin_vel_z", "start_iter": 0, "end_iter": 1500, "start_value": 1.0, "end_value": 0.0},
      ),
      "base_height": CurriculumTermCfg(
        func=mdp.curriculums.reward_weight_schedule,
        params={"term_name": "correct_base_height", "start_iter": 0, "end_iter": 5000, "start_value": 1.0, "end_value": 10.0},
      ),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="base_link",
      distance=3.0, elevation=-5.0, azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=80,
      njmax=2000,
      contact_sensor_maxmatch=128,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20,
                       cone="elliptic" if rg_align else "pyramidal", impratio=100.0 if rg_align else 1.0),
    ),
    decimation=4,
    episode_length_s=25.0,
  )

  if play:
    cfg.scene.num_envs = 50
    cfg.episode_length_s = int(1e9)
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.curriculum = {}
    cfg.events.pop("push")
  return cfg
