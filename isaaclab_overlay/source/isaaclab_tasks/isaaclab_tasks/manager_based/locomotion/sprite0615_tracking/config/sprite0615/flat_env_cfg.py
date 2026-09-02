from __future__ import annotations

import copy
import math
from pathlib import Path

from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.sprite0615_tracking import mdp
from isaaclab_tasks.manager_based.locomotion.sprite0615_tracking.tracking_env_cfg import TrackingEnvCfg, VELOCITY_RANGE
from isaaclab_tasks.manager_based.locomotion.velocity import mdp as velocity_mdp
from isaaclab_tasks.manager_based.locomotion.sprite0615_tracking.coupled_ankle_actuator import (
    CoupledAnkleContinuousDCMotorCfg,
    CoupledAnkleDCMotorCfg,
)
from isaaclab_tasks.manager_based.locomotion.sprite0615_tracking.base_asset_cfg import (
    SPRITE0615_CFG,
    SPRITE0615_PM01_FOOT_USD,
)


SPRITE0615_TRACKING_MOTION = (
    "/home/tony/sprite/sprite_isaaclab/assets/sprite0615_references/"
    "sprite0615_bmlrub058_0008_normal_walk4_isaac_pm01foot_candidate_zplus045mm.npz"
)


SPRITE0615_TRACKING_MOTION_STRIDE085 = (
    "/home/tony/sprite/sprite_isaaclab/assets/sprite0615_references/"
    "sprite0615_bmlrub058_0008_normal_walk4_isaac_pm01foot_candidate_zplus045mm_stride085.npz"
)

SPRITE0615_TRACKING_MOTION_STRIDE075 = (
    "/home/tony/sprite/sprite_isaaclab/assets/sprite0615_references/"
    "sprite0615_bmlrub058_0008_normal_walk4_isaac_pm01foot_candidate_zplus045mm_stride075.npz"
)


SPRITE0615_TRACKING_MOTION_CONSISTENT075 = (
    "/home/tony/sprite/sprite_isaaclab/assets/sprite0615_references/"
    "sprite0615_bmlrub058_0008_normal_walk4_isaac_pm01foot_candidate_zplus045mm_stride075_leg075_consistent_zplus015.npz"
)


SPRITE0615_TRACKING_MOTION_CONSISTENT075_TILE8 = (
    "/home/tony/sprite/sprite_isaaclab/assets/sprite0615_references/"
    "sprite0615_bmlrub058_0008_normal_walk4_isaac_pm01foot_candidate_zplus045mm_stride075_leg075_consistent_zplus015_tile8.npz"
)


SPRITE0615_TRACKING_MOTION_CONSISTENT075_TIMESCALE150 = (
    "/home/tony/sprite/sprite_isaaclab/assets/sprite0615_references/"
    "sprite0615_bmlrub058_walk4_consistent075_timescale150.npz"
)


SPRITE0615_TRACKING_MOTION_CONSISTENT075_TIMESCALE150_TILE6 = (
    "/home/tony/sprite/sprite_isaaclab/assets/sprite0615_references/"
    "sprite0615_bmlrub058_walk4_consistent075_timescale150_tile6.npz"
)


SPRITE0615_TRACKING_MOTION_TIMESCALE150_PHASECYCLE_B_TILE12 = (
    "/home/tony/sprite/sprite_isaaclab/assets/sprite0615_references/"
    "sprite0615_walk_timescale150_phasecycle_b_75_214_tile12.npz"
)

SPRITE0825_PM01_NATIVE_G55_STRAIGHT_MOTION = (
    "/home/tony/sprite/sprite_isaaclab/assets/sprite0615_references/stage2_amp/"
    "sprite0825_pm01_native_straight_vx045_wbt_fk_headingcanon_v1.npz"
)

SPRITE0825_PM01_NATIVE_G56_PHASE_BALANCED_MOTION = (
    "/home/tony/sprite/sprite_isaaclab/assets/sprite0615_references/stage2_amp/"
    "sprite0825_pm01_native_straight_vx045_wbt_fk_headingcanon_phasebalanced_v1.npz"
)


SPRITE0615_WBT_CFG = copy.deepcopy(SPRITE0615_CFG)
SPRITE0615_WBT_CFG.spawn.usd_path = SPRITE0615_PM01_FOOT_USD

# DM-J4340P-2EC at the robot's approximately 38 V bus voltage. The 40 Nm
# value is the instantaneous peak envelope; 14 Nm remains a separate
# continuous/thermal validation limit and is not modeled as a hard clip here.
SPRITE0615_WBT_J4340_CFG = copy.deepcopy(SPRITE0615_WBT_CFG)
SPRITE0615_WBT_J4340_CFG.actuators["legs"] = DCMotorCfg(
    joint_names_expr=[".*_hip_.*", ".*_knee_joint"],
    effort_limit=40.0,
    effort_limit_sim=40.0,
    saturation_effort=40.0,
    velocity_limit=9.3,
    velocity_limit_sim=9.3,
    stiffness={
        ".*_hip_pitch_joint": 110.0,
        ".*_hip_roll_joint": 80.0,
        ".*_hip_yaw_joint": 55.0,
        ".*_knee_joint": 130.0,
    },
    damping={
        ".*_hip_pitch_joint": 6.0,
        ".*_hip_roll_joint": 5.0,
        ".*_hip_yaw_joint": 3.5,
        ".*_knee_joint": 6.0,
    },
    armature=0.01,
)

SPRITE0615_WBT_MOTOR_ENVELOPE_CFG = copy.deepcopy(SPRITE0615_WBT_J4340_CFG)
SPRITE0615_WBT_MOTOR_ENVELOPE_CFG.actuators["feet"] = CoupledAnkleDCMotorCfg(
    joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
    effort_limit=12.5,
    effort_limit_sim=25.0,
    saturation_effort=12.5,
    velocity_limit=36.2,
    velocity_limit_sim=36.2,
    stiffness=70.0,
    damping=5.0,
    armature=0.01,
)


# PM01 follows a 10 Hz, damping-ratio 2 impedance design from reflected motor
# inertia. This branch uses the same derivation with Sprite0615's actual motors.
_PM01_NATURAL_FREQUENCY = 2.0 * math.pi * 10.0
_PM01_DAMPING_RATIO = 2.0
_J4340_REFLECTED_INERTIA = 2.00e-5 * 40.0**2
_J4310_ANKLE_REFLECTED_INERTIA = 2.0 * 1.80e-5 * 10.0**2
_J4340_STIFFNESS = _J4340_REFLECTED_INERTIA * _PM01_NATURAL_FREQUENCY**2
_J4340_DAMPING = (
    2.0 * _PM01_DAMPING_RATIO * _J4340_REFLECTED_INERTIA * _PM01_NATURAL_FREQUENCY
)
_J4310_ANKLE_STIFFNESS = _J4310_ANKLE_REFLECTED_INERTIA * _PM01_NATURAL_FREQUENCY**2
_J4310_ANKLE_DAMPING = (
    2.0 * _PM01_DAMPING_RATIO * _J4310_ANKLE_REFLECTED_INERTIA * _PM01_NATURAL_FREQUENCY
)

SPRITE0615_WBT_PM01_RATED_CFG = copy.deepcopy(SPRITE0615_WBT_CFG)
SPRITE0615_WBT_PM01_RATED_CFG.actuators["legs"] = DCMotorCfg(
    joint_names_expr=[".*_hip_.*", ".*_knee_joint"],
    effort_limit=14.0,
    effort_limit_sim=14.0,
    saturation_effort=40.0,
    velocity_limit=9.3,
    velocity_limit_sim=9.3,
    stiffness=_J4340_STIFFNESS,
    damping=_J4340_DAMPING,
    armature=_J4340_REFLECTED_INERTIA,
)
SPRITE0615_WBT_PM01_RATED_CFG.actuators["feet"] = CoupledAnkleContinuousDCMotorCfg(
    joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
    effort_limit=3.5,
    effort_limit_sim=7.0,
    saturation_effort=12.5,
    velocity_limit=36.2,
    velocity_limit_sim=36.2,
    stiffness=_J4310_ANKLE_STIFFNESS,
    damping=_J4310_ANKLE_DAMPING,
    armature=_J4310_ANKLE_REFLECTED_INERTIA,
)


# Diagnostic-only upper-body envelope from the authoritative Sprite0615 URDF.
# This deliberately keeps the existing impedance gains and action scales so an
# evaluation isolates the effect of replacing the inherited 60 Nm simulation
# limits. Actual upper-body motor data must replace these values before the
# final sim-to-real configuration is frozen.
SPRITE0615_WBT_UPPER_DECLARED_CFG = copy.deepcopy(SPRITE0615_WBT_PM01_RATED_CFG)
SPRITE0615_WBT_UPPER_DECLARED_CFG.actuators["torso"] = ImplicitActuatorCfg(
    joint_names_expr=["waist_roll_joint", "waist_yaw_joint"],
    effort_limit_sim={"waist_roll_joint": 30.0, "waist_yaw_joint": 9.0},
    velocity_limit_sim=3.8,
    stiffness=35.0,
    damping=3.0,
    armature=0.01,
)
del SPRITE0615_WBT_UPPER_DECLARED_CFG.actuators["arms_head"]
SPRITE0615_WBT_UPPER_DECLARED_CFG.actuators["arms_proximal"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_yaw_joint"],
    effort_limit_sim=3.0,
    velocity_limit_sim=3.8,
    stiffness=35.0,
    damping=4.0,
    armature=0.005,
)
SPRITE0615_WBT_UPPER_DECLARED_CFG.actuators["wrists_head"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_roll_joint", "head_.*"],
    effort_limit_sim=0.8,
    velocity_limit_sim=3.8,
    stiffness=35.0,
    damping=4.0,
    armature=0.005,
)


# Full motor torque-speed envelope with the existing upper-body impedance.
# This is the second isolated audit stage: it verifies physical clipping before
# PM01-derived gains and action scales are introduced as a separate change.
_J3507_NO_LOAD_RAD_S_38V = (460.0 + (38.0 - 24.0) / 24.0 * (900.0 - 460.0)) * 2.0 * math.pi / 60.0
SPRITE0615_WBT_FULL_ENVELOPE_CFG = copy.deepcopy(SPRITE0615_WBT_PM01_RATED_CFG)
del SPRITE0615_WBT_FULL_ENVELOPE_CFG.actuators["torso"]
del SPRITE0615_WBT_FULL_ENVELOPE_CFG.actuators["arms_head"]
SPRITE0615_WBT_FULL_ENVELOPE_CFG.actuators["waist_roll_j6248"] = DCMotorCfg(
    joint_names_expr=["waist_roll_joint"],
    effort_limit=30.0,
    effort_limit_sim=30.0,
    saturation_effort=97.0,
    velocity_limit=2.0 * math.pi,
    velocity_limit_sim=2.0 * math.pi,
    stiffness=35.0,
    damping=3.0,
    armature=0.01,
)
SPRITE0615_WBT_FULL_ENVELOPE_CFG.actuators["waist_yaw_j4340"] = DCMotorCfg(
    joint_names_expr=["waist_yaw_joint"],
    effort_limit=14.0,
    effort_limit_sim=14.0,
    saturation_effort=40.0,
    velocity_limit=9.3,
    velocity_limit_sim=9.3,
    stiffness=35.0,
    damping=3.0,
    armature=0.01,
)
SPRITE0615_WBT_FULL_ENVELOPE_CFG.actuators["arms_j4310"] = DCMotorCfg(
    joint_names_expr=[".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_yaw_joint"],
    effort_limit=3.5,
    effort_limit_sim=3.5,
    saturation_effort=12.5,
    velocity_limit=36.2,
    velocity_limit_sim=36.2,
    stiffness=35.0,
    damping=4.0,
    armature=0.005,
)
SPRITE0615_WBT_FULL_ENVELOPE_CFG.actuators["wrists_head_j3507"] = DCMotorCfg(
    joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_roll_joint", "head_.*"],
    effort_limit=0.8,
    effort_limit_sim=0.8,
    saturation_effort=3.0,
    velocity_limit=_J3507_NO_LOAD_RAD_S_38V,
    velocity_limit_sim=_J3507_NO_LOAD_RAD_S_38V,
    stiffness=35.0,
    damping=4.0,
    armature=0.005,
)


# V31 group-isolation variants. Each starts from the stable V30 implicit-
# actuator envelope and changes exactly one upper-body motor family to
# DCMotorCfg, allowing actuator-class semantics to be diagnosed without PPO.
SPRITE0615_WBT_ARMS_DC_AUDIT_CFG = copy.deepcopy(SPRITE0615_WBT_UPPER_DECLARED_CFG)
SPRITE0615_WBT_ARMS_DC_AUDIT_CFG.actuators["arms_proximal"] = copy.deepcopy(
    SPRITE0615_WBT_FULL_ENVELOPE_CFG.actuators["arms_j4310"]
)

SPRITE0615_WBT_TORSO_DC_AUDIT_CFG = copy.deepcopy(SPRITE0615_WBT_UPPER_DECLARED_CFG)
del SPRITE0615_WBT_TORSO_DC_AUDIT_CFG.actuators["torso"]
SPRITE0615_WBT_TORSO_DC_AUDIT_CFG.actuators["waist_roll_j6248"] = copy.deepcopy(
    SPRITE0615_WBT_FULL_ENVELOPE_CFG.actuators["waist_roll_j6248"]
)
SPRITE0615_WBT_TORSO_DC_AUDIT_CFG.actuators["waist_yaw_j4340"] = copy.deepcopy(
    SPRITE0615_WBT_FULL_ENVELOPE_CFG.actuators["waist_yaw_j4340"]
)

SPRITE0615_WBT_DISTAL_DC_AUDIT_CFG = copy.deepcopy(SPRITE0615_WBT_UPPER_DECLARED_CFG)
SPRITE0615_WBT_DISTAL_DC_AUDIT_CFG.actuators["wrists_head"] = copy.deepcopy(
    SPRITE0615_WBT_FULL_ENVELOPE_CFG.actuators["wrists_head_j3507"]
)


# Clean PM01-style actuator branch: every simulated joint uses PhysX implicit
# drives, peak torque is the instantaneous limit, and rated torque remains a
# rollout/thermal acceptance gate. J3507 and J6248 armatures are provisional
# values until their rotor inertia is confirmed from hardware documentation.
_J4310_SINGLE_REFLECTED_INERTIA = 1.80e-5 * 10.0**2
_J3507_PROVISIONAL_REFLECTED_INERTIA = 0.005
_J6248_PROVISIONAL_REFLECTED_INERTIA = 0.01


def _pm01_stiffness(armature: float) -> float:
    return armature * _PM01_NATURAL_FREQUENCY**2


def _pm01_damping(armature: float) -> float:
    return 2.0 * _PM01_DAMPING_RATIO * armature * _PM01_NATURAL_FREQUENCY


_J4310_SINGLE_STIFFNESS = _pm01_stiffness(_J4310_SINGLE_REFLECTED_INERTIA)
_J4310_SINGLE_DAMPING = _pm01_damping(_J4310_SINGLE_REFLECTED_INERTIA)
_J3507_PROVISIONAL_STIFFNESS = _pm01_stiffness(_J3507_PROVISIONAL_REFLECTED_INERTIA)
_J3507_PROVISIONAL_DAMPING = _pm01_damping(_J3507_PROVISIONAL_REFLECTED_INERTIA)
_J6248_PROVISIONAL_STIFFNESS = _pm01_stiffness(_J6248_PROVISIONAL_REFLECTED_INERTIA)
_J6248_PROVISIONAL_DAMPING = _pm01_damping(_J6248_PROVISIONAL_REFLECTED_INERTIA)

SPRITE0615_WBT_PM01_FULL_PEAK_CFG = copy.deepcopy(SPRITE0615_WBT_CFG)
SPRITE0615_WBT_PM01_FULL_PEAK_CFG.actuators = {
    "legs_j4340": ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_.*", ".*_knee_joint"],
        effort_limit_sim=40.0,
        velocity_limit_sim=9.3,
        stiffness=_J4340_STIFFNESS,
        damping=_J4340_DAMPING,
        armature=_J4340_REFLECTED_INERTIA,
    ),
    "feet_j4310_pair": ImplicitActuatorCfg(
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        effort_limit_sim=25.0,
        velocity_limit_sim=36.2,
        stiffness=_J4310_ANKLE_STIFFNESS,
        damping=_J4310_ANKLE_DAMPING,
        armature=_J4310_ANKLE_REFLECTED_INERTIA,
    ),
    "waist_roll_j6248": ImplicitActuatorCfg(
        joint_names_expr=["waist_roll_joint"],
        effort_limit_sim=97.0,
        velocity_limit_sim=2.0 * math.pi,
        stiffness=_J6248_PROVISIONAL_STIFFNESS,
        damping=_J6248_PROVISIONAL_DAMPING,
        armature=_J6248_PROVISIONAL_REFLECTED_INERTIA,
    ),
    "waist_yaw_j4340": ImplicitActuatorCfg(
        joint_names_expr=["waist_yaw_joint"],
        effort_limit_sim=40.0,
        velocity_limit_sim=9.3,
        stiffness=_J4340_STIFFNESS,
        damping=_J4340_DAMPING,
        armature=_J4340_REFLECTED_INERTIA,
    ),
    "arms_j4310": ImplicitActuatorCfg(
        joint_names_expr=[".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_yaw_joint"],
        effort_limit_sim=12.5,
        velocity_limit_sim=36.2,
        stiffness=_J4310_SINGLE_STIFFNESS,
        damping=_J4310_SINGLE_DAMPING,
        armature=_J4310_SINGLE_REFLECTED_INERTIA,
    ),
    "wrists_head_j3507": ImplicitActuatorCfg(
        joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_roll_joint", "head_.*"],
        effort_limit_sim=3.0,
        velocity_limit_sim=_J3507_NO_LOAD_RAD_S_38V,
        stiffness=_J3507_PROVISIONAL_STIFFNESS,
        damping=_J3507_PROVISIONAL_DAMPING,
        armature=_J3507_PROVISIONAL_REFLECTED_INERTIA,
    ),
}

SPRITE0615_PM01_FULL_PEAK_ACTION_SCALE = {
    ".*_hip_.*": 0.25 * 40.0 / _J4340_STIFFNESS,
    ".*_knee_joint": 0.25 * 40.0 / _J4340_STIFFNESS,
    ".*_ankle_pitch_joint": 0.25 * 25.0 / _J4310_ANKLE_STIFFNESS,
    ".*_ankle_roll_joint": 0.25 * 25.0 / _J4310_ANKLE_STIFFNESS,
    "waist_roll_joint": 0.25 * 97.0 / _J6248_PROVISIONAL_STIFFNESS,
    "waist_yaw_joint": 0.25 * 40.0 / _J4340_STIFFNESS,
    ".*_shoulder_.*": 0.25 * 12.5 / _J4310_SINGLE_STIFFNESS,
    ".*_elbow_joint": 0.25 * 12.5 / _J4310_SINGLE_STIFFNESS,
    ".*_wrist_yaw_joint": 0.25 * 12.5 / _J4310_SINGLE_STIFFNESS,
    ".*_wrist_pitch_joint": 0.25 * 3.0 / _J3507_PROVISIONAL_STIFFNESS,
    ".*_wrist_roll_joint": 0.25 * 3.0 / _J3507_PROVISIONAL_STIFFNESS,
    "head_.*": 0.25 * 3.0 / _J3507_PROVISIONAL_STIFFNESS,
}


SPRITE0615_ACTION_SCALE = {
    ".*_hip_pitch_joint": 0.20,
    ".*_hip_roll_joint": 0.18,
    ".*_hip_yaw_joint": 0.16,
    ".*_knee_joint": 0.25,
    ".*_ankle_pitch_joint": 0.16,
    ".*_ankle_roll_joint": 0.14,
    "waist_roll_joint": 0.12,
    "waist_yaw_joint": 0.12,
    ".*_shoulder_.*": 0.25,
    ".*_elbow_joint": 0.25,
    ".*_wrist_.*": 0.18,
    "head_.*": 0.10,
}


# EngineAI's action scale is 0.25 * effort_limit / stiffness. For the
# differential ankle, two motors contribute to a pure pitch or roll command.
SPRITE0615_PM01_RATED_ACTION_SCALE = copy.deepcopy(SPRITE0615_ACTION_SCALE)
SPRITE0615_PM01_RATED_ACTION_SCALE.update(
    {
        ".*_hip_pitch_joint": 0.25 * 40.0 / _J4340_STIFFNESS,
        ".*_hip_roll_joint": 0.25 * 40.0 / _J4340_STIFFNESS,
        ".*_hip_yaw_joint": 0.25 * 40.0 / _J4340_STIFFNESS,
        ".*_knee_joint": 0.25 * 40.0 / _J4340_STIFFNESS,
        ".*_ankle_pitch_joint": 0.25 * (2.0 * 3.5) / _J4310_ANKLE_STIFFNESS,
        ".*_ankle_roll_joint": 0.25 * (2.0 * 3.5) / _J4310_ANKLE_STIFFNESS,
    }
)


SPRITE0615_TRACKED_BODIES = [
    "pelvis_link",
    "left_hip_roll_link",
    "left_knee_link",
    "left_foot_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_foot_link",
    "waist_yaw_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_link",
]


@configclass
class Sprite0615WbtFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SPRITE0615_WBT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = SPRITE0615_ACTION_SCALE
        self.commands.motion.motion_file = SPRITE0615_TRACKING_MOTION
        self.commands.motion.anchor_body_name = "pelvis_link"
        self.commands.motion.body_names = SPRITE0615_TRACKED_BODIES
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="pelvis_link")
        self.terminations.ee_body_pos.params["body_names"] = [
            "left_foot_link",
            "right_foot_link",
            "left_wrist_roll_link",
            "right_wrist_roll_link",
        ]
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[
                r"^(?!left_foot_link$)(?!right_foot_link$)(?!left_wrist_roll_link$)(?!right_wrist_roll_link$).+$"
            ],
        )


@configclass
class Sprite0615WbtFlatWoStateEstimationEnvCfg(Sprite0615WbtFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None

@configclass
class Sprite0615WbtFlatStride085WoStateEstimationEnvCfg(Sprite0615WbtFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = SPRITE0615_TRACKING_MOTION_STRIDE085


@configclass
class Sprite0615WbtFlatStride075WoStateEstimationEnvCfg(Sprite0615WbtFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = SPRITE0615_TRACKING_MOTION_STRIDE075

@configclass
class Sprite0615WbtFlatConsistent075WoStateEstimationEnvCfg(Sprite0615WbtFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = SPRITE0615_TRACKING_MOTION_CONSISTENT075

@configclass
class Sprite0615WbtFlatConsistent075CleanPlayEnvCfg(Sprite0615WbtFlatConsistent075WoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.debug_vis = False
        self.scene.contact_forces.debug_vis = False
        self.events.push_robot = None

@configclass
class Sprite0615WbtFlatConsistent075Tile8CleanPlayEnvCfg(Sprite0615WbtFlatConsistent075CleanPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = SPRITE0615_TRACKING_MOTION_CONSISTENT075_TILE8
        self.episode_length_s = 30.0


@configclass
class Sprite0615WbtFlatTimescale150J4340WoStateEstimationEnvCfg(
    Sprite0615WbtFlatWoStateEstimationEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SPRITE0615_WBT_J4340_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.commands.motion.motion_file = SPRITE0615_TRACKING_MOTION_CONSISTENT075_TIMESCALE150


@configclass
class Sprite0615WbtFlatTimescale150MotorEnvelopeWoStateEstimationEnvCfg(
    Sprite0615WbtFlatWoStateEstimationEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SPRITE0615_WBT_MOTOR_ENVELOPE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.commands.motion.motion_file = SPRITE0615_TRACKING_MOTION_CONSISTENT075_TIMESCALE150


@configclass
class Sprite0615WbtFlatTimescale150MotorEnvelopeCleanPlayEnvCfg(
    Sprite0615WbtFlatTimescale150MotorEnvelopeWoStateEstimationEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = SPRITE0615_TRACKING_MOTION_CONSISTENT075_TIMESCALE150_TILE6
        self.commands.motion.debug_vis = False
        self.scene.contact_forces.debug_vis = False
        self.events.push_robot = None
        self.episode_length_s = 30.0


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedWoStateEstimationEnvCfg(
    Sprite0615WbtFlatWoStateEstimationEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SPRITE0615_WBT_PM01_RATED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = SPRITE0615_PM01_RATED_ACTION_SCALE
        self.commands.motion.motion_file = SPRITE0615_TRACKING_MOTION_CONSISTENT075_TIMESCALE150


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedCleanPlayEnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedWoStateEstimationEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = SPRITE0615_TRACKING_MOTION_CONSISTENT075_TIMESCALE150_TILE6
        self.commands.motion.debug_vis = False
        self.scene.contact_forces.debug_vis = False
        self.events.push_robot = None
        self.episode_length_s = 30.0


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBWoStateEstimationEnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedWoStateEstimationEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = SPRITE0615_TRACKING_MOTION_TIMESCALE150_PHASECYCLE_B_TILE12
        # Keep every 10-second training episode inside the continuous reference.
        self.commands.motion.sampling_end_margin_steps = 500


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBCleanPlayEnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBWoStateEstimationEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.start_at_zero = True
        self.commands.motion.debug_vis = False
        self.scene.contact_forces.debug_vis = False
        self.events.push_robot = None
        self.episode_length_s = 30.0


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015WoStateEstimationEnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBWoStateEstimationEnvCfg
):
    """PhaseCycleB with a modestly stronger standard action-rate penalty."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2.weight = -0.15


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth020WoStateEstimationEnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBWoStateEstimationEnvCfg
):
    """PhaseCycleB with the stronger action-rate A/B setting."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2.weight = -0.20


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedEnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015WoStateEstimationEnvCfg
):
    """V14A dynamics and rewards with a smoothly variable reference speed."""

    def __post_init__(self):
        super().__post_init__()
        old = self.commands.motion
        self.commands.motion = mdp.VariableSpeedMotionCommandCfg(
            asset_name=old.asset_name,
            resampling_time_range=old.resampling_time_range,
            debug_vis=old.debug_vis,
            motion_file=old.motion_file,
            anchor_body_name=old.anchor_body_name,
            body_names=old.body_names,
            pose_range=old.pose_range,
            velocity_range=old.velocity_range,
            joint_position_range=old.joint_position_range,
            adaptive_kernel_size=old.adaptive_kernel_size,
            adaptive_lambda=old.adaptive_lambda,
            adaptive_uniform_ratio=old.adaptive_uniform_ratio,
            adaptive_alpha=old.adaptive_alpha,
            start_at_zero=old.start_at_zero,
            sampling_end_margin_steps=650,
            anchor_visualizer_cfg=old.anchor_visualizer_cfg,
            body_visualizer_cfg=old.body_visualizer_cfg,
            speed_scale_range=(0.80, 1.15),
            segment_time_range=(2.0, 5.0),
            speed_ramp_rate=0.20,
        )


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedUpperDeclaredAuditEnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedEnvCfg
):
    """V15 task with the URDF-declared upper-body actuator envelope."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SPRITE0615_WBT_UPPER_DECLARED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullEnvelopeAuditEnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedEnvCfg
):
    """V15 task with motor-specific torque-speed envelopes on all joints."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SPRITE0615_WBT_FULL_ENVELOPE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedArmsDcAuditEnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SPRITE0615_WBT_ARMS_DC_AUDIT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedTorsoDcAuditEnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SPRITE0615_WBT_TORSO_DC_AUDIT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedDistalDcAuditEnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SPRITE0615_WBT_DISTAL_DC_AUDIT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedPm01FullPeakEnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedEnvCfg
):
    """Whole-body PM01 implicit drives with motor peak envelopes."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SPRITE0615_WBT_PM01_FULL_PEAK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = SPRITE0615_PM01_FULL_PEAK_ACTION_SCALE


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakActuatorLegacyScaleEnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedPm01FullPeakEnvCfg
):
    """Diagnostic: FullPeak actuators with the frozen policy's legacy action scale."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = SPRITE0615_PM01_RATED_ACTION_SCALE


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedRatedActuatorFullPeakScaleEnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedEnvCfg
):
    """Diagnostic: legacy rated actuators with the FullPeak action scale."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = SPRITE0615_PM01_FULL_PEAK_ACTION_SCALE


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakLegacyScaleCleanTrainingEnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakActuatorLegacyScaleEnvCfg
):
    """Clean consolidation after changing only the physical actuator model."""

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = None


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakLegacyScaleAnkleTorque1e5CleanEnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakLegacyScaleCleanTrainingEnvCfg
):
    """Hardware adaptation using Isaac Lab's standard torque regularizer on the differential ankles."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.ankle_torques_l2 = RewTerm(
            func=mdp.joint_torques_l2,
            weight=-1.0e-5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                )
            },
        )


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakLegacyScaleAnkleTorque1e3EnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakActuatorLegacyScaleEnvCfg
):
    """Ankle torque adaptation with a term comparable to ten percent of action-rate regularization."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.ankle_torques_l2 = RewTerm(
            func=mdp.joint_torques_l2,
            weight=-1.0e-3,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                )
            },
        )


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakLegacyScaleAnkleTorque1e3CleanEnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakLegacyScaleAnkleTorque1e3EnvCfg
):
    """Clean-training variant of the ankle torque adaptation task."""

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = None


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakLegacyScaleAnkleTorque1e3PushStage1EnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakLegacyScaleAnkleTorque1e3EnvCfg
):
    """V35 hardware adaptation with sparse half-envelope recovery pushes."""

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot.interval_range_s = (2.5, 4.0)
        self.events.push_robot.params["velocity_range"] = _scaled_push_velocity_range(0.50)


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakLegacyScaleAnkleTorque1e3PushStage1FeetSlide010EnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakLegacyScaleAnkleTorque1e3PushStage1EnvCfg
):
    """V38: retain push transfer while gently reducing stance-foot slip."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_slide = RewTerm(
            func=velocity_mdp.feet_slide,
            weight=-0.10,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=["left_foot_link", "right_foot_link"]
                ),
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["left_foot_link", "right_foot_link"]
                ),
            },
        )


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakLegacyScalePushStage1EnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakActuatorLegacyScaleEnvCfg
):
    """FullPeak actuator migration with sparse half-envelope pushes."""

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot.interval_range_s = (2.5, 4.0)
        self.events.push_robot.params["velocity_range"] = _scaled_push_velocity_range(0.50)


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakLegacyScalePushStage2EnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakActuatorLegacyScaleEnvCfg
):
    """FullPeak actuator migration with 75% of the upstream push envelope."""

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot.interval_range_s = (1.5, 3.0)
        self.events.push_robot.params["velocity_range"] = _scaled_push_velocity_range(0.75)


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakLegacyScalePushStage3EnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakActuatorLegacyScaleEnvCfg
):
    """FullPeak actuator migration with the complete upstream push envelope."""

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot.interval_range_s = (1.0, 3.0)
        self.events.push_robot.params["velocity_range"] = _scaled_push_velocity_range(1.00)


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedCleanTrainingEnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedEnvCfg
):
    """Clean hard-phase consolidation with the standard task otherwise unchanged."""

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = None


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedMixedPushEnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedEnvCfg
):
    """Train clean and pushed trajectories together using upstream pushes on half the events."""

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot.func = mdp.push_by_setting_velocity_subset
        self.events.push_robot.params["pushed_fraction"] = 0.50


def _scaled_push_velocity_range(scale: float) -> dict[str, tuple[float, float]]:
    """Scale BeyondMimic's original push envelope without changing its axes."""

    return {
        axis: (lower * scale, upper * scale)
        for axis, (lower, upper) in VELOCITY_RANGE.items()
    }


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedPushStage1EnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedEnvCfg
):
    """Recovery consolidation with sparse, half-envelope BeyondMimic pushes."""

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot.interval_range_s = (2.5, 4.0)
        self.events.push_robot.params["velocity_range"] = _scaled_push_velocity_range(0.50)


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedPushStage2EnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedEnvCfg
):
    """Intermediate recovery stage with 75% of the upstream push envelope."""

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot.interval_range_s = (1.5, 3.0)
        self.events.push_robot.params["velocity_range"] = _scaled_push_velocity_range(0.75)


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedPushStage3EnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedEnvCfg
):
    """Final stage using BeyondMimic's complete push envelope and cadence."""

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot.interval_range_s = (1.0, 3.0)
        self.events.push_robot.params["velocity_range"] = _scaled_push_velocity_range(1.00)


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedRecoveryStage1EnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedEnvCfg
):
    """Frequent upstream-envelope pushes for recovery-focused continuation."""

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot.interval_range_s = (0.75, 2.0)
        self.events.push_robot.params["velocity_range"] = _scaled_push_velocity_range(1.00)


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedRecoveryStage2EnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedEnvCfg
):
    """Second recovery stage with a 10% larger velocity envelope."""

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot.interval_range_s = (0.75, 1.75)
        self.events.push_robot.params["velocity_range"] = _scaled_push_velocity_range(1.10)


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedRecoveryStage3EnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedEnvCfg
):
    """Final recovery stage with a 20% larger velocity envelope."""

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot.interval_range_s = (0.75, 1.5)
        self.events.push_robot.params["velocity_range"] = _scaled_push_velocity_range(1.20)


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedEndpointRecoveryEnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedRecoveryStage1EnvCfg
):
    """RecoveryStage1 with a conservative four-end-effector position term."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.motion_end_effector_pos = RewTerm(
            func=mdp.motion_relative_body_position_error_exp,
            weight=0.25,
            params={
                "command_name": "motion",
                "std": 0.15,
                "body_names": [
                    "left_foot_link",
                    "right_foot_link",
                    "left_wrist_roll_link",
                    "right_wrist_roll_link",
                ],
            },
        )


@configclass
class Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFreeArmRecoveryEnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedRecoveryStage1EnvCfg
):
    """Recovery curriculum that lets the arms complete balance corrections."""

    def __post_init__(self):
        super().__post_init__()
        self.terminations.ee_body_pos.params["body_names"] = [
            "left_foot_link",
            "right_foot_link",
        ]


@configclass
class Sprite0825WbtPm01LocomotionCleanEnvCfg(
    Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakLegacyScaleAnkleTorque1e3PushStage1FeetSlide010EnvCfg
):
    """Clean Sprite0825 WBT adaptation on PM01's continuous locomotion clip."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.usd_path = str(
            Path.home()
            / "sprite/sprite_isaaclab/assets/sprite0825_sanitized_v4/sprite0825.usd"
        )
        self.commands.motion.motion_file = str(
            Path.home()
            / "sprite/sprite_isaaclab/assets/sprite0615_references/stage2_amp/"
            "sprite0615_pm01_locomotion_axis_default_mapped_v2.npz"
        )
        # An eight-second episode can start as late as 13.6 s and still remain
        # inside the 23.04 s clip at the maximum 1.15 playback rate. This gives
        # random starts access to initial stand, start, walk, stop, and final stand
        # without an end-of-clip reference jump.
        self.episode_length_s = 8.0
        self.commands.motion.sampling_end_margin_steps = 470
        self.commands.motion.start_at_zero = False
        self.events.push_robot = None


@configclass
class Sprite0825WbtPm01LocomotionFkV1CleanEnvCfg(
    Sprite0825WbtPm01LocomotionCleanEnvCfg
):
    """Sprite0825 WBT task using URDF-FK-rebuilt PM01 body targets."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = str(
            Path.home()
            / "sprite/sprite_isaaclab/assets/sprite0615_references/stage2_amp/"
            "sprite0825_pm01_locomotion_wbt_fk_v1.npz"
        )


@configclass
class Sprite0825WbtPm01LocomotionFkV1CleanPlayEnvCfg(
    Sprite0825WbtPm01LocomotionFkV1CleanEnvCfg
):
    """Deterministic full-clip review of the G54b FKv1 tracking task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 22.5
        self.commands.motion.start_at_zero = True
        self.commands.motion.sampling_end_margin_steps = 0
        self.commands.motion.speed_scale_range = (1.0, 1.0)
        self.commands.motion.debug_vis = False
        for event_name in ("physics_material", "add_joint_default_pos", "base_com"):
            if hasattr(self.events, event_name):
                setattr(self.events, event_name, None)


@configclass
class Sprite0825WbtPm01NativeStraightG55CleanEnvCfg(
    Sprite0825WbtPm01LocomotionFkV1CleanEnvCfg
):
    """Isolated G55 adaptation on the audited native PM01 straight clip."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = SPRITE0825_PM01_NATIVE_G55_STRAIGHT_MOTION
        # At the maximum 1.15 playback rate, 470 source frames cover a full
        # eight-second rollout and keep every random window inside the clip.
        self.episode_length_s = 8.0
        self.commands.motion.sampling_end_margin_steps = 470
        self.commands.motion.start_at_zero = False
        self.events.push_robot = None


@configclass
class Sprite0825WbtPm01NativeStraightG55CleanPlayEnvCfg(
    Sprite0825WbtPm01NativeStraightG55CleanEnvCfg
):
    """Deterministic full-clip policy review for the isolated G55 task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 20.5
        self.commands.motion.start_at_zero = True
        self.commands.motion.sampling_end_margin_steps = 0
        self.commands.motion.speed_scale_range = (1.0, 1.0)
        self.commands.motion.debug_vis = False
        for event_name in ("physics_material", "add_joint_default_pos", "base_com"):
            if hasattr(self.events, event_name):
                setattr(self.events, event_name, None)


@configclass
class Sprite0825WbtPm01NativeStraightG56PhaseBalancedCleanEnvCfg(
    Sprite0825WbtPm01NativeStraightG55CleanEnvCfg
):
    """G56 continuation with stationary endpoint padding for phase balance."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = SPRITE0825_PM01_NATIVE_G56_PHASE_BALANCED_MOTION


@configclass
class Sprite0825WbtPm01NativeStraightG56PhaseBalancedCleanPlayEnvCfg(
    Sprite0825WbtPm01NativeStraightG56PhaseBalancedCleanEnvCfg
):
    """Deterministic full-clip review for the G56 phase-balanced task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 32.5
        self.commands.motion.start_at_zero = True
        self.commands.motion.sampling_end_margin_steps = 0
        self.commands.motion.speed_scale_range = (1.0, 1.0)
        self.commands.motion.debug_vis = False
        for event_name in ("physics_material", "add_joint_default_pos", "base_com"):
            if hasattr(self.events, event_name):
                setattr(self.events, event_name, None)
