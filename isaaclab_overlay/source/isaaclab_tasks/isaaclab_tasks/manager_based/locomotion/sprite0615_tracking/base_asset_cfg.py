"""Base Sprite0615 articulation inherited by the Sprite0825 AMP task family."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


# These legacy paths are retained for the historical Sprite0615 tracking classes.
# The public G57-G58 Sprite0825 classes replace the USD path before simulation.
SPRITE0615_USD = (
    "/home/tony/sprite/sprite_isaaclab/assets/"
    "sprite0615_sanitized_v19_full_ankle_mirror/sprite0615.usd"
)
SPRITE0615_PM01_FOOT_USD = (
    "/home/tony/sprite/sprite_isaaclab/assets/"
    "sprite0615_sanitized_v24_pm01_style_foot/sprite0615.usd"
)


SPRITE0615_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=SPRITE0615_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.52),
        joint_pos={
            "left_hip_pitch_joint": -0.15,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.30,
            "left_ankle_pitch_joint": 0.15,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": 0.15,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": -0.30,
            "right_ankle_pitch_joint": 0.15,
            "right_ankle_roll_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.25,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": -0.60,
            "left_wrist_yaw_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -0.25,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.60,
            "right_wrist_yaw_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "head_pitch_joint": 0.0,
            "head_roll_joint": 0.0,
            "head_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_knee_joint"],
            effort_limit_sim=180.0,
            velocity_limit_sim=3.8,
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
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=90.0,
            velocity_limit_sim=3.8,
            stiffness=70.0,
            damping=5.0,
            armature=0.01,
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["waist_roll_joint", "waist_yaw_joint"],
            effort_limit_sim=60.0,
            stiffness=35.0,
            damping=3.0,
            armature=0.01,
        ),
        "arms_head": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*", "head_.*"],
            effort_limit_sim=60.0,
            stiffness=35.0,
            damping=4.0,
            armature=0.005,
        ),
    },
)
