import gymnasium as gym

from . import agents, flat_env_cfg


gym.register(
    id="Tracking-Flat-Sprite0615-WBT-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Sprite0615WbtFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


gym.register(
    id=(
        "Tracking-Flat-Sprite0615-WBT-Timescale150-PM01Rated-PhaseCycleB-"
        "Smooth015-VariableSpeed-FreeArmRecovery-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFreeArmRecoveryEnvCfg
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-Sprite0615-WBT-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Sprite0615WbtFlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Sprite0615-WBT-Stride085-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Sprite0615WbtFlatStride085WoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-Sprite0615-WBT-Stride075-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Sprite0615WbtFlatStride075WoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Sprite0615-WBT-Consistent075-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Sprite0615WbtFlatConsistent075WoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Sprite0615-WBT-Consistent075-CleanPlay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Sprite0615WbtFlatConsistent075CleanPlayEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Sprite0615-WBT-Consistent075-Tile8-CleanPlay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Sprite0615WbtFlatConsistent075Tile8CleanPlayEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-Sprite0615-WBT-Timescale150-J4340-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Sprite0615WbtFlatTimescale150J4340WoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-Sprite0615-WBT-Timescale150-MotorEnvelope-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Sprite0615WbtFlatTimescale150MotorEnvelopeWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-Sprite0615-WBT-Timescale150-MotorEnvelope-CleanPlay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Sprite0615WbtFlatTimescale150MotorEnvelopeCleanPlayEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-Sprite0615-WBT-Timescale150-PM01Rated-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-Sprite0615-WBT-Timescale150-PM01Rated-CleanPlay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedCleanPlayEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-Sprite0615-WBT-Timescale150-PM01Rated-PhaseCycleB-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-Sprite0615-WBT-Timescale150-PM01Rated-PhaseCycleB-CleanPlay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBCleanPlayEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-Sprite0615-WBT-Timescale150-PM01Rated-PhaseCycleB-Smooth015-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015WoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-Sprite0615-WBT-Timescale150-PM01Rated-PhaseCycleB-Smooth020-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth020WoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-Sprite0615-WBT-Timescale150-PM01Rated-PhaseCycleB-Smooth015-VariableSpeed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedEnvCfg
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


gym.register(
    id=(
        "Tracking-Flat-Sprite0615-WBT-Timescale150-PM01Rated-PhaseCycleB-"
        "Smooth015-VariableSpeed-UpperDeclaredAudit-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedUpperDeclaredAuditEnvCfg
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


gym.register(
    id=(
        "Tracking-Flat-Sprite0615-WBT-Timescale150-PM01Rated-PhaseCycleB-"
        "Smooth015-VariableSpeed-FullEnvelopeAudit-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullEnvelopeAuditEnvCfg
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


for audit_name, audit_cfg in (
    (
        "ArmsDcAudit",
        flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedArmsDcAuditEnvCfg,
    ),
    (
        "TorsoDcAudit",
        flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedTorsoDcAuditEnvCfg,
    ),
    (
        "DistalDcAudit",
        flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedDistalDcAuditEnvCfg,
    ),
):
    gym.register(
        id=(
            "Tracking-Flat-Sprite0615-WBT-Timescale150-PM01Rated-PhaseCycleB-"
            f"Smooth015-VariableSpeed-{audit_name}-v0"
        ),
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": audit_cfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
        },
    )


gym.register(
    id=(
        "Tracking-Flat-Sprite0615-WBT-Timescale150-PM01Rated-PhaseCycleB-"
        "Smooth015-VariableSpeed-PM01FullPeak-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedPm01FullPeakEnvCfg
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


for diagnostic_name, diagnostic_cfg in (
    (
        "FullPeakActuatorLegacyScale",
        flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakActuatorLegacyScaleEnvCfg,
    ),
    (
        "RatedActuatorFullPeakScale",
        flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedRatedActuatorFullPeakScaleEnvCfg,
    ),
):
    gym.register(
        id=(
            "Tracking-Flat-Sprite0615-WBT-Timescale150-PM01Rated-PhaseCycleB-"
            f"Smooth015-VariableSpeed-{diagnostic_name}-v0"
        ),
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": diagnostic_cfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
        },
    )


for stage_name, stage_cfg in (
    (
        "FullPeakLegacyScale-CleanTraining",
        flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakLegacyScaleCleanTrainingEnvCfg,
    ),
    (
        "FullPeakLegacyScale-AnkleTorque1e5-CleanTraining",
        flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakLegacyScaleAnkleTorque1e5CleanEnvCfg,
    ),
    (
        "FullPeakLegacyScale-AnkleTorque1e3-CleanTraining",
        flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakLegacyScaleAnkleTorque1e3CleanEnvCfg,
    ),
    (
        "FullPeakLegacyScale-AnkleTorque1e3-PushStage1",
        flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakLegacyScaleAnkleTorque1e3PushStage1EnvCfg,
    ),
    (
        "FullPeakLegacyScale-AnkleTorque1e3-PushStage1-FeetSlide010",
        flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakLegacyScaleAnkleTorque1e3PushStage1FeetSlide010EnvCfg,
    ),
    (
        "FullPeakLegacyScale-PushStage1",
        flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakLegacyScalePushStage1EnvCfg,
    ),
    (
        "FullPeakLegacyScale-PushStage2",
        flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakLegacyScalePushStage2EnvCfg,
    ),
    (
        "FullPeakLegacyScale-PushStage3",
        flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakLegacyScalePushStage3EnvCfg,
    ),
):
    gym.register(
        id=(
            "Tracking-Flat-Sprite0615-WBT-Timescale150-PM01Rated-PhaseCycleB-"
            f"Smooth015-VariableSpeed-{stage_name}-v0"
        ),
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": stage_cfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
        },
    )


gym.register(
    id=(
        "Tracking-Flat-Sprite0615-WBT-Timescale150-PM01Rated-PhaseCycleB-"
        "Smooth015-VariableSpeed-FullPeakLegacyScale-AnkleTorque1e3-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedFullPeakLegacyScaleAnkleTorque1e3EnvCfg
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


gym.register(
    id=(
        "Tracking-Flat-Sprite0615-WBT-Timescale150-PM01Rated-PhaseCycleB-"
        "Smooth015-VariableSpeed-CleanTraining-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedCleanTrainingEnvCfg
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


gym.register(
    id=(
        "Tracking-Flat-Sprite0615-WBT-Timescale150-PM01Rated-PhaseCycleB-"
        "Smooth015-VariableSpeed-MixedPush-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedMixedPushEnvCfg
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


for stage, env_cfg in (
    (
        1,
        flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedPushStage1EnvCfg,
    ),
    (
        2,
        flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedPushStage2EnvCfg,
    ),
    (
        3,
        flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedPushStage3EnvCfg,
    ),
):
    gym.register(
        id=(
            "Tracking-Flat-Sprite0615-WBT-Timescale150-PM01Rated-PhaseCycleB-"
            f"Smooth015-VariableSpeed-PushStage{stage}-v0"
        ),
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
        },
    )


gym.register(
    id="Tracking-Flat-Sprite0825-WBT-PM01Locomotion-Clean-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Sprite0825WbtPm01LocomotionCleanEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-Sprite0825-WBT-PM01Locomotion-FKv1-Clean-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Sprite0825WbtPm01LocomotionFkV1CleanEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-Sprite0825-WBT-PM01Locomotion-FKv1-CleanPlay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Sprite0825WbtPm01LocomotionFkV1CleanPlayEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-Sprite0825-WBT-PM01NativeStraight-G55-Clean-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Sprite0825WbtPm01NativeStraightG55CleanEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-Sprite0825-WBT-PM01NativeStraight-G55-CleanPlay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Sprite0825WbtPm01NativeStraightG55CleanPlayEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-Sprite0825-WBT-PM01NativeStraight-G56-PhaseBalanced-Clean-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Sprite0825WbtPm01NativeStraightG56PhaseBalancedCleanEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-Sprite0825-WBT-PM01NativeStraight-G56-PhaseBalanced-CleanPlay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Sprite0825WbtPm01NativeStraightG56PhaseBalancedCleanPlayEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


gym.register(
    id=(
        "Tracking-Flat-Sprite0615-WBT-Timescale150-PM01Rated-PhaseCycleB-"
        "Smooth015-VariableSpeed-EndpointRecovery-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedEndpointRecoveryEnvCfg
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
    },
)


for stage, env_cfg in (
    (
        1,
        flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedRecoveryStage1EnvCfg,
    ),
    (
        2,
        flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedRecoveryStage2EnvCfg,
    ),
    (
        3,
        flat_env_cfg.Sprite0615WbtFlatTimescale150Pm01RatedPhaseCycleBSmooth015VariableSpeedRecoveryStage3EnvCfg,
    ),
):
    gym.register(
        id=(
            "Tracking-Flat-Sprite0615-WBT-Timescale150-PM01Rated-PhaseCycleB-"
            f"Smooth015-VariableSpeed-RecoveryStage{stage}-v0"
        ),
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Sprite0615TrackingPPORunnerCfg",
        },
    )
