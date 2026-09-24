#!/usr/bin/env python3
"""Verify the serialized G72 smoke run against the sim2real static contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


EXPECTED_OBSERVATIONS = [
    "joint_pos",
    "joint_vel",
    "actions",
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
]

EXPECTED_ACTION_JOINTS = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_yaw_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "head_roll_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "head_yaw_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
]


def number(value: object) -> float:
    return float(str(value))


def require(checks: list[dict], name: str, actual: object, expected: object) -> None:
    passed = actual == expected
    checks.append({"name": name, "passed": passed, "actual": actual, "expected": expected})


def require_number(checks: list[dict], name: str, actual: object, expected: float) -> None:
    value = number(actual)
    passed = abs(value - expected) <= 1.0e-9
    checks.append({"name": name, "passed": passed, "actual": value, "expected": expected})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    env = yaml.load(args.env.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    agent = yaml.load(args.agent.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    structure = json.loads(args.structure.read_text(encoding="utf-8"))
    checks: list[dict] = []

    require_number(checks, "physics_dt_s", env["sim"]["dt"], 0.002)
    require_number(checks, "decimation", env["decimation"], 10.0)
    require_number(checks, "policy_dt_s", structure["step_dt"], 0.02)
    require(
        checks,
        "asset_revision",
        Path(env["scene"]["robot"]["spawn"]["usd_path"]).parts[-2:],
        ("sprite0825_sanitized_v5_4340_shoulders", "sprite0825.usd"),
    )

    actuators = env["scene"]["robot"]["actuators"]
    for group in ("legs_j4340", "waist_yaw_j4340", "shoulder_pitch_roll_j4340"):
        require_number(checks, f"{group}.peak_torque_nm", actuators[group]["effort_limit"], 40.0)
        require_number(checks, f"{group}.no_load_speed_rad_s", actuators[group]["velocity_limit"], 9.3)
        require_number(checks, f"{group}.sim_peak_torque_nm", actuators[group]["effort_limit_sim"], 40.0)

    shoulders = actuators["shoulder_pitch_roll_j4340"]
    require(checks, "shoulder_4340_joint_pattern", shoulders["joint_names_expr"], [".*_shoulder_(pitch|roll)_joint"])
    require_number(checks, "shoulder_4340_armature", shoulders["armature"], 0.032)

    ankles = actuators["feet_j4310_pair"]
    require_number(checks, "ankle_motor_peak_torque_nm", ankles["effort_limit"], 12.5)
    require_number(checks, "ankle_pair_generalized_peak_nm", ankles["effort_limit_sim"], 25.0)
    require(checks, "ankle_actuator", ankles["class_type"].split(":")[-1], "DelayedCoupledAnkleDCMotor")

    distal = actuators["arms_distal_j4310"]
    require_number(checks, "distal_arm_peak_torque_nm", distal["effort_limit"], 12.5)

    policy_terms = [
        name
        for name, config in env["observations"]["policy"].items()
        if isinstance(config, dict) and "func" in config
    ]
    require(checks, "actor_observation_terms", policy_terms, EXPECTED_OBSERVATIONS)
    require(checks, "actor_contains_base_lin_vel", structure["contains_base_lin_vel"], False)
    require(checks, "runtime_observation_terms", structure["observation_terms"], EXPECTED_OBSERVATIONS)
    require(checks, "action_joint_order", structure["action_joint_names"], EXPECTED_ACTION_JOINTS)
    require(checks, "action_dimension", len(structure["action_joint_names"]), 31)

    require(
        checks,
        "algorithm",
        agent["algorithm"]["class_name"].split(":")[-1],
        "UnconditionedAMPPPO",
    )
    require(checks, "actor_groups", agent["obs_groups"]["actor"], ["policy"])
    require(checks, "critic_groups", agent["obs_groups"]["critic"], ["critic"])
    require(checks, "amp_expert_frame_stride", int(agent["expert_frame_stride"]), 2)
    require(
        checks,
        "amp_dataset",
        Path(agent["dataset_path"]).parts[-2:],
        ("stage2_amp", "g72_g71_4340_shoulders_100hz_v1"),
    )
    require(checks, "amp_symmetry", agent["algorithm"]["symmetry_cfg"]["use_mirror_loss"], "true")

    motor_contract = structure["motor_envelope_contract"]
    require_number(checks, "j4340_rated_torque_nm", motor_contract["j4340p"]["rated_torque_nm"], 14.0)
    require_number(checks, "j4340_peak_torque_nm", motor_contract["j4340p"]["peak_torque_nm"], 40.0)
    require_number(checks, "j4310_rated_torque_nm", motor_contract["j4310p_differential_ankle"]["rated_torque_nm"], 3.5)
    require_number(checks, "j4310_peak_torque_nm", motor_contract["j4310p_differential_ankle"]["peak_torque_nm"], 12.5)
    require(
        checks,
        "ankle_torque_mapping",
        motor_contract["j4310p_differential_ankle"]["motor_torque_mapping"],
        "0.5 * (joint_pitch_torque +/- joint_roll_torque)",
    )

    failures = [check for check in checks if not check["passed"]]
    result = {
        "schema": "sprite0825_g72_static_sim2real_contract_v1",
        "qualified": not failures,
        "checks": checks,
        "failures": failures,
        "sources": {"env": str(args.env), "agent": str(args.agent), "structure": str(args.structure)},
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"qualified": result["qualified"], "checks": len(checks), "failures": len(failures)}, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
