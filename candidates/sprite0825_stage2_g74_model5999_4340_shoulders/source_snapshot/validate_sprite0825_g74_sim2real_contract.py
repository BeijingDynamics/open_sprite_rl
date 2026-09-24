#!/usr/bin/env python3
"""Fail closed when a Sprite0825 deployment contract drifts from qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_TERMS = [
    ("joint_pos", 0, 248, 31, 8),
    ("joint_vel", 248, 496, 31, 8),
    ("actions", 496, 744, 31, 8),
    ("base_ang_vel", 744, 768, 3, 8),
    ("projected_gravity", 768, 792, 3, 8),
    ("velocity_commands", 792, 795, 3, 0),
]


def close(actual: float, expected: float, tolerance: float = 1.0e-5) -> bool:
    return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=tolerance)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_path(contract: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else contract.parent / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text())
    assert contract["checkpoint_sha256"] == args.checkpoint_sha256
    assert contract["package_path_resolution"] == "relative_to_contract_directory"
    for path_key, hash_key in (
        ("policy_onnx", "policy_onnx_sha256"),
        ("policy_jit", "policy_jit_sha256"),
        ("asset_urdf", "asset_urdf_sha256"),
        ("asset_usd", "asset_usd_sha256"),
    ):
        path = contract_path(args.contract, contract[path_key]).resolve()
        assert path.is_file(), path
        assert sha256(path) == contract[hash_key], (path_key, path)
    assert close(contract["policy_dt"], 0.02)
    assert close(contract["physics_dt"], 0.002)
    assert contract["decimation"] == 10
    assert contract["actor_observation_dim"] == 795
    assert contract["action_dim"] == 31
    assert contract["mujoco_joint_mapping"] == "explicit_name_to_qpos_and_dof_addresses"
    assert contract["mujoco_native_joint_order_may_differ"] is True
    assert contract["observation_history_reset"] == "replicate_first_sample_across_all_history_frames"
    assert not contract["observation_has_horizontal_base_velocity"]
    assert not contract["observation_has_global_position"]
    assert not contract["observation_has_global_yaw"]
    assert contract["command_layout"] == ["vx", "vy", "yaw_rate"]

    actual_terms = [
        (term["name"], term["start"], term["end"], term["frame_dim"], term["history_length"])
        for term in contract["observation_terms"]
    ]
    assert actual_terms == EXPECTED_TERMS, (actual_terms, EXPECTED_TERMS)

    names = contract["joint_names"]
    assert len(names) == 31
    assert all(len(contract[key]) == 31 for key in (
        "default_joint_pos",
        "action_scale",
        "action_offset",
        "stiffness",
        "damping",
        "effort_limit",
        "velocity_limit",
    ))
    joint_index = {name: index for index, name in enumerate(names)}
    assert len(joint_index) == 31
    for side in ("left", "right"):
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee"):
            index = joint_index[f"{side}_{joint}_joint"]
            assert close(contract["effort_limit"][index], 40.0)
            assert close(contract["velocity_limit"][index], 9.3, tolerance=1.0e-3)
        for joint in ("shoulder_pitch", "shoulder_roll"):
            index = joint_index[f"{side}_{joint}_joint"]
            assert close(contract["effort_limit"][index], 40.0)
            assert close(contract["velocity_limit"][index], 9.3, tolerance=1.0e-3)
        for joint in ("ankle_pitch", "ankle_roll"):
            index = joint_index[f"{side}_{joint}_joint"]
            assert close(contract["effort_limit"][index], 25.0)
            assert close(contract["velocity_limit"][index], 36.2, tolerance=1.0e-3)

    ankle = contract["physical_ankle_differential"]
    assert ankle["enabled"]
    assert close(ankle["rated_torque_nm"], 3.5)
    assert close(ankle["peak_torque_nm"], 12.5)
    assert close(ankle["rated_speed_rad_s"], 12.56)
    assert close(ankle["no_load_speed_rad_s"], 36.2)
    assert close(ankle["joint_to_motor_ratio"], 1.0)
    assert ankle["joint_pairs"] == {
        side: [f"{side}_ankle_pitch_joint", f"{side}_ankle_roll_joint"]
        for side in ("left", "right")
    }
    assert ankle["motor_torque_formula"] == [
        "0.5 * (tau_pitch + tau_roll)",
        "0.5 * (tau_pitch - tau_roll)",
    ]
    assert ankle["motor_velocity_formula"] == [
        "qd_pitch + qd_roll",
        "qd_pitch - qd_roll",
    ]
    j4340 = contract["physical_j4340p"]
    assert j4340["joint_names"] == [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee")
    ] + [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in ("shoulder_pitch", "shoulder_roll")
    ]
    assert close(j4340["rated_torque_nm"], 14.0)
    assert close(j4340["peak_torque_nm"], 40.0)
    assert close(j4340["rated_speed_rad_s"], 3.7699111843)
    assert close(j4340["no_load_speed_rad_s"], 9.3)
    assert close(contract["deployment_pd_scale"], 1.0)
    assert close(contract["deployment_handoff_seconds"], 0.04)
    assert contract["deployment_handoff_mode"] == "smoothstep_from_pose_equivalent_action"
    assert contract["deployment_initial_velocity_mode"] == "zero"
    assert contract["base_ang_vel_frame"] == "root_link_local"
    assert contract["projected_gravity_definition"] == "R_world_to_base @ [0, 0, -1]"
    print("Sprite0825 sim2real deployment contract validated")


if __name__ == "__main__":
    main()
