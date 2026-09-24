#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from validate_sprite0825_g74_dualseed_package import validate


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def build(root: Path, *, manifest_seeds: list[int] | None = None) -> Path:
    iteration = 5000
    model = root / f"model_{iteration}.pt"
    model.write_bytes(b"checkpoint")
    checksum = hashlib.sha256(model.read_bytes()).hexdigest()
    write_json(
        root / "candidate_manifest.json",
        {
            "schema": "sprite0825_g74_candidate_v1",
            "iteration": iteration,
            "checkpoint_sha256": checksum,
            "isaac_evaluation_seeds": manifest_seeds or [303, 404],
            "g74_changes": {
                "left_shoulder_roll_center_rad": 0.082816130,
                "right_shoulder_roll_center_rad": -0.076321766,
            },
        },
    )
    j4340_names = [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee")
    ] + [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in ("shoulder_pitch", "shoulder_roll")
    ]
    joint_names = j4340_names + [f"other_{index}_joint" for index in range(19)]
    write_json(
        root / "deploy" / "contract.json",
        {
            "joint_names": joint_names,
            "effort_limit": [40.0] * len(j4340_names) + [1.0] * 19,
            "velocity_limit": [9.3] * len(j4340_names) + [1.0] * 19,
            "physical_j4340p": {
                "joint_names": j4340_names,
                "rated_torque_nm": 14.0,
                "peak_torque_nm": 40.0,
                "rated_speed_rad_s": 3.7699111843,
                "no_load_speed_rad_s": 9.3,
            },
        },
    )
    runtime = root / "runtime" / "run_sprite0825_stage2_mujoco.py"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text("# audited external-PD runtime\n", encoding="utf-8")
    mjcf = root / "assets" / "mujoco" / "sprite0825_v5_4340_shoulders" / "scene_external_pd.xml"
    mjcf.parent.mkdir(parents=True, exist_ok=True)
    mjcf.write_text("<mujoco/>\n", encoding="utf-8")
    required_checks = {
        "policy_and_mujoco_joint_name_sets_match": True,
        "all_policy_names_resolve": True,
        "runtime_resolves_names_with_mj_name2id": True,
        "runtime_writes_torque_through_address_map": True,
        "mujoco_external_pd_has_no_internal_actuators": True,
        "mujoco_physics_dt_is_0p002": True,
        "mujoco_gravity_is_minus_9p81_z": True,
    }
    write_json(
        root / "evaluation" / "mujoco_mapping_audit.json",
        {
            "qualified": True,
            "checks": required_checks,
            "inputs": {
                "runtime_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
                "mjcf_sha256": hashlib.sha256(mjcf.read_bytes()).hexdigest(),
            },
        },
    )
    write_json(
        root / "evaluation" / "dualseed_selection.json",
        {
            "schema": "sprite0825_g72_dualseed_selection_v1",
            "qualified": True,
            "evaluation_seeds": [303, 404],
            "winner_iteration": iteration,
            "passing_both": [iteration],
        },
    )
    for seed, dirname in ((303, "isaac_bridge_seed303"), (404, "isaac_confirmation_seed404")):
        base = root / "evaluation" / dirname
        write_json(
            base / f"multiobjective_summary_seed{seed}.json",
            {
                "evaluation_seed": seed,
                "motor_contract_finalized": True,
                "rows": {str(iteration): {"passes_screen": True, "failed_gates": []}},
            },
        )
        candidate = base / f"model{iteration}"
        for name in (
            f"transitions20_seed{seed}.json",
            f"yaw_transitions120_seed{seed}.json",
            f"style4800_seed{seed}.json",
        ):
            write_json(candidate / name, {"ok": True})
    return root


def main() -> None:
    with TemporaryDirectory() as tmp:
        result = validate(build(Path(tmp)))
        assert result["qualified"] and result["iteration"] == 5000

    with TemporaryDirectory() as tmp:
        package = build(Path(tmp), manifest_seeds=[303])
        try:
            validate(package)
        except ValueError as exc:
            assert "seeds 303 and 404" in str(exc)
        else:
            raise AssertionError("single-seed package was accepted")

    with TemporaryDirectory() as tmp:
        package = build(Path(tmp))
        contract_path = package / "deploy" / "contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        shoulder = contract["joint_names"].index("left_shoulder_pitch_joint")
        contract["effort_limit"][shoulder] = 14.0
        write_json(contract_path, contract)
        try:
            validate(package)
        except ValueError as exc:
            assert "40 Nm peak limit" in str(exc)
        else:
            raise AssertionError("rated-only J4340P shoulder contract was accepted")

    with TemporaryDirectory() as tmp:
        package = build(Path(tmp))
        manifest_path = package / "candidate_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["g74_changes"]["left_shoulder_roll_center_rad"] = 0.25
        write_json(manifest_path, manifest)
        try:
            validate(package)
        except ValueError as exc:
            assert "left_shoulder_roll_center_rad" in str(exc)
        else:
            raise AssertionError("incorrect G74 shoulder-roll center was accepted")

    print("G74_DUALSEED_PACKAGE_TEST_PASS")


if __name__ == "__main__":
    main()
