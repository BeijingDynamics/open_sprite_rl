#!/usr/bin/env python3
"""Validate that a packaged G74 checkpoint owns complete dual-seed Isaac evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_SEEDS = (303, 404)
EXPECTED_SHOULDER_ROLL_CENTERS = {
    "left_shoulder_roll_center_rad": 0.082816130,
    "right_shoulder_roll_center_rad": -0.076321766,
}
EXPECTED_J4340_JOINTS = tuple(
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee")
) + tuple(
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("shoulder_pitch", "shoulder_roll")
)
REQUIRED_EXTERNAL_PD_CHECKS = (
    "policy_and_mujoco_joint_name_sets_match",
    "all_policy_names_resolve",
    "runtime_resolves_names_with_mj_name2id",
    "runtime_writes_torque_through_address_map",
    "mujoco_external_pd_has_no_internal_actuators",
    "mujoco_physics_dt_is_0p002",
    "mujoco_gravity_is_minus_9p81_z",
)


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(package: Path) -> dict:
    package = package.resolve()
    manifest = load(package / "candidate_manifest.json")
    if manifest["schema"] != "sprite0825_g74_candidate_v1":
        raise ValueError("unexpected candidate manifest schema")
    if tuple(manifest["isaac_evaluation_seeds"]) != EXPECTED_SEEDS:
        raise ValueError("candidate manifest does not require seeds 303 and 404")
    changes = manifest.get("g74_changes", {})
    for name, expected in EXPECTED_SHOULDER_ROLL_CENTERS.items():
        if abs(float(changes.get(name, float("nan"))) - expected) > 1.0e-9:
            raise ValueError(f"G74 manifest has an incorrect {name}")

    iteration = int(manifest["iteration"])
    model = package / f"model_{iteration}.pt"
    if sha256(model) != manifest["checkpoint_sha256"]:
        raise ValueError("packaged checkpoint SHA-256 differs from candidate manifest")

    contract = load(package / "deploy" / "contract.json")
    joint_names = list(contract["joint_names"])
    j4340 = contract["physical_j4340p"]
    if tuple(j4340["joint_names"]) != EXPECTED_J4340_JOINTS:
        raise ValueError("physical J4340P joint inventory is not the required legs and shoulders")
    if float(j4340["rated_torque_nm"]) != 14.0 or float(j4340["peak_torque_nm"]) != 40.0:
        raise ValueError("physical J4340P torque contract is not 14 Nm rated / 40 Nm peak")
    if abs(float(j4340["rated_speed_rad_s"]) - 3.7699111843) > 1.0e-6:
        raise ValueError("physical J4340P rated speed contract is incorrect")
    if abs(float(j4340["no_load_speed_rad_s"]) - 9.3) > 1.0e-6:
        raise ValueError("physical J4340P no-load speed contract is incorrect")
    for name in EXPECTED_J4340_JOINTS:
        index = joint_names.index(name)
        if abs(float(contract["effort_limit"][index]) - 40.0) > 1.0e-6:
            raise ValueError(f"{name} does not use the 40 Nm peak limit")
        if abs(float(contract["velocity_limit"][index]) - 9.3) > 1.0e-6:
            raise ValueError(f"{name} does not use the 9.3 rad/s no-load limit")

    mapping = load(package / "evaluation" / "mujoco_mapping_audit.json")
    if not mapping.get("qualified"):
        raise ValueError("MuJoCo external-PD mapping audit is not qualified")
    checks = mapping.get("checks", {})
    failed = [name for name in REQUIRED_EXTERNAL_PD_CHECKS if checks.get(name) is not True]
    if failed:
        raise ValueError(f"MuJoCo external-PD mapping checks failed: {failed}")
    runtime = package / "runtime" / "run_sprite0825_stage2_mujoco.py"
    mjcf = package / "assets" / "mujoco" / "sprite0825_v5_4340_shoulders" / "scene_external_pd.xml"
    if sha256(runtime) != mapping["inputs"]["runtime_sha256"]:
        raise ValueError("packaged MuJoCo runtime differs from the audited external-PD runtime")
    if sha256(mjcf) != mapping["inputs"]["mjcf_sha256"]:
        raise ValueError("packaged MuJoCo asset differs from the audited external-PD asset")

    selection = load(package / "evaluation" / "dualseed_selection.json")
    # G74 intentionally reuses the frozen, regression-tested G72 gate implementation.
    if selection["schema"] != "sprite0825_g72_dualseed_selection_v1":
        raise ValueError("unexpected dual-seed selection schema")
    if not selection["qualified"]:
        raise ValueError("dual-seed selection is not qualified")
    if tuple(selection["evaluation_seeds"]) != EXPECTED_SEEDS:
        raise ValueError("dual-seed selection does not use independent seeds 303 and 404")
    if int(selection["winner_iteration"]) != iteration:
        raise ValueError("manifest checkpoint is not the dual-seed winner")
    if iteration not in {int(value) for value in selection["passing_both"]}:
        raise ValueError("winner is absent from the dual-seed passing intersection")

    evidence = {}
    for seed, dirname in (
        (303, "isaac_bridge_seed303"),
        (404, "isaac_confirmation_seed404"),
    ):
        root = package / "evaluation" / dirname
        summary = load(root / f"multiobjective_summary_seed{seed}.json")
        if int(summary["evaluation_seed"]) != seed:
            raise ValueError(f"summary seed mismatch for {seed}")
        if not summary.get("motor_contract_finalized"):
            raise ValueError(f"motor contract is not finalized for seed {seed}")
        row = summary["rows"][str(iteration)]
        if not row.get("passes_screen") or row.get("failed_gates"):
            raise ValueError(f"winner failed one or more gates for seed {seed}")
        candidate = root / f"model{iteration}"
        required = (
            candidate / f"transitions20_seed{seed}.json",
            candidate / f"yaw_transitions120_seed{seed}.json",
            candidate / f"style4800_seed{seed}.json",
        )
        for path in required:
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing dual-seed evidence: {path}")
        evidence[str(seed)] = [str(path.relative_to(package)) for path in required]

    return {
        "schema": "sprite0825_g74_dualseed_package_validation_v1",
        "qualified": True,
        "iteration": iteration,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "evaluation_seeds": list(EXPECTED_SEEDS),
        "j4340_joint_count": len(EXPECTED_J4340_JOINTS),
        "external_pd_mapping_verified": True,
        "evidence": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.package)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f'G74_DUALSEED_PACKAGE_OK model_{result["iteration"]}.pt '
        f'seeds={result["evaluation_seeds"]}'
    )


if __name__ == "__main__":
    main()
