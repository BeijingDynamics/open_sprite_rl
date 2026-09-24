#!/usr/bin/env python3
"""Audit Sprite0825 policy-joint to MuJoCo qpos/dof name mapping."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", required=True, type=Path)
    parser.add_argument("--mjcf", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    structure = json.loads(args.structure.read_text(encoding="utf-8"))
    policy_names = list(structure["action_joint_names"])
    model = mujoco.MjModel.from_xml_path(str(args.mjcf.resolve()))
    native_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(1, model.njnt)
    ]
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in policy_names
    ]
    qpos_addresses = [int(model.jnt_qposadr[index]) for index in joint_ids]
    dof_addresses = [int(model.jnt_dofadr[index]) for index in joint_ids]
    runtime_text = args.runtime.read_text(encoding="utf-8")

    checks = {
        "policy_has_31_unique_joint_names": len(policy_names) == len(set(policy_names)) == 31,
        "mujoco_has_31_unique_actuated_joint_names": len(native_names) == len(set(native_names)) == 31,
        "policy_and_mujoco_joint_name_sets_match": set(policy_names) == set(native_names),
        "all_policy_names_resolve": all(index >= 0 for index in joint_ids),
        "qpos_addresses_are_unique": len(qpos_addresses) == len(set(qpos_addresses)) == 31,
        "dof_addresses_are_unique": len(dof_addresses) == len(set(dof_addresses)) == 31,
        "runtime_resolves_names_with_mj_name2id": "mujoco.mj_name2id" in runtime_text,
        "runtime_reads_qpos_through_address_map": "data.qpos[qpos_addr]" in runtime_text,
        "runtime_reads_qvel_through_address_map": "data.qvel[dof_addr]" in runtime_text,
        "runtime_writes_torque_through_address_map": "data.qfrc_applied[dof_addr] = tau" in runtime_text,
        "mujoco_external_pd_has_no_internal_actuators": model.nu == 0,
        "mujoco_physics_dt_is_0p002": abs(float(model.opt.timestep) - 0.002) <= 1.0e-12,
        "mujoco_gravity_is_minus_9p81_z": list(model.opt.gravity) == [0.0, 0.0, -9.81],
    }
    result = {
        "schema": "sprite0825_g72_mujoco_mapping_audit_v1",
        "qualified": all(checks.values()),
        "checks": checks,
        "native_order_equals_policy_order": native_names == policy_names,
        "mapping_mode": "explicit_name_to_qpos_and_dof_addresses",
        "policy_joint_names": policy_names,
        "mujoco_native_joint_names": native_names,
        "policy_order_qpos_addresses": qpos_addresses,
        "policy_order_dof_addresses": dof_addresses,
        "inputs": {
            "structure": str(args.structure.resolve()),
            "structure_sha256": sha256(args.structure),
            "mjcf": str(args.mjcf.resolve()),
            "mjcf_sha256": sha256(args.mjcf),
            "runtime": str(args.runtime.resolve()),
            "runtime_sha256": sha256(args.runtime),
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["qualified"]:
        raise SystemExit("G72 MuJoCo mapping audit failed")


if __name__ == "__main__":
    main()
