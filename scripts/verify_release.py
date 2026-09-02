#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "baselines/sprite0615_wbt_v36_model100_warmstart/model_100.pt": "e78e32edbb8c1bfdb5188ae2943b507a1b7bb67d5d53c55bedb8bf26d4113c1c",
    "baselines/sprite0615_wbt_v38_model2_stage1_qualified/checkpoints/model_2.pt": "b8d1851de07e654c367d0ad9d6d2e32a9fbc9588e4f26aabd7e5e4d0d71166ff",
    "baselines/sprite0825_stage2_g57_model500_forward_candidate/model_500.pt": "c4bb13b271e840f6f9c182e5681c5ab7ebc1f48055406fb49bb1d6afda8e419d",
    "baselines/sprite0825_stage2_g58a_model799_physical_candidate/model_799.pt": "47202164047b044e3e524cb65134b583725204057cae91fe41047583ba08a66b",
    "baselines/sprite0825_stage2_g58b_model925_speed_candidate/model_925.pt": "e4d74619be6ea0786057ede910785bf58f8f92088f72c431578fdf56644523e9",
    "baselines/sprite0825_stage2_g58f_model1050_stage2_qualified/model_1050.pt": "deda0cc80023c00e6fd7723830bf3a9b9ceb4b33e72c9e1e446d9b0873f1d912",
    "baselines/sprite0825_stage2_g58f_model1050_stage2_qualified/deploy/policy.onnx": "75a5c89552539da344e21566843f6c1fd1eac52e7601bbf8b5de64aaaae9eb26",
    "deploy/sprite0825_v4_stage2/scene_external_pd.xml": "6ec42ff665fde05db58b507cc2747fb2a877e64efea003ff6b3d9e2f8c72e04f",
    "assets/sprite0825_sanitized_v4/sprite0825.usd": "3a5e233385bd1dd32e012cf07a54325a58656c6abccd56c014469925353106e7",
    "assets/sprite0825_sanitized_v4/sprite0825_float.urdf": "80924527cb9e61d85dc3f51f7bf43c631a938bcbe4fc9b1d4419c55ccb4a39e3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    errors: list[str] = []
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
            continue
        actual = sha256(path)
        if actual != expected:
            errors.append(f"hash mismatch: {relative}: {actual}")

    contract_path = (
        ROOT
        / "baselines/sprite0825_stage2_g58f_model1050_stage2_qualified/deploy/contract.json"
    )
    if contract_path.is_file():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        checks = {
            "actor_observation_dim": contract.get("actor_observation_dim") == 1488,
            "action_dim": len(contract.get("joint_names", [])) == 31,
            "no_horizontal_velocity": contract.get("observation_has_horizontal_base_velocity") is False,
            "policy_dt": contract.get("policy_dt") == 0.01,
            "physics_dt": contract.get("physics_dt") == 0.002,
        }
        errors.extend(f"contract check failed: {name}" for name, ok in checks.items() if not ok)

    overlay = ROOT / "isaaclab_overlay"
    for path in overlay.rglob("*.py"):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            errors.append(f"Python parse failed: {path.relative_to(ROOT)}: {exc}")

    residue = [p.relative_to(ROOT) for p in overlay.rglob("__pycache__")]
    residue.extend(p.relative_to(ROOT) for p in overlay.rglob("*.pyc"))
    if residue:
        errors.append(f"generated Python cache present: {residue[:5]}")

    if errors:
        print("RELEASE VERIFICATION FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"RELEASE VERIFICATION PASSED ({len(EXPECTED)} artifact hashes)")
    print("contract: 1488 observations, 31 actions, no horizontal base velocity")


if __name__ == "__main__":
    main()
