#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import xml.etree.ElementTree as ET


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
    "baselines/sprite0825_stage2_g59_model2999_native50_parent/model_2999.pt": "eb84e5aedb8ace647d877788a98c86db6ce9670e4c003890e6e1d2ef21205a98",
    "baselines/sprite0825_stage2_g59_model2999_native50_parent/deploy/policy.onnx": "38f2fbc52d385904d42e2410db89a88de7394e4525b60f5ae999921fb50ed1a7",
    "baselines/sprite0825_stage2_g60_model3450_current/model_3450.pt": "6a1a80a2a2f7073698c0886133c325a46462ace6bbd3cf7de70246beafb85f15",
    "baselines/sprite0825_stage2_g60_model3450_current/deploy/policy.onnx": "43f213e4c5b9079e13b7f6f3635f224766227757417a0940ca49d585231016e3",
    "candidates/sprite0825_stage2_g74_model5999_4340_shoulders/model_5999.pt": "bc8e84802703926366f6d7348c1da7365d4fbbe678297888443ed8949c3b7825",
    "candidates/sprite0825_stage2_g74_model5999_4340_shoulders/deploy/policy.onnx": "8ce307c446a7089587ec5a1ec28534fe8e2a395f4875797a5613ac408a72a8ea",
    "candidates/sprite0825_stage2_g74_model5999_4340_shoulders/assets/isaac/sprite0825_sanitized_v5_4340_shoulders/sprite0825.usd": "d21e1ceed89cabcf65ff3d73b7fee9da3640c2b97e0ac704261e52401b97af27",
    "candidates/sprite0825_stage2_g74_model5999_4340_shoulders/assets/isaac/sprite0825_sanitized_v5_4340_shoulders/sprite0825_float.urdf": "6c753d563b54103e4278c27a11a22e0f4f1076e4156f0873cb2325f1701a921b",
    "candidates/sprite0825_stage2_g74_model5999_4340_shoulders/assets/mujoco/sprite0825_v5_4340_shoulders/scene_external_pd.xml": "b4f6dd161f6af1d4fadb44da155d1ff3c319341b541aab99fc4c176c22a40d9d",
}

RELEASE_ASSET_ROOTS = (
    ROOT / "assets",
    ROOT / "deploy",
    ROOT / "candidates/sprite0825_stage2_g74_model5999_4340_shoulders/assets",
)
CAD_MARKERS = (
    b"solidworks",
    b"sw2urdf",
    b"solidworks to urdf",
    b"autodesk inventor",
    b"blender",
    b"catia",
    b"creo parametric",
    b"freecad",
    b"fusion 360",
    b"meshlab",
    b"onshape",
    b"siemens nx",
    b"solid edge",
)
TEXT_ASSET_SUFFIXES = {
    ".dae",
    ".json",
    ".mjcf",
    ".mtl",
    ".obj",
    ".py",
    ".sh",
    ".txt",
    ".urdf",
    ".usda",
    ".xacro",
    ".xml",
    ".yaml",
    ".yml",
}
HOST_PATH_MARKERS = (b"/home/", b"/users/", b"\\users\\")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_release_assets(errors: list[str]) -> None:
    asset_files = [
        path
        for root in RELEASE_ASSET_ROOTS
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file()
    ]
    for path in asset_files:
        relative = path.relative_to(ROOT)
        lowered_name = path.name.lower().encode("utf-8")
        lowered_content = path.read_bytes().lower()
        for marker in CAD_MARKERS:
            if marker in lowered_name or marker in lowered_content:
                errors.append(f"CAD exporter residue in release asset: {relative}: {marker.decode()}")
        if path.suffix.lower() in TEXT_ASSET_SUFFIXES:
            for marker in HOST_PATH_MARKERS:
                if marker in lowered_content:
                    errors.append(f"host-specific path in release asset: {relative}: {marker.decode()}")

    for path in [*ROOT.rglob("*.urdf"), *ROOT.rglob("*.xacro")]:
        try:
            document = ET.parse(path)
        except ET.ParseError as exc:
            errors.append(f"URDF parse failed: {path.relative_to(ROOT)}: {exc}")
            continue
        for mesh in document.iterfind(".//mesh"):
            filename = mesh.get("filename")
            if not filename:
                errors.append(f"mesh without filename: {path.relative_to(ROOT)}")
                continue
            if (
                "://" in filename
                or PurePosixPath(filename).is_absolute()
                or PureWindowsPath(filename).is_absolute()
            ):
                errors.append(f"non-portable mesh path: {path.relative_to(ROOT)}: {filename}")
                continue
            resolved = (path.parent / filename).resolve()
            if not resolved.is_file():
                errors.append(f"missing mesh: {path.relative_to(ROOT)}: {filename}")


def resolve_relative(base: Path, value: str, label: str, errors: list[str]) -> Path | None:
    if "://" in value or PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        errors.append(f"non-portable {label}: {value}")
        return None
    resolved = (base / value).resolve()
    if not resolved.exists():
        errors.append(f"unresolved {label}: {value}")
        return None
    return resolved


def audit_portable_metadata(errors: list[str]) -> None:
    asset_dir = ROOT / "assets/sprite0825_sanitized_v4"
    config_values: dict[str, str] = {}
    for line in (asset_dir / "config.yaml").read_text(encoding="utf-8").splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            key, value = line.split(":", 1)
            config_values[key.strip()] = value.strip()
    for key in ("asset_path", "usd_dir"):
        value = config_values.get(key)
        if not value:
            errors.append(f"missing asset config key: {key}")
        else:
            resolve_relative(asset_dir, value, f"asset config {key}", errors)

    asset_manifest = json.loads((asset_dir / "manifest.json").read_text(encoding="utf-8"))
    output = resolve_relative(
        asset_dir, asset_manifest.get("output", ""), "asset manifest output", errors
    )
    if output is not None and sha256(output) != asset_manifest.get("output_sha256"):
        errors.append("asset manifest output hash mismatch")
    for key in ("qualified_joint_contract", "source"):
        value = asset_manifest.get(key, "")
        if not value.startswith("external://"):
            errors.append(f"asset provenance is not an external URI: {key}: {value}")

    reference_dir = (
        ROOT
        / "assets/references/stage2_amp/g57_sprite0825_native_pm01_100hz_v1"
    )
    reference_manifest = json.loads(
        (reference_dir / "manifest.json").read_text(encoding="utf-8")
    )
    for path_key, hash_key in (
        ("mapped_source", "mapped_sha256"),
        ("phase_source", "phase_source_sha256"),
    ):
        source = resolve_relative(
            reference_dir,
            reference_manifest.get(path_key, ""),
            f"reference manifest {path_key}",
            errors,
        )
        if source is not None and sha256(source) != reference_manifest.get(hash_key):
            errors.append(f"reference manifest hash mismatch: {path_key}")


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

    current_contract_path = (
        ROOT
        / "baselines/sprite0825_stage2_g60_model3450_current/deploy/contract.json"
    )
    if current_contract_path.is_file():
        current = json.loads(current_contract_path.read_text(encoding="utf-8"))
        checks = {
            "current_actor_observation_dim": current.get("actor_observation_dim") == 795,
            "current_action_dim": len(current.get("joint_names", [])) == 31,
            "current_no_horizontal_velocity": current.get("observation_has_horizontal_base_velocity") is False,
            "current_policy_dt": current.get("policy_dt") == 0.02,
            "current_physics_dt": current.get("physics_dt") == 0.002,
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

    audit_release_assets(errors)
    audit_portable_metadata(errors)

    if errors:
        print("RELEASE VERIFICATION FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"RELEASE VERIFICATION PASSED ({len(EXPECTED)} artifact hashes)")
    print("current contract: 795 observations, 31 actions, no horizontal base velocity")


if __name__ == "__main__":
    main()
