#!/usr/bin/env python3
"""Create a package-local Stage 2 contract without changing policy semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()

contract = json.loads(args.source.read_text(encoding="utf-8"))
deploy_dir = args.output.parent.resolve()
onnx_path = deploy_dir / "policy.onnx"
jit_path = deploy_dir / "policy.pt"
if not onnx_path.is_file() or not jit_path.is_file():
    raise FileNotFoundError("policy.onnx and policy.pt must be beside the localized contract")
if sha256(onnx_path) != contract["policy_onnx_sha256"]:
    raise RuntimeError("policy.onnx does not match the exported contract")

contract["policy_onnx"] = "policy.onnx"
contract["policy_jit"] = "policy.pt"
contract["package_path_resolution"] = "relative_to_contract_directory"
contract["source_export_contract_sha256"] = sha256(args.source)
args.output.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
print(f"WROTE {args.output}")
print(f"ONNX {sha256(onnx_path)}")
print(f"JIT {sha256(jit_path)}")
print(f"CONTRACT {sha256(args.output)}")
