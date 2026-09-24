#!/usr/bin/env python3
"""Validate a complete, self-hashed G72 formal checkpoint bundle."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_relative(value: str) -> str:
    while value.startswith("./"):
        value = value[2:]
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe manifest path: {value!r}")
    return path.as_posix()


def validate(root: Path, step: int) -> dict:
    root = root.resolve()
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        raise ValueError("missing SHA256SUMS")

    entries: dict[str, str] = {}
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        match = LINE.fullmatch(raw)
        if match is None:
            raise ValueError(f"invalid SHA256SUMS line: {raw!r}")
        relative = normalize_relative(match.group(2))
        if relative in entries:
            raise ValueError(f"duplicate manifest path: {relative}")
        entries[relative] = match.group(1)

    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(entries) != actual:
        missing = sorted(actual - set(entries))
        stale = sorted(set(entries) - actual)
        raise ValueError(f"manifest file set mismatch: missing={missing} stale={stale}")

    required = {
        f"model_{step}.pt",
        "params/env.yaml",
        "params/agent.yaml",
        "RECOVERY.txt",
    }
    if not required.issubset(actual):
        raise ValueError(f"required recovery files missing: {sorted(required - actual)}")

    for relative, expected in entries.items():
        path = root / relative
        if path.is_symlink():
            raise ValueError(f"symlink is not allowed in formal bundle: {relative}")
        if sha256(path) != expected:
            raise ValueError(f"SHA-256 mismatch: {relative}")

    return {"step": step, "files": len(entries), "qualified": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    result = validate(args.root, args.step)
    print(f'G72_FORMAL_BUNDLE_OK model_{result["step"]} files={result["files"]}')


if __name__ == "__main__":
    main()
