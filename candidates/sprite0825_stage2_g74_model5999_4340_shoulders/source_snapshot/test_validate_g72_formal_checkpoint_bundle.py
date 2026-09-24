#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from validate_g72_formal_checkpoint_bundle import validate


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root: Path, step: int = 2500) -> Path:
    files = {
        f"model_{step}.pt": b"model",
        "params/env.yaml": b"env\n",
        "params/agent.yaml": b"agent\n",
        "RECOVERY.txt": b"recovery\n",
    }
    for relative, data in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    lines = [f"{digest(root / relative)}  ./{relative}" for relative in sorted(files)]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def expect_failure(root: Path, text: str) -> None:
    try:
        validate(root, 2500)
    except ValueError as exc:
        assert text in str(exc), exc
    else:
        raise AssertionError("invalid formal checkpoint bundle was accepted")


def main() -> None:
    with TemporaryDirectory() as tmp:
        result = validate(build(Path(tmp)), 2500)
        assert result == {"step": 2500, "files": 4, "qualified": True}

    with TemporaryDirectory() as tmp:
        root = build(Path(tmp))
        model = root / "model_2500.pt"
        (root / "SHA256SUMS").write_text(
            f"{digest(model)}  ./model_2500.pt\n", encoding="utf-8"
        )
        expect_failure(root, "manifest file set mismatch")

    with TemporaryDirectory() as tmp:
        root = build(Path(tmp))
        (root / "params" / "env.yaml").write_text("tampered\n", encoding="utf-8")
        expect_failure(root, "SHA-256 mismatch: params/env.yaml")

    print("G72_FORMAL_BUNDLE_TEST_PASS")


if __name__ == "__main__":
    main()
