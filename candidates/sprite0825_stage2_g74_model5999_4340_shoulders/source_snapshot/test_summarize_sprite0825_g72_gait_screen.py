#!/usr/bin/env python3
"""Regression tests for the fixed G72 coarse gait screen."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).with_name("summarize_sprite0825_g72_gait_screen.py")


def evaluation(*, checkpoint: str, cadence: float = 148.0) -> dict:
    foot = {
        "mean_abs_rad": 0.01,
        "p90_abs_rad": 0.03,
        "over_0p06_fraction": 0.0,
        "signed_mean_rad": 0.005,
    }
    return {
        "checkpoint": checkpoint,
        "survival_rate": 1.0,
        "gait": {
            "mean_path_speed_mps": 0.27,
            "mean_cadence_steps_per_min": cadence,
            "mean_estimated_step_length_m": 0.11,
            "mean_abs_heading_drift_deg": 2.0,
            "contact_foot_roll": {"left": foot, "right": foot},
        },
    }


def run_screen(
    root: Path, *, diagnostic_fallback: bool = False
) -> subprocess.CompletedProcess[str]:
    output = root / "summary.json"
    top = root / "top3.txt"
    command = [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(root),
            "--output",
            str(output),
            "--top-output",
            str(top),
            "--top",
            "3",
        ]
    if diagnostic_fallback:
        command.append("--diagnostic-fallback")
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        good = "/run/model_2500.pt"
        (root / "model2500_gait30_seed43.json").write_text(
            json.dumps(evaluation(checkpoint=good)), encoding="utf-8"
        )
        result = run_screen(root)
        assert result.returncode == 0, result.stderr
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        assert summary["selected"][0]["checkpoint"] == good
        assert (root / "top3.txt").read_text(encoding="utf-8") == f"{good}\n"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "model5000_gait30_seed43.json").write_text(
            json.dumps(evaluation(checkpoint="/run/model_5000.pt", cadence=190.0)),
            encoding="utf-8",
        )
        result = run_screen(root)
        assert result.returncode != 0
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        assert summary["selected"] == []
        assert not summary["rows"][0]["gates"]["cadence_120_to_170"]

        result = run_screen(root, diagnostic_fallback=True)
        assert result.returncode == 0, result.stderr
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        assert summary["selected"] == []
        assert summary["diagnostic_fallback_used"]
        assert summary["diagnostic_selected"][0]["iteration"] == 5000
        assert (root / "top3.txt").read_text(encoding="utf-8") == (
            "/run/model_5000.pt\n"
        )

    print("G72_GAIT_SCREEN_TEST_PASS")


if __name__ == "__main__":
    main()
