#!/usr/bin/env python3
"""Summarize G58F command and model-925 style preservation screens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ITERATIONS = (925, 950, 975, 1000, 1025, 1050, 1075, 1100, 1125, 1150, 1175, 1200, 1225, 1250, 1275, 1300, 1324)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for iteration in ITERATIONS:
        yaw = load(args.input / f"model_{iteration}_yaw20.json")
        grid = load(args.input / f"model_{iteration}_grid128.json")
        style_pass = True
        failed_style_gates: list[str] = []
        if iteration != 925:
            style = load(args.input / f"model_{iteration}_vs_925_style_gates.json")
            style_pass = bool(style["pass"])
            failed_style_gates = list(style["failed_gates"])

        yaw_grid = yaw["yaw_grid"]
        speed_grid = grid["grid"]
        command_pass = (
            float(yaw["survival_rate"]) == 1.0
            and float(grid["survival_rate"]) == 1.0
            and abs(float(speed_grid["0.15"]["mean_vx"]) - 0.15) <= 0.08
            and abs(float(speed_grid["0.30"]["mean_vx"]) - 0.30) <= 0.08
            and abs(float(speed_grid["0.45"]["mean_vx"]) - 0.45) <= 0.08
            and abs(float(yaw_grid["+0.20"]["mean_wz_rad_s"]) - 0.20) <= 0.12
            and abs(float(yaw_grid["-0.20"]["mean_wz_rad_s"]) + 0.20) <= 0.12
        )
        rows.append(
            {
                "iteration": iteration,
                "command_pass": command_pass,
                "style_pass": style_pass,
                "eligible": command_pass and style_pass,
                "failed_style_gates": failed_style_gates,
                "vx_015": float(speed_grid["0.15"]["mean_vx"]),
                "vx_030": float(speed_grid["0.30"]["mean_vx"]),
                "vx_045": float(speed_grid["0.45"]["mean_vx"]),
                "yaw_right_020": float(yaw_grid["-0.20"]["mean_wz_rad_s"]),
                "yaw_left_020": float(yaw_grid["+0.20"]["mean_wz_rad_s"]),
                "yaw_magnitude_gap": float(yaw["yaw_symmetry"]["magnitude_gap_rad_s"]),
            }
        )

    payload = {
        "schema": "sprite0825_g58f_screen_v1",
        "baseline_iteration": 925,
        "rows": rows,
        "eligible_iterations": [row["iteration"] for row in rows if row["eligible"]],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
