#!/usr/bin/env python3
"""Rank G60 checkpoints using deterministic gait metrics before full qualification."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TARGET_CADENCE = (135.0, 155.0)
TARGET_STEP_LENGTH = (0.10, 0.13)
TARGET_SPEED = 0.30


def interval_distance(value: float, limits: tuple[float, float]) -> float:
    low, high = limits
    if value < low:
        return (low - value) / (high - low)
    if value > high:
        return (value - high) / (high - low)
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-output", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.input.glob("model_*_gait24.json")):
        match = re.fullmatch(r"model_(\d+)_gait24\.json", path.name)
        if match is None:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        gait = data["gait"]
        iteration = int(match.group(1))
        survival = float(data["survival_rate"])
        cadence = float(gait["mean_cadence_steps_per_min"])
        step_length = float(gait["mean_estimated_step_length_m"])
        speed = float(gait["mean_path_speed_mps"])
        drift = float(gait["mean_abs_heading_drift_deg"])
        score = (
            20.0 * (1.0 - survival)
            + 2.5 * interval_distance(cadence, TARGET_CADENCE)
            + 2.5 * interval_distance(step_length, TARGET_STEP_LENGTH)
            + abs(speed - TARGET_SPEED) / 0.08
            + drift / 15.0
        )
        rows.append(
            {
                "iteration": iteration,
                "survival_rate": survival,
                "cadence_steps_min": cadence,
                "estimated_step_length_m": step_length,
                "path_speed_m_s": speed,
                "mean_abs_heading_drift_deg": drift,
                "cadence_in_target": TARGET_CADENCE[0] <= cadence <= TARGET_CADENCE[1],
                "step_length_in_target": TARGET_STEP_LENGTH[0] <= step_length <= TARGET_STEP_LENGTH[1],
                "score_lower_is_better": score,
            }
        )

    if not rows:
        raise RuntimeError(f"no gait screen JSON files found in {args.input}")
    ranked = sorted(rows, key=lambda row: (row["score_lower_is_better"], -row["iteration"]))
    g60_ranked = [row for row in ranked if row["iteration"] != 2999]
    top = g60_ranked[: args.top_k]
    result = {
        "schema": "sprite0825_g60_gait_screen_v1",
        "targets": {
            "cadence_steps_min": list(TARGET_CADENCE),
            "estimated_step_length_m": list(TARGET_STEP_LENGTH),
            "command_speed_m_s": TARGET_SPEED,
        },
        "baseline_iteration": 2999,
        "ranked": ranked,
        "top_g60_iterations": [row["iteration"] for row in top],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.top_output.write_text(
        " ".join(str(row["iteration"]) for row in top) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
