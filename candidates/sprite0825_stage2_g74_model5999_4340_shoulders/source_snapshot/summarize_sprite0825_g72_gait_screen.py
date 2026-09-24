#!/usr/bin/env python3
"""Rank G72 checkpoints using fixed gait, heading, and loaded-foot gates."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def checkpoint_iteration(path: Path) -> int:
    match = re.search(r"model[_-]?(\d+)", path.stem)
    if not match:
        raise ValueError(f"cannot parse checkpoint iteration from {path}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-output", required=True, type=Path)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument(
        "--diagnostic-fallback",
        action="store_true",
        help="Write the best unqualified candidates for deeper diagnosis without qualifying them.",
    )
    args = parser.parse_args()

    rows = []
    for path in sorted(args.input.glob("model*_gait*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        gait = data["gait"]
        feet = gait["contact_foot_roll"]
        max_roll_mean = max(feet[side]["mean_abs_rad"] for side in ("left", "right"))
        max_roll_p90 = max(feet[side]["p90_abs_rad"] for side in ("left", "right"))
        max_roll_over = max(
            feet[side]["over_0p06_fraction"] for side in ("left", "right")
        )
        roll_symmetry = abs(
            abs(feet["left"]["signed_mean_rad"])
            - abs(feet["right"]["signed_mean_rad"])
        )
        speed = float(gait["mean_path_speed_mps"])
        cadence = float(gait["mean_cadence_steps_per_min"])
        step_length = float(gait["mean_estimated_step_length_m"])
        heading = float(gait["mean_abs_heading_drift_deg"])
        gates = {
            "survival_ge_0p995": data["survival_rate"] >= 0.995,
            "speed_0p23_to_0p34": 0.23 <= speed <= 0.34,
            "cadence_120_to_170": 120.0 <= cadence <= 170.0,
            "step_length_0p08_to_0p17": 0.08 <= step_length <= 0.17,
            "heading_drift_le_12deg": heading <= 12.0,
            "contact_roll_mean_le_0p02": max_roll_mean <= 0.02,
            "contact_roll_p90_le_0p05": max_roll_p90 <= 0.05,
            "contact_roll_over_0p06_le_0p01": max_roll_over <= 0.01,
            "contact_roll_symmetry_le_0p01": roll_symmetry <= 0.01,
        }
        score = (
            25.0 * max_roll_mean
            + 5.0 * max_roll_p90
            + 4.0 * roll_symmetry
            + abs(speed - 0.26) / 0.08
            + abs(cadence - 148.148148) / 45.0
            + abs(step_length - 0.105) / 0.08
            + heading / 12.0
            + 100.0 * (1.0 - data["survival_rate"])
        )
        rows.append(
            {
                "iteration": checkpoint_iteration(path),
                "checkpoint": data["checkpoint"],
                "evaluation": str(path),
                "survival_rate": data["survival_rate"],
                "speed_m_s": speed,
                "cadence_steps_min": cadence,
                "step_length_m": step_length,
                "heading_drift_deg": heading,
                "max_contact_roll_mean_rad": max_roll_mean,
                "max_contact_roll_p90_rad": max_roll_p90,
                "max_contact_roll_over_0p06_fraction": max_roll_over,
                "contact_roll_symmetry_rad": roll_symmetry,
                "gates": gates,
                "qualified": all(gates.values()),
                "score": score,
            }
        )
    if not rows:
        raise SystemExit("no G72 gait evaluations found")
    ranked = sorted(rows, key=lambda row: (not row["qualified"], row["score"], row["iteration"]))
    qualified = [row for row in ranked if row["qualified"]]
    selected = qualified[: args.top]
    diagnostic_selected = [] if selected else ranked[: args.top]
    top_rows = selected or (diagnostic_selected if args.diagnostic_fallback else [])
    summary = {
        "schema": "sprite0825_g72_gait_screen_v1",
        "reference_cadence_steps_min": 148.148148,
        "fixed_gates": True,
        "rows": ranked,
        "selected": selected,
        "diagnostic_fallback_enabled": args.diagnostic_fallback,
        "diagnostic_fallback_used": bool(not selected and top_rows),
        "diagnostic_selected": diagnostic_selected if args.diagnostic_fallback else [],
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.top_output.write_text(
        "".join(f'{row["checkpoint"]}\n' for row in top_rows),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not summary["selected"] and not args.diagnostic_fallback:
        raise SystemExit("no checkpoint passed the fixed G72 gait screen")


if __name__ == "__main__":
    main()
