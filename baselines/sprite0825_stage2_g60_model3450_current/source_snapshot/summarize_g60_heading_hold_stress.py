#!/usr/bin/env python3
"""Compare open-loop and PM01-style heading-hold robust gait runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(root: Path, pattern: str) -> list[dict]:
    rows = []
    for path in sorted(root.glob(pattern)):
        data = json.loads(path.read_text(encoding="utf-8"))
        gait = data["gait"]
        rows.append(
            {
                "seed": int(path.stem.rsplit("seed", 1)[1]),
                "survival_rate": float(data["survival_rate"]),
                "mean_abs_heading_drift_deg": float(gait["mean_abs_heading_drift_deg"]),
                "cadence_steps_min": float(gait["mean_cadence_steps_per_min"]),
                "step_length_m": float(gait["mean_estimated_step_length_m"]),
                "path_speed_m_s": float(gait["mean_path_speed_mps"]),
                "heading_hold": bool(data["heading_hold"]),
                "heading_control_stiffness": data.get("heading_control_stiffness"),
                "heading_yaw_rate_limit_rad_s": data.get(
                    "heading_yaw_rate_limit_rad_s"
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open-loop", required=True, type=Path)
    parser.add_argument("--heading-hold", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    open_loop = load_rows(args.open_loop, "gait60_seed*.json")
    heading_hold = load_rows(args.heading_hold, "gait60_heading_hold_seed*.json")
    open_by_seed = {row["seed"]: row for row in open_loop}
    paired = [
        {
            "seed": row["seed"],
            "open_loop_heading_deg": open_by_seed[row["seed"]][
                "mean_abs_heading_drift_deg"
            ],
            "heading_hold_deg": row["mean_abs_heading_drift_deg"],
            "improvement_deg": open_by_seed[row["seed"]][
                "mean_abs_heading_drift_deg"
            ]
            - row["mean_abs_heading_drift_deg"],
        }
        for row in heading_hold
        if row["seed"] in open_by_seed
    ]
    gates = {
        "three_paired_seeds": len(paired) == 3,
        "survival_all": len(heading_hold) == 3
        and all(row["survival_rate"] >= 0.99 for row in heading_hold),
        "heading_hold_le_5deg": len(heading_hold) == 3
        and all(row["mean_abs_heading_drift_deg"] <= 5.0 for row in heading_hold),
        "heading_improves_each_seed": len(paired) == 3
        and all(row["improvement_deg"] > 0.0 for row in paired),
        "controller_contract_matches_runtime": len(heading_hold) == 3
        and all(
            row["heading_hold"]
            and row["heading_control_stiffness"] == 0.5
            and row["heading_yaw_rate_limit_rad_s"] == 0.2
            for row in heading_hold
        ),
        "cadence_120_to_150": len(heading_hold) == 3
        and all(120.0 <= row["cadence_steps_min"] <= 150.0 for row in heading_hold),
        "step_length_0p10_to_0p16m": len(heading_hold) == 3
        and all(0.10 <= row["step_length_m"] <= 0.16 for row in heading_hold),
        "path_speed_0p24_to_0p32m_s": len(heading_hold) == 3
        and all(0.24 <= row["path_speed_m_s"] <= 0.32 for row in heading_hold),
    }
    result = {
        "schema": "sprite0825_g60_model3450_heading_hold_stress_v1",
        "controller": {
            "type": "PM01 outer heading controller",
            "heading_control_stiffness": 0.5,
            "actor_global_yaw_observation": False,
        },
        "open_loop": open_loop,
        "heading_hold": heading_hold,
        "paired": paired,
        "gates": gates,
        "qualified": all(gates.values()),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
