#!/usr/bin/env python3
"""Summarize aligned Sprite0825 Stage 2 physical-motor qualification evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("inputs", nargs="+", type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()


def worst_named(group: dict[str, dict[str, float]], metric: str) -> dict[str, object]:
    name, values = max(group.items(), key=lambda item: float(item[1][metric]))
    return {"name": name, "value": float(values[metric])}


rows = []
for path in args.inputs:
    data = json.loads(path.read_text(encoding="utf-8"))
    j4340 = data["aligned_j4340p_motor_ratio_by_joint"]
    j4310 = data["aligned_j4310p_motor_ratio_by_motor"]
    rows.append(
        {
            "source": str(path.resolve()),
            "target_vx_m_s": float(data["target_vx_m_s"]),
            "target_yaw_rate_rad_s": float(data["target_yaw_rate_rad_s"]),
            "survival_rate": float(data["survival_rate"]),
            "actor_contains_horizontal_base_velocity": bool(data["contains_base_lin_vel"]),
            "j4340p": {
                "rated_torque_rms_worst": worst_named(j4340["rated_torque"], "rms"),
                "peak_torque_max_worst": worst_named(j4340["peak_torque"], "max"),
                "max_speed_worst": worst_named(j4340["max_speed"], "max"),
                "peak_torque_speed_max_worst": worst_named(j4340["peak_torque_speed"], "max"),
                "peak_torque_speed_p99_worst": worst_named(j4340["peak_torque_speed"], "p99"),
            },
            "j4310p": {
                "rated_torque_rms_worst": worst_named(j4310["rated_torque"], "rms"),
                "peak_torque_max_worst": worst_named(j4310["peak_torque"], "max"),
                "max_speed_worst": worst_named(j4310["max_speed"], "max"),
                "peak_torque_speed_max_worst": worst_named(j4310["peak_torque_speed"], "max"),
                "peak_torque_speed_p99_worst": worst_named(j4310["peak_torque_speed"], "p99"),
            },
        }
    )


def aggregate(motor: str, field: str) -> dict[str, object]:
    row = max(rows, key=lambda item: float(item[motor][field]["value"]))
    return {
        **row[motor][field],
        "target_vx_m_s": row["target_vx_m_s"],
        "target_yaw_rate_rad_s": row["target_yaw_rate_rad_s"],
    }


overall = {
    motor: {
        field: aggregate(motor, field)
        for field in (
            "rated_torque_rms_worst",
            "peak_torque_max_worst",
            "max_speed_worst",
            "peak_torque_speed_max_worst",
            "peak_torque_speed_p99_worst",
        )
    }
    for motor in ("j4340p", "j4310p")
}

tolerance = 1.0e-6
gates = {
    "all_scenarios_survive": all(row["survival_rate"] == 1.0 for row in rows),
    "actor_has_no_horizontal_base_velocity": not any(
        row["actor_contains_horizontal_base_velocity"] for row in rows
    ),
}
for motor in ("j4340p", "j4310p"):
    gates[f"{motor}_rated_torque_rms_le_1"] = overall[motor]["rated_torque_rms_worst"]["value"] <= 1.0
    gates[f"{motor}_peak_torque_le_1"] = overall[motor]["peak_torque_max_worst"]["value"] <= 1.0
    gates[f"{motor}_max_speed_le_1"] = overall[motor]["max_speed_worst"]["value"] <= 1.0
    gates[f"{motor}_peak_torque_speed_le_1"] = (
        overall[motor]["peak_torque_speed_max_worst"]["value"] <= 1.0 + tolerance
    )

result = {
    "schema": "sprite0825_stage2_aligned_motor_qualification_v1",
    "pairing": "actuator.applied_effort with actuator._joint_vel from the same physics substep",
    "float_tolerance": tolerance,
    "scenarios": rows,
    "overall": overall,
    "gates": gates,
    "qualified": all(gates.values()),
    "residual_risks": [
        "J4340P knee torque-speed p99 reaches the envelope boundary in high-speed or turning scenarios.",
        "Recheck knee margin in MuJoCo and on hardware telemetry even though all present limits pass.",
    ],
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"qualified": result["qualified"], "gates": gates, "overall": overall}, indent=2))
raise SystemExit(0 if result["qualified"] else 1)
