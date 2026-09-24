#!/usr/bin/env python3
"""Apply G72 motor-contract gates and select an Isaac-qualified winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_J4340 = {
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in (
        "hip_pitch",
        "hip_roll",
        "hip_yaw",
        "knee",
        "shoulder_pitch",
        "shoulder_roll",
    )
}
EXPECTED_ANKLE = {
    f"{side}_motor_{motor}"
    for side in ("left", "right")
    for motor in ("a", "b")
}


def worst(metrics: dict[str, dict[str, float]], field: str) -> float:
    return max(float(value[field]) for value in metrics.values())


def motor_gates(
    style: dict,
    key: str,
    expected_names: set[str],
    prefix: str,
) -> tuple[dict[str, float | list[str]], dict[str, bool]]:
    ratios = style[key]
    metric_names = {
        "rated_torque",
        "peak_torque",
        "rated_speed",
        "max_speed",
        "peak_torque_speed",
    }
    if set(ratios) != metric_names:
        raise ValueError(f"{key} metric names differ: {sorted(ratios)}")
    names = set(ratios["peak_torque_speed"])
    same_names = all(set(values) == names for values in ratios.values())
    values: dict[str, float | list[str]] = {
        "names": sorted(names),
        "rated_torque_rms_max": worst(ratios["rated_torque"], "rms"),
        "peak_torque_p99_max": worst(ratios["peak_torque"], "p99"),
        "rated_speed_rms_max": worst(ratios["rated_speed"], "rms"),
        "max_speed_p99_max": worst(ratios["max_speed"], "p99"),
        "directional_torque_speed_p99_max": worst(
            ratios["peak_torque_speed"], "p99"
        ),
        "directional_torque_speed_absolute_max": worst(
            ratios["peak_torque_speed"], "max"
        ),
    }
    gates = {
        f"{prefix}_exact_motor_names": same_names and names == expected_names,
        f"{prefix}_rated_torque_rms_le_1": values["rated_torque_rms_max"] <= 1.0,
        f"{prefix}_peak_torque_p99_le_1": values["peak_torque_p99_max"] <= 1.0,
        f"{prefix}_rated_speed_rms_le_1": values["rated_speed_rms_max"] <= 1.0,
        f"{prefix}_max_speed_p99_le_1": values["max_speed_p99_max"] <= 1.0,
        f"{prefix}_directional_torque_speed_p99_le_1": (
            values["directional_torque_speed_p99_max"] <= 1.0 + 1.0e-6
        ),
        f"{prefix}_directional_torque_speed_absolute_max_le_1": (
            values["directional_torque_speed_absolute_max"] <= 1.0 + 1.0e-6
        ),
    }
    return values, gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--bridge", required=True, type=Path)
    parser.add_argument("--winner", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=303)
    args = parser.parse_args()

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    for iteration, row in data["rows"].items():
        if not row.get("complete"):
            continue
        style_path = args.bridge / f"model{iteration}" / f"style4800_seed{args.seed}.json"
        style = json.loads(style_path.read_text(encoding="utf-8"))
        j4340_values, j4340_gates = motor_gates(
            style,
            "aligned_j4340p_motor_ratio_by_joint",
            EXPECTED_J4340,
            "j4340",
        )
        ankle_values, ankle_gates = motor_gates(
            style,
            "aligned_j4310p_motor_ratio_by_motor",
            EXPECTED_ANKLE,
            "ankle_j4310",
        )
        row["motor_contract"] = {"j4340": j4340_values, "ankle_j4310": ankle_values}
        row["gates"].update(j4340_gates)
        row["gates"].update(ankle_gates)
        row["passes_screen"] = all(row["gates"].values())
        row["failed_gates"] = [
            name for name, passed in row["gates"].items() if not passed
        ]

    passing = [
        iteration
        for iteration in data["ranked_candidates"]
        if data["rows"][str(iteration)].get("passes_screen")
    ]
    data["schema"] = "sprite0825_g72_multiobjective_checkpoint_screen_v1"
    data["evaluation_seed"] = args.seed
    data["motor_contract_finalized"] = True
    data["passing_candidates"] = passing
    if not passing:
        args.summary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        raise SystemExit("No G72 checkpoint passed the full Isaac and motor contract")
    args.winner.write_text(f"{passing[0]}\n", encoding="utf-8")
    args.summary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"G72_ISAAC_WINNER model_{passing[0]}.pt")


if __name__ == "__main__":
    main()
