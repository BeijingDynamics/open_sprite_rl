#!/usr/bin/env python3
"""Summarize the G60 model3450 robust-task stress matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(metric: dict) -> float:
    return float(metric["mean"])


def maximum(values: dict[str, float]) -> float:
    return max(float(value) for value in values.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    gait = []
    for path in sorted(args.input.glob("gait60_seed*.json")):
        data = load(path)
        metrics = data["gait"]
        gait.append(
            {
                "seed": int(path.stem.rsplit("seed", 1)[1]),
                "survival_rate": float(data["survival_rate"]),
                "cadence_steps_min": float(metrics["mean_cadence_steps_per_min"]),
                "estimated_step_length_m": float(metrics["mean_estimated_step_length_m"]),
                "path_speed_m_s": float(metrics["mean_path_speed_mps"]),
                "mean_abs_heading_drift_deg": float(metrics["mean_abs_heading_drift_deg"]),
            }
        )
    transitions = load(args.input / "transitions20_seed43.json")
    yaw = load(args.input / "yaw60_seed43.json")
    style = load(args.input / "style2400_seed43.json")
    rows = {
        "gait": gait,
        "transitions": {
            "survival_rate": float(transitions["survival_rate"]),
            "complete_cycle_protocol_rate": float(transitions["complete_cycle_protocol_rate"]),
            "all_starts_within_1s_rate": float(transitions["all_starts_within_1s_rate"]),
            "all_stops_within_1p5s_rate": float(transitions["all_stops_within_1p5s_rate"]),
        },
        "yaw": {
            "survival_rate": float(yaw["survival_rate"]),
            "right_actual_rad_s": float(yaw["yaw_grid"]["-0.20"]["mean_wz_rad_s"]),
            "left_actual_rad_s": float(yaw["yaw_grid"]["+0.20"]["mean_wz_rad_s"]),
            "magnitude_gap_rad_s": float(yaw["yaw_symmetry"]["magnitude_gap_rad_s"]),
        },
        "style": {
            "survival_rate": float(style["survival_rate"]),
            "root_vx_m_s": mean(style["root_vx"]),
            "cadence_steps_min": 60.0 * mean(style["cadence"]["total_touchdown_rate_hz"]),
            "alternation_fraction": mean(style["cadence"]["alternation_fraction"]),
            "contact_duty_gap": float(style["contact_duty_gap"]),
            "swing_height_gap_m": float(style["swing_height_mean_gap_m"]),
            "joint_tracking_rmse_rad": mean(style["phase_invariant_joint_rmse"]),
            "left_contact_slip_mean_m_s": mean(style["left_contact_slip_m_s"]),
            "right_contact_slip_mean_m_s": mean(style["right_contact_slip_m_s"]),
            "left_contact_tilt_mean_rad": mean(style["left_contact_tilt_rad"]),
            "right_contact_tilt_mean_rad": mean(style["right_contact_tilt_rad"]),
            "left_contact_tilt_p99_rad": float(style["left_contact_tilt_rad"]["p99"]),
            "right_contact_tilt_p99_rad": float(style["right_contact_tilt_rad"]["p99"]),
            "left_swing_height_mean_m": mean(style["left_swing_link_height_m"]),
            "right_swing_height_mean_m": mean(style["right_swing_link_height_m"]),
            "j4340_envelope_exceedance_fraction_max": maximum(
                style["aligned_j4340p_peak_torque_speed_exceedance_fraction_by_joint"]
            ),
            "j4310_envelope_exceedance_fraction_max": maximum(
                style["aligned_j4310p_peak_torque_speed_exceedance_fraction_by_motor"]
            ),
            "contains_base_lin_vel": bool(style["contains_base_lin_vel"]),
        },
    }
    gates = {
        "three_gait_seeds_present": len(gait) == 3,
        "gait_survival_ge_0p95": all(row["survival_rate"] >= 0.95 for row in gait),
        "gait_cadence_ge_120": all(row["cadence_steps_min"] >= 120.0 for row in gait),
        "gait_mean_abs_heading_drift_le_12deg": all(
            row["mean_abs_heading_drift_deg"] <= 12.0 for row in gait
        ),
        "transition_survival_ge_0p95": rows["transitions"]["survival_rate"] >= 0.95,
        "transition_complete_ge_0p90": (
            rows["transitions"]["complete_cycle_protocol_rate"] >= 0.90
        ),
        "yaw_survival_ge_0p95": rows["yaw"]["survival_rate"] >= 0.95,
        "yaw_response_within_0p12": (
            abs(rows["yaw"]["right_actual_rad_s"] + 0.20) <= 0.12
            and abs(rows["yaw"]["left_actual_rad_s"] - 0.20) <= 0.12
        ),
        "style_survival_ge_0p95": rows["style"]["survival_rate"] >= 0.95,
        "style_alternation_ge_0p95": rows["style"]["alternation_fraction"] >= 0.95,
        "joint_tracking_rmse_le_0p06rad": (
            rows["style"]["joint_tracking_rmse_rad"] <= 0.06
        ),
        "contact_slip_mean_le_0p05m_s": max(
            rows["style"]["left_contact_slip_mean_m_s"],
            rows["style"]["right_contact_slip_mean_m_s"],
        )
        <= 0.05,
        "contact_tilt_mean_le_0p035rad": max(
            rows["style"]["left_contact_tilt_mean_rad"],
            rows["style"]["right_contact_tilt_mean_rad"],
        )
        <= 0.035,
        "contact_tilt_p99_le_0p30rad": max(
            rows["style"]["left_contact_tilt_p99_rad"],
            rows["style"]["right_contact_tilt_p99_rad"],
        )
        <= 0.30,
        "swing_height_gap_le_0p01m": rows["style"]["swing_height_gap_m"] <= 0.01,
        "j4340_exceedance_le_0p01": (
            rows["style"]["j4340_envelope_exceedance_fraction_max"] <= 0.01
        ),
        "j4310_exceedance_le_0p001": (
            rows["style"]["j4310_envelope_exceedance_fraction_max"] <= 0.001
        ),
        "actor_excludes_base_lin_vel": not rows["style"]["contains_base_lin_vel"],
    }
    result = {
        "schema": "sprite0825_g60_model3450_robust_stress_v1",
        "rows": rows,
        "gates": gates,
        "qualified": all(gates.values()),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
