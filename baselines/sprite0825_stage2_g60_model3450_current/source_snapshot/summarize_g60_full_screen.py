#!/usr/bin/env python3
"""Build the multi-objective G60 Stage 2 qualification table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(data: dict | None) -> float | None:
    return None if data is None else float(data["mean"])


def max_mapping(data: dict | None) -> float | None:
    if not data:
        return None
    values = [float(value) for value in data.values()]
    return max(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    gait_summary = load(args.input / "stage1_summary.json")
    gait_by_iteration = {
        int(row["iteration"]): row for row in gait_summary["ranked"]
    }
    iterations = [
        int(value)
        for value in (args.input / "stage2_iterations.txt").read_text().split()
    ]
    raw_rows = []
    for iteration in iterations:
        grid = load(args.input / f"model_{iteration}_grid.json")
        transitions = load(args.input / f"model_{iteration}_transitions.json")
        yaw = load(args.input / f"model_{iteration}_yaw30.json")
        style = load(args.input / f"model_{iteration}_style_vx030.json")
        speeds = grid["grid"]
        yaws = yaw["yaw_grid"]
        j4340_exceed = max_mapping(
            style.get("aligned_j4340p_peak_torque_speed_exceedance_fraction_by_joint")
        )
        ankle_exceed = max_mapping(
            style.get("aligned_j4310p_peak_torque_speed_exceedance_fraction_by_motor")
        )
        row = {
            **gait_by_iteration[iteration],
            "grid_survival": float(grid["survival_rate"]),
            "vx_000": float(speeds["0.00"]["mean_vx"]),
            "vx_015": float(speeds["0.15"]["mean_vx"]),
            "vx_030": float(speeds["0.30"]["mean_vx"]),
            "vx_045": float(speeds["0.45"]["mean_vx"]),
            "transition_survival": float(transitions["survival_rate"]),
            "complete_cycle_rate": float(transitions["complete_cycle_protocol_rate"]),
            "all_starts_within_1s_rate": float(transitions["all_starts_within_1s_rate"]),
            "all_stops_within_1p5s_rate": float(transitions["all_stops_within_1p5s_rate"]),
            "start_latency_median_s": float(transitions["start_latency_median_s"]),
            "stop_latency_median_s": float(transitions["stop_latency_median_s"]),
            "yaw_survival": float(yaw["survival_rate"]),
            "yaw_right_actual": float(yaws["-0.20"]["mean_wz_rad_s"]),
            "yaw_left_actual": float(yaws["+0.20"]["mean_wz_rad_s"]),
            "yaw_magnitude_gap": float(yaw["yaw_symmetry"]["magnitude_gap_rad_s"]),
            "style_survival": float(style["survival_rate"]),
            "style_root_vx": mean(style["root_vx"]),
            "style_cadence_steps_min": 60.0 * float(
                style["cadence"]["total_touchdown_rate_hz"]["mean"]
            ),
            "alternation_fraction": mean(style["cadence"]["alternation_fraction"]),
            "joint_phase_l2": mean(style["phase_invariant_joint_l2"]),
            "body_phase_l2_m": mean(style.get("phase_invariant_body_mean_l2_m")),
            "left_contact_tilt_rad": mean(style.get("left_contact_tilt_rad")),
            "right_contact_tilt_rad": mean(style.get("right_contact_tilt_rad")),
            "left_contact_slip_m_s": mean(style.get("left_contact_slip_m_s")),
            "right_contact_slip_m_s": mean(style.get("right_contact_slip_m_s")),
            "contact_duty_gap": float(style["contact_duty_gap"]),
            "swing_height_gap_m": float(style["swing_height_mean_gap_m"]),
            "j4340_max_aligned_envelope_exceedance_fraction": j4340_exceed,
            "j4310_max_aligned_envelope_exceedance_fraction": ankle_exceed,
            "contains_base_lin_vel": bool(style["contains_base_lin_vel"]),
        }
        raw_rows.append(row)

    baseline = next(row for row in raw_rows if row["iteration"] == 2999)
    rows = []
    for row in raw_rows:
        command_pass = (
            row["grid_survival"] == 1.0
            and abs(row["vx_000"]) <= 0.05
            and abs(row["vx_015"] - 0.15) <= 0.08
            and abs(row["vx_030"] - 0.30) <= 0.08
            and abs(row["vx_045"] - 0.45) <= 0.08
            and row["transition_survival"] == 1.0
            and row["complete_cycle_rate"] >= 0.99
            and row["all_starts_within_1s_rate"] >= 0.99
            and row["all_stops_within_1p5s_rate"] >= 0.99
            and row["yaw_survival"] == 1.0
            and abs(row["yaw_right_actual"] + 0.20) <= 0.12
            and abs(row["yaw_left_actual"] - 0.20) <= 0.12
        )
        motion_pass = (
            row["survival_rate"] == 1.0
            and row["cadence_in_target"]
            and row["step_length_in_target"]
            and row["style_survival"] == 1.0
            and row["alternation_fraction"] is not None
            and row["alternation_fraction"] >= 0.95
            and row["contact_duty_gap"] <= max(0.03, baseline["contact_duty_gap"] + 0.01)
            and row["swing_height_gap_m"] <= max(0.01, baseline["swing_height_gap_m"] + 0.005)
            and max(row["left_contact_tilt_rad"], row["right_contact_tilt_rad"])
            <= max(baseline["left_contact_tilt_rad"], baseline["right_contact_tilt_rad"]) + 0.02
            and max(row["left_contact_slip_m_s"], row["right_contact_slip_m_s"])
            <= max(baseline["left_contact_slip_m_s"], baseline["right_contact_slip_m_s"]) + 0.03
        )
        motor_pass = (
            row["j4340_max_aligned_envelope_exceedance_fraction"] is not None
            and row["j4340_max_aligned_envelope_exceedance_fraction"] <= 0.01
            and row["j4310_max_aligned_envelope_exceedance_fraction"] is not None
            and row["j4310_max_aligned_envelope_exceedance_fraction"] <= 0.001
        )
        row["command_pass"] = command_pass
        row["motion_pass"] = motion_pass
        row["motor_pass"] = motor_pass
        row["eligible_for_mujoco"] = (
            row["iteration"] != 2999
            and command_pass
            and motion_pass
            and motor_pass
            and not row["contains_base_lin_vel"]
        )
        rows.append(row)

    ranked = sorted(
        rows,
        key=lambda row: (
            not row["eligible_for_mujoco"],
            row["score_lower_is_better"],
            row["joint_phase_l2"],
        ),
    )
    result = {
        "schema": "sprite0825_g60_full_screen_v1",
        "baseline_iteration": 2999,
        "ranked": ranked,
        "eligible_for_mujoco": [
            row["iteration"] for row in ranked if row["eligible_for_mujoco"]
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
