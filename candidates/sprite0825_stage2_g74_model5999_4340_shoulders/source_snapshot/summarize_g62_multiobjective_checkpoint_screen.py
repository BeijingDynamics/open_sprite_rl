#!/usr/bin/env python3
"""Summarize exact G62 yaw/style tests without changing qualification gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_CANDIDATES = (3600, 3650, 3700, 3750, 3800, 3850, 3899)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(metric: dict) -> float:
    return float(metric["mean"])


def p90(metric: dict) -> float:
    return float(metric["p90"])


def nested_max(metrics: dict) -> float:
    return max(float(metric["max"]) for metric in metrics.values())


def transition_row(path: Path) -> dict | None:
    if not path.is_file():
        return None
    d = load(path)
    return {
        "survival_rate": float(d["survival_rate"]),
        "complete_rate": float(d["complete_cycle_protocol_rate"]),
        "starts_within_1s_rate": float(d["all_starts_within_1s_rate"]),
        "stops_within_1p5s_rate": float(d["all_stops_within_1p5s_rate"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--transition-screen", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidates", nargs="+", type=int)
    parser.add_argument("--seed", type=int, default=303)
    args = parser.parse_args()

    candidates = tuple(args.candidates or DEFAULT_CANDIDATES)

    rows: dict[str, dict] = {}
    for iteration in candidates:
        root = args.input / f"model{iteration}"
        yaw_path = root / f"yaw_transitions120_seed{args.seed}.json"
        style_path = root / f"style4800_seed{args.seed}.json"
        if not yaw_path.is_file() or not style_path.is_file():
            rows[str(iteration)] = {
                "complete": False,
                "missing": [str(p) for p in (yaw_path, style_path) if not p.is_file()],
            }
            continue

        yaw = load(yaw_path)
        style = load(style_path)
        shoulder_pitch_names = (
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
        )
        shoulder_names = shoulder_pitch_names + (
            "left_shoulder_roll_joint",
            "right_shoulder_roll_joint",
        )
        shoulder_motion_ratio = style[
            "joint_temporal_std_ratio_to_reference_by_name"
        ]
        shoulder_tracking = style["phase_invariant_joint_abs_by_name_rad"]
        right = float(yaw["yaw_grid"]["-0.20"]["mean_wz_rad_s"])
        left = float(yaw["yaw_grid"]["+0.20"]["mean_wz_rad_s"])
        local_transition = root / f"transitions20_seed{args.seed}.json"
        transition = transition_row(local_transition)
        if transition is None:
            transition = transition_row(
                args.transition_screen
                / "stage_a"
                / f"model{iteration}_transitions20_seed{args.seed}.json"
            )
        metrics = {
            "yaw_survival_rate": float(yaw["survival_rate"]),
            "right_actual_rad_s": right,
            "left_actual_rad_s": left,
            "right_error_rad_s": abs(right + 0.20),
            "left_error_rad_s": abs(left - 0.20),
            "yaw_magnitude_gap_rad_s": float(yaw["yaw_symmetry"]["magnitude_gap_rad_s"]),
            "style_survival_rate": float(style["survival_rate"]),
            "alternation_fraction": mean(style["cadence"]["alternation_fraction"]),
            "joint_tracking_rmse_rad": mean(style["phase_invariant_joint_rmse"]),
            "cadence_ratio_to_reference": mean(style["cadence"]["cadence_ratio_to_reference"]),
            "torso_roll_abs_mean_rad": mean(style["torso_roll_rad"]["abs"]),
            "torso_roll_abs_p90_rad": p90(style["torso_roll_rad"]["abs"]),
            "torso_pitch_abs_mean_rad": mean(style["torso_pitch_rad"]["abs"]),
            "torso_pitch_abs_p90_rad": p90(style["torso_pitch_rad"]["abs"]),
            "foot_lateral_spacing_mean_m": mean(style["foot_lateral_spacing_m"]),
            "double_support_foot_lateral_spacing_mean_m": mean(
                style["double_support_foot_lateral_spacing_m"]
            ),
            "contact_duty_gap": float(style["contact_duty_gap"]),
            "contact_slip_mean_gap_m_s": float(style["contact_slip_mean_gap_m_s"]),
            "contact_tilt_mean_gap_rad": float(style["contact_tilt_mean_gap_rad"]),
            "contact_slip_mean_max_m_s": max(
                mean(style["left_contact_slip_m_s"]), mean(style["right_contact_slip_m_s"])
            ),
            "contact_tilt_mean_max_rad": max(
                mean(style["left_contact_tilt_rad"]), mean(style["right_contact_tilt_rad"])
            ),
            "swing_height_gap_m": float(style["swing_height_mean_gap_m"]),
            "j4340_envelope_max": nested_max(
                style["aligned_j4340p_motor_ratio_by_joint"]["peak_torque_speed"]
            ),
            "j4310_envelope_max": nested_max(
                style["aligned_j4310p_motor_ratio_by_motor"]["peak_torque_speed"]
            ),
            "contains_base_lin_vel": bool(style["contains_base_lin_vel"]),
            "shoulder_pitch_motion_ratio_min": min(
                float(shoulder_motion_ratio[name]) for name in shoulder_pitch_names
            ),
            "shoulder_pitch_motion_ratio_max": max(
                float(shoulder_motion_ratio[name]) for name in shoulder_pitch_names
            ),
            "shoulder_tracking_p90_max_rad": max(
                float(shoulder_tracking[name]["p90"]) for name in shoulder_names
            ),
        }
        gates = {
            "stage_a_transition_gate": bool(
                transition
                and transition["survival_rate"] >= 0.99
                and transition["complete_rate"] >= 0.98
                and transition["starts_within_1s_rate"] >= 0.99
                and transition["stops_within_1p5s_rate"] >= 0.99
            ),
            "yaw_survival_ge_0p99": metrics["yaw_survival_rate"] >= 0.99,
            "yaw_response_within_0p04": (
                metrics["right_error_rad_s"] <= 0.04 and metrics["left_error_rad_s"] <= 0.04
            ),
            "yaw_magnitude_gap_le_0p04": metrics["yaw_magnitude_gap_rad_s"] <= 0.04,
            "style_survival_ge_0p99": metrics["style_survival_rate"] >= 0.99,
            "style_alternation_ge_0p98": metrics["alternation_fraction"] >= 0.98,
            "style_tracking_rmse_le_0p06rad": metrics["joint_tracking_rmse_rad"] <= 0.06,
            "style_cadence_ratio_0p75_to_1p10": 0.75 <= metrics["cadence_ratio_to_reference"] <= 1.10,
            "style_torso_roll_mean_le_0p12rad": metrics["torso_roll_abs_mean_rad"] <= 0.12,
            "style_torso_roll_p90_le_0p20rad": metrics["torso_roll_abs_p90_rad"] <= 0.20,
            "style_torso_pitch_mean_le_0p06rad": metrics["torso_pitch_abs_mean_rad"] <= 0.06,
            "style_torso_pitch_p90_le_0p12rad": metrics["torso_pitch_abs_p90_rad"] <= 0.12,
            "style_foot_spacing_0p09_to_0p18m": 0.09 <= metrics["foot_lateral_spacing_mean_m"] <= 0.18,
            "style_double_support_spacing_le_0p19m": metrics["double_support_foot_lateral_spacing_mean_m"] <= 0.19,
            "style_swing_height_gap_le_0p01m": metrics["swing_height_gap_m"] <= 0.01,
            "style_contact_duty_gap_le_0p05": metrics["contact_duty_gap"] <= 0.05,
            "style_contact_slip_gap_le_0p02m_s": metrics["contact_slip_mean_gap_m_s"] <= 0.02,
            "style_contact_tilt_gap_le_0p02rad": metrics["contact_tilt_mean_gap_rad"] <= 0.02,
            "style_contact_slip_mean_le_0p05m_s": metrics["contact_slip_mean_max_m_s"] <= 0.05,
            "style_contact_tilt_mean_le_0p035rad": metrics["contact_tilt_mean_max_rad"] <= 0.035,
            "j4340_envelope_le_1_plus_float32_eps": metrics["j4340_envelope_max"] <= 1.0 + 1.0e-6,
            "j4310_envelope_le_1_plus_float32_eps": metrics["j4310_envelope_max"] <= 1.0 + 1.0e-6,
            "actor_excludes_base_lin_vel": not metrics["contains_base_lin_vel"],
            "shoulder_pitch_motion_ratio_0p50_to_1p80": (
                metrics["shoulder_pitch_motion_ratio_min"] >= 0.50
                and metrics["shoulder_pitch_motion_ratio_max"] <= 1.80
            ),
            "shoulder_tracking_p90_le_0p12rad": (
                metrics["shoulder_tracking_p90_max_rad"] <= 0.12
            ),
        }
        rows[str(iteration)] = {
            "complete": True,
            "transition_stage_a": transition,
            "metrics": metrics,
            "gates": gates,
            "passes_screen": all(gates.values()),
            "failed_gates": [name for name, passed in gates.items() if not passed],
        }

    complete = all(row["complete"] for row in rows.values())
    passing = [int(key) for key, row in rows.items() if row.get("passes_screen")]
    ranked = sorted(
        (int(key) for key, row in rows.items() if row.get("complete")),
        key=lambda iteration: (
            len(rows[str(iteration)]["failed_gates"]),
            max(
                rows[str(iteration)]["metrics"]["right_error_rad_s"],
                rows[str(iteration)]["metrics"]["left_error_rad_s"],
            ),
            rows[str(iteration)]["metrics"]["torso_roll_abs_mean_rad"],
            -rows[str(iteration)]["metrics"]["shoulder_pitch_motion_ratio_min"],
            rows[str(iteration)]["metrics"]["shoulder_tracking_p90_max_rad"],
            -iteration,
        ),
    )
    result = {
        "schema": "sprite0825_g62_multiobjective_checkpoint_screen_v1",
        "evaluation_seed": args.seed,
        "fixed_gates": True,
        "complete": complete,
        "passing_candidates": passing,
        "ranked_candidates": ranked,
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
