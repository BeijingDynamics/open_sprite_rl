#!/usr/bin/env python3
"""Apply the Stage 2 <=10% V38 style-regression gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("schema") not in {"sprite0615_stage2_style_v1", "sprite0615_stage2_style_v2"}:
        raise ValueError(f"unsupported style schema in {path}")
    return data


def nested(data: dict, path: str) -> float:
    value = data
    for part in path.split("."):
        value = value[part]
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-regression", type=float, default=0.10)
    args = parser.parse_args()
    baseline = load(args.baseline)
    candidate = load(args.candidate)
    multiplier = 1.0 + args.max_regression

    lower_is_better = {
        "joint_phase_error": "phase_invariant_joint_l2.mean",
        "body_phase_error": "phase_invariant_body_mean_l2_m.mean",
        "physical_target_delta": "physical_target_delta_rms_rad.mean",
        "raw_action_second_delta": "raw_action_second_delta_rms.mean",
        "left_contact_slip": "left_contact_slip_m_s.mean",
        "right_contact_slip": "right_contact_slip_m_s.mean",
        "left_contact_tilt": "left_contact_tilt_rad.mean",
        "right_contact_tilt": "right_contact_tilt_rad.mean",
        "contact_duty_symmetry": "contact_duty_gap",
        "swing_height_symmetry": "swing_height_mean_gap_m",
    }
    gates = {}
    for name, path in lower_is_better.items():
        baseline_value = nested(baseline, path)
        candidate_value = nested(candidate, path)
        limit = baseline_value * multiplier
        gates[name] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "ratio": candidate_value / baseline_value if baseline_value else None,
            "limit": limit,
            "pass": candidate_value <= limit,
        }

    gates["survival"] = {
        "baseline": float(baseline["survival_rate"]),
        "candidate": float(candidate["survival_rate"]),
        "limit": 1.0,
        "pass": float(candidate["survival_rate"]) >= 1.0,
    }
    result = {
        "schema": "sprite0615_stage2_style_comparison_v1",
        "baseline": str(Path(args.baseline).resolve()),
        "candidate": str(Path(args.candidate).resolve()),
        "max_regression": args.max_regression,
        "baseline_mean_vx": nested(baseline, "root_vx.mean"),
        "candidate_mean_vx": nested(candidate, "root_vx.mean"),
        "diagnostics": {
            "baseline_phase_backward_jump_rate": nested(baseline, "phase_backward_jump_rate"),
            "candidate_phase_backward_jump_rate": nested(candidate, "phase_backward_jump_rate"),
            "note": "Nearest-phase jumps are diagnostic only because bilateral poses can alias by half a gait cycle.",
        },
        "gates": gates,
        "pass": all(gate["pass"] for gate in gates.values()),
        "failed_gates": [name for name, gate in gates.items() if not gate["pass"]],
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
