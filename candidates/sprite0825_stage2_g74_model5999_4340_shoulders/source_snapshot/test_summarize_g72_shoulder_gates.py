from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent


def summarizer_path() -> Path:
    name = "summarize_g62_multiobjective_checkpoint_screen.py"
    for candidate in (ROOT / name, ROOT.parent / name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(name)


def metric(value: float) -> dict[str, float]:
    return {"mean": value, "p90": value, "max": value}


def style(right_motion_ratio: float) -> dict:
    shoulders = (
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
    )
    return {
        "survival_rate": 1.0,
        "cadence": {
            "alternation_fraction": metric(1.0),
            "cadence_ratio_to_reference": metric(1.0),
        },
        "phase_invariant_joint_rmse": metric(0.03),
        "torso_roll_rad": {"abs": metric(0.04)},
        "torso_pitch_rad": {"abs": metric(0.03)},
        "foot_lateral_spacing_m": metric(0.12),
        "double_support_foot_lateral_spacing_m": metric(0.13),
        "contact_duty_gap": 0.01,
        "contact_slip_mean_gap_m_s": 0.01,
        "contact_tilt_mean_gap_rad": 0.01,
        "left_contact_slip_m_s": metric(0.02),
        "right_contact_slip_m_s": metric(0.02),
        "left_contact_tilt_rad": metric(0.02),
        "right_contact_tilt_rad": metric(0.02),
        "swing_height_mean_gap_m": 0.005,
        "aligned_j4340p_motor_ratio_by_joint": {
            "peak_torque_speed": {"joint": metric(0.5)}
        },
        "aligned_j4310p_motor_ratio_by_motor": {
            "peak_torque_speed": {"motor": metric(0.5)}
        },
        "contains_base_lin_vel": False,
        "joint_temporal_std_ratio_to_reference_by_name": {
            "left_shoulder_pitch_joint": 0.9,
            "right_shoulder_pitch_joint": right_motion_ratio,
        },
        "phase_invariant_joint_abs_by_name_rad": {
            name: metric(0.06) for name in shoulders
        },
    }


def run_case(root: Path, right_motion_ratio: float) -> dict:
    candidate = root / "model2500"
    candidate.mkdir(parents=True)
    (candidate / "yaw_transitions120_seed303.json").write_text(
        json.dumps(
            {
                "survival_rate": 1.0,
                "yaw_grid": {
                    "-0.20": {"mean_wz_rad_s": -0.20},
                    "+0.20": {"mean_wz_rad_s": 0.20},
                },
                "yaw_symmetry": {"magnitude_gap_rad_s": 0.0},
            }
        )
    )
    (candidate / "style4800_seed303.json").write_text(
        json.dumps(style(right_motion_ratio))
    )
    (candidate / "transitions20_seed303.json").write_text(
        json.dumps(
            {
                "survival_rate": 1.0,
                "complete_cycle_protocol_rate": 1.0,
                "all_starts_within_1s_rate": 1.0,
                "all_stops_within_1p5s_rate": 1.0,
            }
        )
    )
    output = root / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(summarizer_path()),
            "--input",
            str(root),
            "--transition-screen",
            str(root),
            "--output",
            str(output),
            "--candidates",
            "2500",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(output.read_text())


def test_shoulder_participation_passes_and_stall_fails() -> None:
    with TemporaryDirectory() as directory:
        passed = run_case(Path(directory) / "pass", 0.8)
        assert passed["rows"]["2500"]["passes_screen"]
    with TemporaryDirectory() as directory:
        failed = run_case(Path(directory) / "fail", 0.2)
        row = failed["rows"]["2500"]
        assert not row["passes_screen"]
        assert "shoulder_pitch_motion_ratio_0p50_to_1p80" in row["failed_gates"]


if __name__ == "__main__":
    test_shoulder_participation_passes_and_stall_fails()
    print("PASS")
