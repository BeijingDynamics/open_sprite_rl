#!/usr/bin/env python3
"""Regression checks for the G72 log-only trend parser."""

from pathlib import Path
from tempfile import TemporaryDirectory

from analyze_sprite0825_g72_training_trend import parse_log, summarize


def main() -> None:
    text = """
Learning iteration 2500/15000
Mean reward: 120.0
Episode_Reward/feet_orientation: 0.15
Episode_Reward/arm_pitch_position: 0.07
Episode_Termination/illegal_contact: 0.001
Learning iteration 5000/15000
Mean reward: 134.0
Episode_Reward/feet_orientation: 0.17
Episode_Reward/arm_pitch_position: 0.08
Episode_Termination/illegal_contact: 0.000
"""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "train.log"
        path.write_text(text)
        rows = parse_log(path)

    result = summarize(rows, window=1, formal_steps=[2500, 5000, 7500])
    assert result["iteration_latest"] == 5000
    assert result["latest"]["feet_orientation_reward"] == 0.17
    assert result["latest"]["arm_pitch_position_reward"] == 0.08
    assert set(result["formal_snapshots"]) == {"2500", "5000"}
    assert result["formal_snapshots"]["2500"]["mean_reward"] == 120.0
    print("G72_TRAINING_TREND_TEST_PASS")


if __name__ == "__main__":
    main()
