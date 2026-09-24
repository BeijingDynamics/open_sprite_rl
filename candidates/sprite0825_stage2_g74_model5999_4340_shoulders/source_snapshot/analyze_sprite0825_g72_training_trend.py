#!/usr/bin/env python3
"""Summarize G72 RSL-RL training metrics without loading Isaac Sim."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ITERATION_RE = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
METRIC_RE = re.compile(r"^\s*([^:]+):\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$")
WANTED = {
    "Mean reward": "mean_reward",
    "Mean episode length": "mean_episode_length",
    "Mean action std": "mean_action_std",
    "Metrics/base_velocity/error_vel_xy": "error_vel_xy",
    "Metrics/base_velocity/error_vel_yaw": "error_vel_yaw",
    "Episode_Termination/time_out": "time_out",
    "Episode_Termination/illegal_contact": "illegal_contact",
    "Step_Reward/style_reward": "style_reward",
    "Step_Reward/task_reward": "task_reward",
    "Episode_Reward/feet_slide": "feet_slide",
    "Episode_Reward/feet_orientation": "feet_orientation_reward",
    "Episode_Reward/feet_contact": "feet_contact_reward",
    "Episode_Reward/feet_air_time_dense": "feet_air_time_dense_reward",
    "Episode_Reward/foot_stumble": "foot_stumble",
    "Episode_Reward/action_rate": "action_rate",
    "Episode_Reward/action_smoothness": "action_smoothness",
    "Episode_Reward/arm_pitch_position": "arm_pitch_position_reward",
    "Episode_Reward/arm_roll_position": "arm_roll_position_reward",
    "Episode_Reward/arm_yaw_position": "arm_yaw_position_reward",
    "Episode_Reward/joint_torque": "joint_torque_reward",
}


def parse_log(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    current: dict[str, float | int] | None = None
    for raw in path.read_text(errors="replace").splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw)
        iteration_match = ITERATION_RE.search(line)
        if iteration_match:
            if current is not None:
                rows.append(current)
            current = {
                "iteration": int(iteration_match.group(1)),
                "max_iterations": int(iteration_match.group(2)),
            }
            continue
        if current is None:
            continue
        metric_match = METRIC_RE.match(line)
        if metric_match and metric_match.group(1).strip() in WANTED:
            current[WANTED[metric_match.group(1).strip()]] = float(metric_match.group(2))
    if current is not None:
        rows.append(current)
    return [row for row in rows if "mean_reward" in row]


def mean(rows: list[dict[str, float | int]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row]
    return sum(values) / len(values) if values else None


def summarize(
    rows: list[dict[str, float | int]], window: int, formal_steps: list[int]
) -> dict:
    metrics = sorted({key for row in rows for key in row} - {"iteration", "max_iterations"})
    windows = []
    for end in range(window, len(rows) + window, window):
        chunk = rows[max(0, end - window) : min(end, len(rows))]
        if not chunk:
            continue
        windows.append(
            {
                "iteration_start": chunk[0]["iteration"],
                "iteration_end": chunk[-1]["iteration"],
                "samples": len(chunk),
                **{key: mean(chunk, key) for key in metrics},
            }
        )
    latest = rows[-1]
    by_iteration = {int(row["iteration"]): row for row in rows}
    return {
        "samples": len(rows),
        "iteration_first": rows[0]["iteration"],
        "iteration_latest": latest["iteration"],
        "max_iterations": latest["max_iterations"],
        "latest": latest,
        "window_size_samples": window,
        "windows": windows,
        "formal_snapshots": {
            str(step): by_iteration[step]
            for step in formal_steps
            if step in by_iteration
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument(
        "--formal-steps",
        type=int,
        nargs="*",
        default=[2500, 5000, 7500, 10000, 12500, 14999],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = parse_log(args.log)
    if not rows:
        raise SystemExit("no complete learning-iteration metric blocks found")
    result = summarize(rows, args.window, args.formal_steps)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
