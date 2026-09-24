#!/usr/bin/env python3
"""Summarize gait, posture, spacing, and reconstructed sole contact from a trace."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np


def percentile_abs(values: np.ndarray, q: float) -> float:
    return float(np.percentile(np.abs(values), q))


def contact_metric(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        raise ValueError("trace contains no reconstructed sole-contact samples")
    return {
        "mean": float(values.mean()),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def signed_metric(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        raise ValueError("trace contains no signed sole-contact samples")
    absolute = np.abs(values)
    return {
        "signed_mean": float(values.mean()),
        "mean_abs": float(absolute.mean()),
        "p90_abs": float(np.percentile(absolute, 90)),
        "p99_abs": float(np.percentile(absolute, 99)),
        "max_abs": float(absolute.max()),
        "over_0p06_fraction": float((absolute > 0.06).mean()),
    }


def longest_true_run(mask: np.ndarray, dt: float) -> float:
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return float(longest * dt)


def dominant_cycle_frequency(times: np.ndarray, joint_pos: np.ndarray) -> tuple[float, float]:
    centered = joint_pos - joint_pos.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    usable = scale[0] > 1.0e-4
    normalized = centered[:, usable] / scale[:, usable]
    temporal_mode = np.linalg.svd(normalized, full_matrices=False)[0][:, 0]
    dt = float(np.median(np.diff(times)))
    frequencies = np.fft.rfftfreq(len(temporal_mode), dt)
    power = np.abs(np.fft.rfft(temporal_mode - temporal_mode.mean())) ** 2
    band = (frequencies >= 0.35) & (frequencies <= 3.0)
    index = np.flatnonzero(band)[np.argmax(power[band])]
    band_fraction = float(power[index] / max(power[band].sum(), 1.0e-12))
    return float(frequencies[index]), band_fraction


def touchdown_indices(
    times: np.ndarray,
    contact: np.ndarray,
    *,
    minimum_airborne_s: float = 0.08,
    minimum_event_interval_s: float = 0.20,
) -> list[int]:
    """Return debounced landings that follow a meaningful airborne interval."""
    events: list[int] = []
    airborne_start: int | None = None
    for index in range(1, len(contact)):
        if not contact[index] and contact[index - 1]:
            airborne_start = index
        elif contact[index] and not contact[index - 1]:
            if airborne_start is None:
                continue
            airborne_s = float(times[index] - times[airborne_start])
            separated = not events or float(times[index] - times[events[-1]]) >= minimum_event_interval_s
            if airborne_s >= minimum_airborne_s and separated:
                events.append(index)
            airborne_start = None
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--steady-after", type=float, default=10.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    rows = [row for row in trace if float(row["time_s"]) >= args.steady_after]
    if len(rows) < 100:
        raise ValueError("trace has too few steady-state samples")

    times = np.asarray([row["time_s"] for row in rows], dtype=np.float64)
    root_pos = np.asarray([row["root_pos_w"] for row in rows], dtype=np.float64)
    pelvis_xmat = np.asarray([row["pelvis_xmat"] for row in rows], dtype=np.float64)
    joint_pos = np.asarray([row["joint_pos"] for row in rows], dtype=np.float64)
    joint_names = list(contract["joint_names"])
    name_to_index = {name: i for i, name in enumerate(joint_names)}

    roll = np.arctan2(pelvis_xmat[:, 2, 1], pelvis_xmat[:, 2, 2])
    pitch = np.arctan2(
        -pelvis_xmat[:, 2, 0],
        np.sqrt(pelvis_xmat[:, 2, 1] ** 2 + pelvis_xmat[:, 2, 2] ** 2),
    )
    cycle_hz, spectral_fraction = dominant_cycle_frequency(times, joint_pos)
    duration = float(times[-1] - times[0])
    displacement = root_pos[-1] - root_pos[0]
    forward_speed = float(displacement[0] / duration)
    step_hz = 2.0 * cycle_hz

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)
    left_ankle_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"
    )
    right_ankle_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"
    )
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "waist_yaw_link")
    assert min(left_ankle_id, right_ankle_id, pelvis_id, torso_id) >= 0
    qpos_addresses = [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in joint_names
    ]
    ankle_lateral_separation = []
    ankle_forward_separation = []
    torso_roll = []
    torso_pitch = []
    sole_center_local = np.asarray([-0.005, 0.0, -0.034], dtype=np.float64)
    sole_half_size = np.asarray([0.052, 0.022, 0.005], dtype=np.float64)
    sole_corners_local = np.asarray(
        [
            sole_center_local + sole_half_size * np.asarray([sx, sy, sz])
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
    )
    contact_height_threshold_m = 0.003
    foot_samples = {
        "left": {
            "body_id": left_ankle_id, "center": [], "contact": [],
            "tilt": [], "roll": [], "pitch": [], "yaw": [],
        },
        "right": {
            "body_id": right_ankle_id, "center": [], "contact": [],
            "tilt": [], "roll": [], "pitch": [], "yaw": [],
        },
    }
    for row in rows:
        data.qpos[:3] = row["root_pos_w"]
        data.qpos[3:7] = row["root_quat_w"]
        for address, value in zip(qpos_addresses, row["joint_pos"], strict=True):
            data.qpos[address] = value
        mujoco.mj_forward(model, data)
        delta_w = data.xpos[left_ankle_id] - data.xpos[right_ankle_id]
        pelvis_rotation = data.xmat[pelvis_id].reshape(3, 3)
        delta_b = pelvis_rotation.T @ delta_w
        ankle_forward_separation.append(abs(float(delta_b[0])))
        ankle_lateral_separation.append(abs(float(delta_b[1])))
        torso_rotation = data.xmat[torso_id].reshape(3, 3)
        torso_roll.append(math.atan2(torso_rotation[2, 1], torso_rotation[2, 2]))
        torso_pitch.append(
            math.atan2(
                -torso_rotation[2, 0],
                math.sqrt(torso_rotation[2, 1] ** 2 + torso_rotation[2, 2] ** 2),
            )
        )
        for foot in foot_samples.values():
            body_id = foot["body_id"]
            foot_rotation = data.xmat[body_id].reshape(3, 3)
            foot_position = data.xpos[body_id]
            sole_center_w = foot_position + foot_rotation @ sole_center_local
            sole_corners_w = foot_position + sole_corners_local @ foot_rotation.T
            minimum_height = float(sole_corners_w[:, 2].min())
            contact = minimum_height <= contact_height_threshold_m
            foot_rotation_b = pelvis_rotation.T @ foot_rotation
            foot["center"].append(sole_center_w.copy())
            foot["contact"].append(contact)
            foot["tilt"].append(
                math.acos(float(np.clip(foot_rotation[2, 2], -1.0, 1.0)))
            )
            foot["roll"].append(math.atan2(foot_rotation[2, 1], foot_rotation[2, 2]))
            foot["pitch"].append(
                math.atan2(
                    -foot_rotation[2, 0],
                    math.sqrt(foot_rotation[2, 1] ** 2 + foot_rotation[2, 2] ** 2),
                )
            )
            foot["yaw"].append(
                math.atan2(foot_rotation_b[1, 0], foot_rotation_b[0, 0])
            )

    sole_contact = {}
    touchdown_events: list[dict[str, float | int | str]] = []
    dt = np.diff(times)
    for side, foot in foot_samples.items():
        centers = np.asarray(foot["center"])
        contact = np.asarray(foot["contact"], dtype=bool)
        contact_pairs = contact[1:] & contact[:-1]
        slip_speed = np.linalg.norm(np.diff(centers[:, :2], axis=0), axis=1) / dt
        tilt = np.asarray(foot["tilt"])[contact]
        roll = np.asarray(foot["roll"])[contact]
        pitch = np.asarray(foot["pitch"])[contact]
        yaw_abs = np.abs(np.asarray(foot["yaw"])[contact])
        sole_contact[side] = {
            "contact_duty": float(contact.mean()),
            "sample_count": int(contact.sum()),
            "tilt_rad": contact_metric(tilt),
            "lateral_roll_rad": signed_metric(roll),
            "fore_aft_pitch_rad": signed_metric(pitch),
            "slip_speed_m_s": contact_metric(slip_speed[contact_pairs]),
            "yaw_abs_rad": contact_metric(yaw_abs),
        }
        events = touchdown_indices(times, contact)
        sole_contact[side]["touchdown_count"] = len(events)
        touchdown_events.extend(
            {
                "side": side,
                "index": int(index),
                "time_s": float(times[index]),
                "foot_x_w_m": float(centers[index, 0]),
            }
            for index in events
        )

    contact_load = {}
    dt_sample = float(np.median(np.diff(times)))
    sole_half_width_m = float(sole_half_size[1])
    for side in ("left", "right"):
        loads = [row["sole_contact_load"][side] for row in rows]
        all_forces = np.asarray([load["normal_force_n"] for load in loads])
        loaded = all_forces >= 5.0
        forces = all_forces[loaded]
        cop_y = np.asarray([load["cop_local_m"][1] for load in loads])[loaded]
        if forces.size == 0:
            raise ValueError(f"trace contains no loaded {side} sole samples")
        edge = np.abs(cop_y) >= 0.75 * sole_half_width_m
        contact_load[side] = {
            "minimum_normal_force_n": 5.0,
            "loaded_sample_count": int(loaded.sum()),
            "loaded_duty": float(loaded.mean()),
            "normal_force_mean_n": float(forces.mean()),
            "cop_lateral_m": {
                "signed_mean": float(cop_y.mean()),
                "mean_abs": float(np.abs(cop_y).mean()),
                "force_weighted_signed_mean": float(np.average(cop_y, weights=forces)),
                "force_weighted_mean_abs": float(np.average(np.abs(cop_y), weights=forces)),
                "p90_abs": float(np.percentile(np.abs(cop_y), 90)),
                "max_abs": float(np.abs(cop_y).max()),
            },
            "edge_threshold_m": 0.75 * sole_half_width_m,
            "edge_sample_fraction": float(edge.mean()),
            "edge_force_fraction": float(forces[edge].sum() / forces.sum()),
            "longest_edge_run_s": longest_true_run(edge, dt_sample),
        }

    touchdown_events.sort(key=lambda event: float(event["time_s"]))
    if len(touchdown_events) < 4:
        raise ValueError("trace contains too few debounced touchdown events")
    touchdown_times = np.asarray(
        [float(event["time_s"]) for event in touchdown_events], dtype=np.float64
    )
    touchdown_intervals = np.diff(touchdown_times)
    touchdown_step_lengths = np.diff(
        np.asarray([float(event["foot_x_w_m"]) for event in touchdown_events], dtype=np.float64)
    )
    alternating = np.asarray(
        [
            touchdown_events[index - 1]["side"] != touchdown_events[index]["side"]
            for index in range(1, len(touchdown_events))
        ],
        dtype=bool,
    )
    touchdown_gait = {
        "event_count": len(touchdown_events),
        "left_event_count": int(sole_contact["left"]["touchdown_count"]),
        "right_event_count": int(sole_contact["right"]["touchdown_count"]),
        "alternating_fraction": float(alternating.mean()),
        "cadence_steps_min": float(60.0 / touchdown_intervals.mean()),
        "interval_s": {
            "median": float(np.median(touchdown_intervals)),
            "p10": float(np.percentile(touchdown_intervals, 10)),
            "p90": float(np.percentile(touchdown_intervals, 90)),
        },
        "forward_step_length_m": {
            "mean": float(touchdown_step_lengths.mean()),
            "median": float(np.median(touchdown_step_lengths)),
            "p10": float(np.percentile(touchdown_step_lengths, 10)),
            "p90": float(np.percentile(touchdown_step_lengths, 90)),
        },
    }

    def joint_range(name: str) -> float:
        values = joint_pos[:, name_to_index[name]]
        return float(np.percentile(values, 95) - np.percentile(values, 5))

    result = {
        "schema": "sprite0825_mujoco_gait_trace_summary_v3",
        "trace": str(Path(args.trace)),
        "steady_after_s": args.steady_after,
        "steady_duration_s": duration,
        "forward_speed_m_s": forward_speed,
        "gait_cycle_hz": cycle_hz,
        "step_cadence_steps_min": step_hz * 60.0,
        "estimated_step_length_m": forward_speed / step_hz,
        "cadence_spectral_peak_fraction": spectral_fraction,
        "touchdown_gait": touchdown_gait,
        "pelvis_roll_deg": {
            "mean": math.degrees(float(roll.mean())),
            "rms": math.degrees(float(np.sqrt(np.mean(roll**2)))),
            "p95_abs": math.degrees(percentile_abs(roll, 95)),
            "peak_abs": math.degrees(float(np.max(np.abs(roll)))),
        },
        "pelvis_pitch_deg": {
            "mean": math.degrees(float(pitch.mean())),
            "rms": math.degrees(float(np.sqrt(np.mean(pitch**2)))),
            "p95_abs": math.degrees(percentile_abs(pitch, 95)),
        },
        "torso_roll_deg": {
            "mean": math.degrees(float(np.mean(torso_roll))),
            "rms": math.degrees(float(np.sqrt(np.mean(np.square(torso_roll))))),
            "p95_abs": math.degrees(percentile_abs(np.asarray(torso_roll), 95)),
            "peak_abs": math.degrees(float(np.max(np.abs(torso_roll)))),
        },
        "torso_pitch_deg": {
            "mean": math.degrees(float(np.mean(torso_pitch))),
            "rms": math.degrees(float(np.sqrt(np.mean(np.square(torso_pitch))))),
            "p95_abs": math.degrees(percentile_abs(np.asarray(torso_pitch), 95)),
        },
        "ankle_lateral_separation_m": {
            "mean": float(np.mean(ankle_lateral_separation)),
            "p10": float(np.percentile(ankle_lateral_separation, 10)),
            "p90": float(np.percentile(ankle_lateral_separation, 90)),
        },
        "ankle_forward_separation_m": {
            "mean": float(np.mean(ankle_forward_separation)),
            "p90": float(np.percentile(ankle_forward_separation, 90)),
            "peak": float(np.max(ankle_forward_separation)),
        },
        "sole_contact_reconstruction": {
            "ground_height_m": 0.0,
            "contact_height_threshold_m": contact_height_threshold_m,
            "sole_center_local_m": sole_center_local.tolist(),
            "sole_half_size_m": sole_half_size.tolist(),
            "left": sole_contact["left"],
            "right": sole_contact["right"],
            "contact_duty_gap": abs(
                sole_contact["left"]["contact_duty"]
                - sole_contact["right"]["contact_duty"]
            ),
            "tilt_mean_gap_rad": abs(
                sole_contact["left"]["tilt_rad"]["mean"]
                - sole_contact["right"]["tilt_rad"]["mean"]
            ),
            "slip_mean_gap_m_s": abs(
                sole_contact["left"]["slip_speed_m_s"]["mean"]
                - sole_contact["right"]["slip_speed_m_s"]["mean"]
            ),
        },
        "sole_contact_load": {
            "definition": "MuJoCo normal-force-weighted contact center in each sole frame",
            "sole_half_width_m": sole_half_width_m,
            "left": contact_load["left"],
            "right": contact_load["right"],
        },
        "joint_p05_p95_range_rad": {
            name: joint_range(name)
            for name in (
                "waist_roll_joint",
                "waist_yaw_joint",
                "left_hip_roll_joint",
                "right_hip_roll_joint",
                "left_hip_pitch_joint",
                "right_hip_pitch_joint",
                "left_knee_joint",
                "right_knee_joint",
                "left_shoulder_pitch_joint",
                "right_shoulder_pitch_joint",
            )
        },
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
