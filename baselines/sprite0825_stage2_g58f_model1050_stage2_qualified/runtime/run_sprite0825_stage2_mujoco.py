#!/usr/bin/env python3
"""Run the exported Sprite0825 Stage2 command policy in articulated MuJoCo."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, deque
from pathlib import Path

import mujoco
import mujoco.viewer as mj_viewer
import numpy as np
import onnxruntime as ort


def body_angular_velocity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    source: str,
) -> np.ndarray:
    if source == "body_local":
        spatial_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            spatial_velocity,
            1,
        )
        return spatial_velocity[:3]
    if source == "freejoint_local":
        return data.qvel[3:6].copy()
    if source == "legacy_rotated_qvel":
        return rotate_world_to_body(data.qpos[3:7], data.qvel[3:6])
    raise ValueError(f"Unknown angular velocity source: {source}")


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def rotate_world_to_body(q_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = q_wxyz / np.linalg.norm(q_wxyz)
    value = np.array([0.0, *vector], dtype=np.float64)
    return quat_multiply(quat_multiply(quat_conjugate(q), value), q)[1:]


class History:
    def __init__(self, length: int, initial: np.ndarray):
        if length < 1:
            raise ValueError(f"History length must be positive, got {length}")
        self.values = deque((initial.copy() for _ in range(length)), maxlen=length)

    def append(self, value: np.ndarray) -> None:
        self.values.append(value.copy())

    def flatten(self) -> np.ndarray:
        return np.concatenate(tuple(self.values))


def joint_addresses(model: mujoco.MjModel, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    qpos = []
    dof = []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"MJCF is missing policy joint: {name}")
        qpos.append(int(model.jnt_qposadr[joint_id]))
        dof.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(qpos), np.asarray(dof)


def clip_differential_ankle(
    pitch_torque: float,
    roll_torque: float,
    pitch_velocity: float,
    roll_velocity: float,
    peak_torque: float,
    no_load_speed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip one differential ankle in physical motor coordinates."""
    motor_torque = np.array(
        [
            0.5 * (pitch_torque + roll_torque),
            0.5 * (pitch_torque - roll_torque),
        ],
        dtype=np.float64,
    )
    motor_velocity = np.array(
        [pitch_velocity + roll_velocity, pitch_velocity - roll_velocity],
        dtype=np.float64,
    )
    limited_velocity = np.clip(motor_velocity, -2.0 * no_load_speed, 2.0 * no_load_speed)
    maximum = np.minimum(
        peak_torque * (1.0 - limited_velocity / no_load_speed), peak_torque
    )
    minimum = np.maximum(
        peak_torque * (-1.0 - limited_velocity / no_load_speed), -peak_torque
    )
    motor_torque = np.clip(motor_torque, minimum, maximum)
    joint_torque = np.array(
        [motor_torque[0] + motor_torque[1], motor_torque[0] - motor_torque[1]],
        dtype=np.float64,
    )
    return joint_torque, motor_torque, motor_velocity


def clip_dc_motor_torque(
    torque: np.ndarray,
    velocity: np.ndarray,
    peak_torque: np.ndarray,
    no_load_speed: np.ndarray,
) -> np.ndarray:
    """Match Isaac Lab DCMotor's signed four-quadrant torque-speed clipping."""
    limited_velocity = np.clip(velocity, -2.0 * no_load_speed, 2.0 * no_load_speed)
    maximum = np.minimum(
        peak_torque * (1.0 - limited_velocity / no_load_speed), peak_torque
    )
    minimum = np.maximum(
        peak_torque * (-1.0 - limited_velocity / no_load_speed), -peak_torque
    )
    return np.clip(torque, minimum, maximum)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--stand-seconds", type=float, default=2.0)
    parser.add_argument("--command-x", type=float, default=0.30)
    parser.add_argument("--command-y", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--command-ramp", type=float, default=0.8)
    parser.add_argument(
        "--yaw-delay-seconds",
        type=float,
        default=0.0,
        help="For headless tests, walk straight for this long before applying command-yaw.",
    )
    parser.add_argument(
        "--auto-walk-seconds",
        type=float,
        default=0.0,
        help="Headless cycle protocol: seconds commanded to walk in each cycle.",
    )
    parser.add_argument(
        "--auto-stop-seconds",
        type=float,
        default=0.0,
        help="Headless cycle protocol: seconds commanded to stop in each cycle.",
    )
    parser.add_argument("--pd-scale", type=float, default=None)
    parser.add_argument(
        "--motor-torque-scale",
        type=float,
        default=1.0,
        help="Diagnostic multiplier for all motor torque envelopes.",
    )
    parser.add_argument(
        "--startup-blend-seconds",
        type=float,
        default=None,
        help="Override the deployment handoff duration stored in the contract.",
    )
    parser.add_argument("--physics-dt", type=float, default=None)
    parser.add_argument(
        "--initial-root-height",
        type=float,
        default=None,
        help="Diagnostic override for the floating-base initial world Z position.",
    )
    parser.add_argument("--passive-damping", type=float, default=0.0)
    parser.add_argument("--friction-loss", type=float, default=0.0)
    parser.add_argument(
        "--contact-friction",
        type=float,
        default=None,
        help="Diagnostic override for the sliding friction of every contact geom.",
    )
    parser.add_argument(
        "--robot-contact-margin",
        type=float,
        default=None,
        help="Diagnostic MuJoCo contact margin applied to robot geoms only.",
    )
    parser.add_argument(
        "--disable-robot-self-collision",
        action="store_true",
        default=None,
        help="Match Isaac's enabled_self_collisions=False while retaining ground contact.",
    )
    parser.add_argument(
        "--actuator-delay-substeps",
        type=int,
        default=0,
        help="Delay joint targets by this many MuJoCo physics substeps.",
    )
    parser.add_argument(
        "--policy-ang-vel-z-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="Diagnostic sign applied only to policy base_ang_vel.z observation.",
    )
    parser.add_argument(
        "--angular-velocity-source",
        choices=("body_local", "freejoint_local", "legacy_rotated_qvel"),
        default=None,
        help="Override the contract source used for the actor base_ang_vel observation.",
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--log-every", type=float, default=1.0)
    parser.add_argument("--trace-output", default="")
    parser.add_argument("--summary-json", default="")
    args = parser.parse_args()

    contract_path = Path(args.contract).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    def resolve_contract_file(value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = contract_path.parent / path
        return path.resolve()

    contact_contract = contract.get("mujoco_contact", {})
    if args.startup_blend_seconds is None:
        args.startup_blend_seconds = float(contract.get("deployment_handoff_seconds", 0.04))
    if args.startup_blend_seconds < 0.0:
        raise ValueError("--startup-blend-seconds must be non-negative")
    if args.yaw_delay_seconds < 0.0:
        raise ValueError("--yaw-delay-seconds must be non-negative")
    auto_cycle_enabled = args.auto_walk_seconds > 0.0 or args.auto_stop_seconds > 0.0
    if auto_cycle_enabled and (
        args.auto_walk_seconds <= 0.0 or args.auto_stop_seconds <= 0.0
    ):
        raise ValueError("Both auto cycle durations must be positive when enabled")
    if args.actuator_delay_substeps < 0:
        raise ValueError("--actuator-delay-substeps must be non-negative")
    if args.pd_scale is None:
        args.pd_scale = float(contract.get("deployment_pd_scale", 1.0))
    if args.initial_root_height is None:
        value = contract.get("deployment_initial_root_height_m")
        args.initial_root_height = None if value is None else float(value)
    if args.contact_friction is None:
        value = contact_contract.get("sliding_friction")
        args.contact_friction = None if value is None else float(value)
    if args.disable_robot_self_collision is None:
        enabled = contact_contract.get("robot_self_collision_enabled")
        args.disable_robot_self_collision = False if enabled is None else not bool(enabled)
    if args.angular_velocity_source is None:
        args.angular_velocity_source = contract.get(
            "mujoco_base_ang_vel_source", "freejoint_local"
        )
    if args.pd_scale <= 0.0:
        raise ValueError("--pd-scale must be positive")
    if args.motor_torque_scale <= 0.0:
        raise ValueError("--motor-torque-scale must be positive")
    if args.contact_friction is not None and args.contact_friction < 0.0:
        raise ValueError("--contact-friction must be non-negative")
    if args.robot_contact_margin is not None and args.robot_contact_margin < 0.0:
        raise ValueError("--robot-contact-margin must be non-negative")
    if args.angular_velocity_source not in {
        "body_local",
        "freejoint_local",
        "legacy_rotated_qvel",
    }:
        raise ValueError(
            f"Unsupported contract angular velocity source: {args.angular_velocity_source}"
        )
    observation_terms = contract["observation_terms"]
    observation_dim = int(contract["actor_observation_dim"])
    model = mujoco.MjModel.from_xml_path(str(Path(args.mjcf).resolve()))
    if args.physics_dt is not None:
        model.opt.timestep = args.physics_dt
    data = mujoco.MjData(model)
    if args.initial_root_height is not None:
        data.qpos[2] = args.initial_root_height
    if model.nu != 0:
        raise RuntimeError("Stage2 runner requires the external-PD MJCF with nu=0")
    if any(float(value) != 0.0 for value in model.dof_damping[6:]):
        raise RuntimeError("Passive MJCF damping must be zero; damping is supplied exactly once by external PD")
    model.dof_damping[6:] = args.passive_damping
    model.dof_frictionloss[6:] = args.friction_loss
    if args.contact_friction is not None:
        model.geom_friction[:, 0] = args.contact_friction
    if args.robot_contact_margin is not None:
        model.geom_margin[model.geom_bodyid > 0] = args.robot_contact_margin
    if args.disable_robot_self_collision:
        robot_geom_mask = model.geom_bodyid > 0
        model.geom_conaffinity[robot_geom_mask] = 0

    joint_names = contract["joint_names"]
    qpos_addr, dof_addr = joint_addresses(model, joint_names)
    default_q = np.asarray(contract["default_joint_pos"], dtype=np.float64)
    action_offset = np.asarray(contract["action_offset"], dtype=np.float64)
    action_scale = np.asarray(contract["action_scale"], dtype=np.float64)
    kp = np.asarray(contract["stiffness"], dtype=np.float64) * args.pd_scale
    kd = np.asarray(contract["damping"], dtype=np.float64) * math.sqrt(args.pd_scale)
    effort = (
        np.asarray(contract["effort_limit"], dtype=np.float64)
        * args.motor_torque_scale
    )
    velocity_limit = np.asarray(contract["velocity_limit"], dtype=np.float64)
    if np.any(effort <= 0.0) or np.any(velocity_limit <= 0.0):
        raise RuntimeError("DC motor effort and velocity limits must be positive")
    j4340_names = [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee")
    ]
    j4340_indices = np.asarray([joint_names.index(name) for name in j4340_names])
    j4340_rated_torque = 14.0 * args.motor_torque_scale
    j4340_rated_speed = 3.7699111843
    j4340_peak_torque = effort[j4340_indices]
    j4340_no_load_speed = velocity_limit[j4340_indices]
    ankle_cfg = contract.get("physical_ankle_differential", {})
    ankle_enabled = bool(ankle_cfg.get("enabled", False))
    ankle_pairs = []
    if ankle_enabled:
        for side, names in ankle_cfg["joint_pairs"].items():
            if len(names) != 2:
                raise RuntimeError(f"Invalid {side} ankle pair in contract: {names}")
            ankle_pairs.append((side, joint_names.index(names[0]), joint_names.index(names[1])))
    ankle_motor_names = [
        f"{side}_motor_{motor}" for side, _, _ in ankle_pairs for motor in ("a", "b")
    ]
    ankle_rated_torque = (
        float(ankle_cfg.get("rated_torque_nm", 3.5)) * args.motor_torque_scale
    )
    ankle_peak_torque = (
        float(ankle_cfg.get("peak_torque_nm", 12.5)) * args.motor_torque_scale
    )
    ankle_no_load_speed = float(ankle_cfg.get("no_load_speed_rad_s", 36.2))
    action_clip = contract.get("action_clip")
    policy_dt = float(contract["policy_dt"])
    physics_dt = float(model.opt.timestep)
    substeps = round(policy_dt / physics_dt)
    if not math.isclose(substeps * physics_dt, policy_dt, rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError(f"policy_dt {policy_dt} is not an integer multiple of MuJoCo dt {physics_dt}")

    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
    if pelvis_id < 0:
        raise RuntimeError("MJCF is missing pelvis_link")
    data.qpos[qpos_addr] = default_q
    mujoco.mj_forward(model, data)

    def current_frames(last_action: np.ndarray) -> dict[str, np.ndarray]:
        q = data.qpos[qpos_addr].copy()
        qd = data.qvel[dof_addr].copy()
        policy_ang_vel = body_angular_velocity(
            model, data, pelvis_id, args.angular_velocity_source
        )
        policy_ang_vel[2] *= args.policy_ang_vel_z_sign
        return {
            "joint_pos": q - default_q,
            "joint_vel": qd,
            "actions": last_action,
            "base_ang_vel": policy_ang_vel,
            "projected_gravity": rotate_world_to_body(data.qpos[3:7], np.array([0.0, 0.0, -1.0])),
        }

    initial_action = np.divide(
        data.qpos[qpos_addr] - action_offset,
        action_scale,
        out=np.zeros_like(action_scale),
        where=np.abs(action_scale) > 1.0e-12,
    )
    last_action = initial_action.copy()
    first = current_frames(last_action)
    supported_terms = {*first, "velocity_commands"}
    resolved_names = [term["name"] for term in observation_terms]
    unsupported = [name for name in resolved_names if name not in supported_terms]
    if unsupported:
        raise RuntimeError(f"Unsupported observation terms in contract: {unsupported}")
    if len(resolved_names) != len(set(resolved_names)):
        raise RuntimeError(f"Duplicate observation terms in contract: {resolved_names}")
    histories = {}
    for term in observation_terms:
        name = term["name"]
        width = int(term["end"]) - int(term["start"])
        history_length = max(int(term["history_length"]), 1)
        frame_dim = int(term["frame_dim"])
        if width != history_length * frame_dim:
            raise RuntimeError(
                f"Observation term {name} width {width} != "
                f"history {history_length} * frame_dim {frame_dim}"
            )
        if name == "velocity_commands":
            if frame_dim != 3 or history_length != 1:
                raise RuntimeError(
                    "velocity_commands must be one [vx, vy, yaw_rate] frame"
                )
            continue
        if first[name].size != frame_dim:
            raise RuntimeError(
                f"Observation term {name} frame_dim {frame_dim} != runtime {first[name].size}"
            )
        histories[name] = History(history_length, first[name])
    if observation_terms[-1]["end"] != observation_dim:
        raise RuntimeError(
            f"Observation term layout ends at {observation_terms[-1]['end']}, "
            f"contract says {observation_dim}"
        )
    policy_onnx = resolve_contract_file(contract["policy_onnx"])
    if not policy_onnx.is_file():
        raise FileNotFoundError(f"Policy ONNX not found: {policy_onnx}")
    session = ort.InferenceSession(str(policy_onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    expected_input = session.get_inputs()[0].shape[-1]
    if isinstance(expected_input, int) and expected_input != observation_dim:
        raise RuntimeError(f"ONNX expects {expected_input}, contract says {observation_dim}")

    target_command = np.zeros(3, dtype=np.float64)
    active_command = np.array([args.command_x, args.command_y, args.command_yaw], dtype=np.float64)
    command = np.zeros(3, dtype=np.float64)
    command_rate = np.array([args.command_ramp, args.command_ramp, 2.0 * args.command_ramp])
    torque_peak = np.zeros(len(joint_names), dtype=np.float64)
    torque_sq_sum = np.zeros(len(joint_names), dtype=np.float64)
    torque_count = 0
    j4340_torque_sq_sum = np.zeros(len(j4340_names), dtype=np.float64)
    j4340_torque_peak = np.zeros(len(j4340_names), dtype=np.float64)
    j4340_speed_sq_sum = np.zeros(len(j4340_names), dtype=np.float64)
    j4340_speed_peak = np.zeros(len(j4340_names), dtype=np.float64)
    j4340_envelope_peak = np.zeros(len(j4340_names), dtype=np.float64)
    j4340_over_rated_count = np.zeros(len(j4340_names), dtype=np.int64)
    ankle_motor_torque_sq_sum = np.zeros(len(ankle_motor_names), dtype=np.float64)
    ankle_motor_torque_peak = np.zeros(len(ankle_motor_names), dtype=np.float64)
    ankle_motor_speed_sq_sum = np.zeros(len(ankle_motor_names), dtype=np.float64)
    ankle_motor_speed_peak = np.zeros(len(ankle_motor_names), dtype=np.float64)
    ankle_motor_envelope_peak = np.zeros(len(ankle_motor_names), dtype=np.float64)
    ankle_motor_over_rated_count = np.zeros(len(ankle_motor_names), dtype=np.int64)
    metric_count = 0
    metric_command_sum = np.zeros(3, dtype=np.float64)
    metric_actual_sum = np.zeros(3, dtype=np.float64)
    metric_abs_error_sum = np.zeros(3, dtype=np.float64)
    metric_sq_error_sum = np.zeros(3, dtype=np.float64)
    robot_self_contact_samples = 0
    robot_self_contact_pairs = Counter()
    failed = False
    trace = []
    startup_blend_steps = max(0, int(round(args.startup_blend_seconds / policy_dt)))
    delayed_targets = deque(
        (data.qpos[qpos_addr].copy() for _ in range(args.actuator_delay_substeps + 1)),
        maxlen=args.actuator_delay_substeps + 1,
    )

    def observation(command_value: np.ndarray) -> np.ndarray:
        values = []
        cursor = 0
        for term in observation_terms:
            name = term["name"]
            value = (
                command_value.copy()
                if name == "velocity_commands"
                else histories[name].flatten()
            )
            scale = term.get("scale")
            if scale is not None:
                value = value * np.asarray(scale, dtype=np.float64)
            clip = term.get("clip")
            if clip is not None:
                value = np.clip(value, float(clip[0]), float(clip[1]))
            start = int(term["start"])
            end = int(term["end"])
            if start != cursor or value.size != end - start:
                raise RuntimeError(
                    f"Observation term {name} layout mismatch: cursor={cursor}, "
                    f"range=({start}, {end}), value={value.size}"
                )
            values.append(value)
            cursor = end
        result = np.concatenate(values).astype(np.float32)
        if result.shape != (observation_dim,):
            raise RuntimeError(
                f"Stage2 observation shape is {result.shape}, expected ({observation_dim},)"
            )
        return result

    def key_callback(keycode: int) -> None:
        nonlocal target_command
        key = chr(keycode).lower() if 0 <= keycode < 256 else ""
        if key == "z":
            target_command = active_command.copy()
        elif key == "x":
            target_command = np.zeros(3, dtype=np.float64)
        elif key == "q":
            target_command[2] = min(target_command[2] + 0.15, 0.6)
        elif key == "e":
            target_command[2] = max(target_command[2] - 0.15, -0.6)

    def run(viewer=None) -> None:
        nonlocal command, failed, last_action, target_command, torque_count, torque_peak
        nonlocal metric_count, metric_command_sum, metric_actual_sum
        nonlocal metric_abs_error_sum, metric_sq_error_sum
        nonlocal robot_self_contact_samples
        steps = round(args.seconds / policy_dt)
        log_interval = max(1, round(args.log_every / policy_dt))
        for step in range(steps):
            if viewer is not None and not viewer.is_running():
                break
            wall_start = time.time()
            elapsed = step * policy_dt
            if elapsed >= args.stand_seconds and not args.viewer:
                if auto_cycle_enabled:
                    cycle_time = args.auto_walk_seconds + args.auto_stop_seconds
                    phase = (elapsed - args.stand_seconds) % cycle_time
                    target_command = (
                        active_command.copy()
                        if phase < args.auto_walk_seconds
                        else np.zeros(3, dtype=np.float64)
                    )
                else:
                    target_command = active_command.copy()
                    if elapsed < args.stand_seconds + args.yaw_delay_seconds:
                        target_command[2] = 0.0
            command += np.clip(target_command - command, -command_rate * policy_dt, command_rate * policy_dt)
            frames = current_frames(last_action)
            obs = observation(command)
            action = session.run([output_name], {input_name: obs[None]})[0][0].astype(np.float64)
            if action_clip is not None:
                action = np.clip(action, -float(action_clip), float(action_clip))
            if startup_blend_steps > 0 and step < startup_blend_steps:
                blend = (step + 1) / startup_blend_steps
                blend = blend * blend * (3.0 - 2.0 * blend)
                action = (1.0 - blend) * initial_action + blend * action
            if args.trace_output:
                pelvis_spatial_vel_local = np.zeros(6, dtype=np.float64)
                pelvis_spatial_vel_world = np.zeros(6, dtype=np.float64)
                mujoco.mj_objectVelocity(
                    model,
                    data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    pelvis_id,
                    pelvis_spatial_vel_local,
                    1,
                )
                mujoco.mj_objectVelocity(
                    model,
                    data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    pelvis_id,
                    pelvis_spatial_vel_world,
                    0,
                )
                trace.append(
                    {
                        "step": step,
                        "time_s": step * policy_dt,
                        "command": command.tolist(),
                        "root_pos_w": data.qpos[:3].tolist(),
                        "root_quat_w": data.qpos[3:7].tolist(),
                        "root_qvel_raw": data.qvel[:6].tolist(),
                        "root_ang_vel_b": frames["base_ang_vel"].tolist(),
                        "pelvis_spatial_vel_local": pelvis_spatial_vel_local.tolist(),
                        "pelvis_spatial_vel_world": pelvis_spatial_vel_world.tolist(),
                        "pelvis_xmat": data.xmat[pelvis_id].reshape(3, 3).tolist(),
                        "projected_gravity_b": frames["projected_gravity"].tolist(),
                        "joint_pos": data.qpos[qpos_addr].tolist(),
                        "joint_vel": data.qvel[dof_addr].tolist(),
                        "action": action.tolist(),
                        "observation": obs.tolist(),
                    }
                )
            target = action_offset + action_scale * action
            for _ in range(substeps):
                delayed_targets.append(target.copy())
                applied_target = delayed_targets[0]
                q = data.qpos[qpos_addr]
                qd = data.qvel[dof_addr]
                raw_tau = kp * (applied_target - q) - kd * qd
                tau = clip_dc_motor_torque(raw_tau, qd, effort, velocity_limit)
                ankle_motor_torque = []
                ankle_motor_speed = []
                for _, pitch_index, roll_index in ankle_pairs:
                    joint_tau, motor_tau, motor_vel = clip_differential_ankle(
                        tau[pitch_index],
                        tau[roll_index],
                        qd[pitch_index],
                        qd[roll_index],
                        ankle_peak_torque,
                        ankle_no_load_speed,
                    )
                    tau[pitch_index], tau[roll_index] = joint_tau
                    ankle_motor_torque.extend(motor_tau)
                    ankle_motor_speed.extend(motor_vel)
                data.qfrc_applied[:] = 0.0
                data.qfrc_applied[dof_addr] = tau
                torque_peak = np.maximum(torque_peak, np.abs(tau))
                torque_sq_sum[:] += tau * tau
                torque_count += 1
                j4340_tau = tau[j4340_indices]
                j4340_speed = qd[j4340_indices]
                j4340_torque_sq_sum[:] += j4340_tau * j4340_tau
                j4340_torque_peak[:] = np.maximum(j4340_torque_peak, np.abs(j4340_tau))
                j4340_speed_sq_sum[:] += j4340_speed * j4340_speed
                j4340_speed_peak[:] = np.maximum(j4340_speed_peak, np.abs(j4340_speed))
                j4340_envelope = np.abs(j4340_tau) / j4340_peak_torque + np.where(
                    j4340_tau * j4340_speed > 0.0,
                    np.abs(j4340_speed) / j4340_no_load_speed,
                    0.0,
                )
                j4340_envelope_peak[:] = np.maximum(j4340_envelope_peak, j4340_envelope)
                j4340_over_rated_count[:] += np.abs(j4340_tau) > j4340_rated_torque
                if ankle_motor_names:
                    motor_tau_array = np.asarray(ankle_motor_torque)
                    motor_speed_array = np.asarray(ankle_motor_speed)
                    ankle_motor_torque_sq_sum[:] += motor_tau_array * motor_tau_array
                    ankle_motor_torque_peak[:] = np.maximum(
                        ankle_motor_torque_peak, np.abs(motor_tau_array)
                    )
                    ankle_motor_speed_sq_sum[:] += motor_speed_array * motor_speed_array
                    ankle_motor_speed_peak[:] = np.maximum(
                        ankle_motor_speed_peak, np.abs(motor_speed_array)
                    )
                    ankle_motor_envelope_peak[:] = np.maximum(
                        ankle_motor_envelope_peak,
                        np.abs(motor_tau_array) / ankle_peak_torque
                        + np.where(
                            motor_tau_array * motor_speed_array > 0.0,
                            np.abs(motor_speed_array) / ankle_no_load_speed,
                            0.0,
                        ),
                    )
                    ankle_motor_over_rated_count[:] += (
                        np.abs(motor_tau_array) > ankle_rated_torque
                    )
                mujoco.mj_step(model, data)
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    body_a = int(model.geom_bodyid[contact.geom1])
                    body_b = int(model.geom_bodyid[contact.geom2])
                    if body_a <= 0 or body_b <= 0 or body_a == body_b:
                        continue
                    name_a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_a)
                    name_b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_b)
                    pair = tuple(sorted((name_a or str(body_a), name_b or str(body_b))))
                    robot_self_contact_pairs[pair] += 1
                    robot_self_contact_samples += 1
            last_action = action
            frames = current_frames(last_action)
            for name, value in frames.items():
                histories[name].append(value)
            # Evaluation only: measure deployable command tracking after the
            # initial stand and command ramp have settled. These values never
            # enter the actor observation or alter the controller.
            if elapsed >= args.stand_seconds + args.yaw_delay_seconds + 2.0:
                body_lin_vel = rotate_world_to_body(data.qpos[3:7], data.qvel[:3])
                physical_ang_vel = body_angular_velocity(
                    model, data, pelvis_id, "freejoint_local"
                )
                actual_velocity = np.array(
                    [body_lin_vel[0], body_lin_vel[1], physical_ang_vel[2]],
                    dtype=np.float64,
                )
                error = actual_velocity - command
                metric_count += 1
                metric_command_sum[:] += command
                metric_actual_sum[:] += actual_velocity
                metric_abs_error_sum[:] += np.abs(error)
                metric_sq_error_sum[:] += error * error
            root_z = float(data.qpos[2])
            tilt_z = float(-frames["projected_gravity"][2])
            if root_z < 0.25 or tilt_z < 0.45:
                failed = True
            if step % log_interval == 0 or step == steps - 1:
                qd_now = np.abs(data.qvel[dof_addr])
                qd_max_index = int(np.argmax(qd_now))
                print(
                    f"STEP {step:5d} t={data.time:6.2f} cmd=({command[0]:.2f},{command[1]:.2f},{command[2]:.2f}) "
                    f"root=({data.qpos[0]:.2f},{data.qpos[1]:.2f},{root_z:.2f}) upright={tilt_z:.3f} "
                    f"ang=({frames['base_ang_vel'][0]:.2f},{frames['base_ang_vel'][1]:.2f},{frames['base_ang_vel'][2]:.2f}) "
                    f"qerr={np.linalg.norm(data.qpos[qpos_addr] - default_q):.3f} "
                    f"qdmax={qd_now[qd_max_index]:.3f}:{joint_names[qd_max_index]} "
                    f"action_abs_mean={np.mean(np.abs(action)):.3f} action_abs_max={np.max(np.abs(action)):.3f}",
                    flush=True,
                )
            if failed:
                print(f"FALL t={data.time:.3f} root_z={root_z:.3f} upright={tilt_z:.3f}", flush=True)
                break
            if viewer is not None:
                viewer.sync()
            if args.real_time:
                time.sleep(max(0.0, policy_dt - (time.time() - wall_start)))

    print(
        f"START policy_dt={policy_dt} physics_dt={physics_dt} substeps={substeps} "
        f"obs={observation_dim} actions={len(joint_names)} handoff_steps={startup_blend_steps} "
        f"actuator_delay_substeps={args.actuator_delay_substeps} "
        f"initial_root_height={args.initial_root_height} "
        f"angular_velocity_source={args.angular_velocity_source} "
        f"policy_ang_vel_z_sign={args.policy_ang_vel_z_sign:+.0f} "
        f"passive_damping={args.passive_damping} "
        f"friction_loss={args.friction_loss} contact_friction={args.contact_friction} "
        f"disable_robot_self_collision={args.disable_robot_self_collision} "
        f"viewer={args.viewer}",
        flush=True,
    )
    if args.viewer:
        print("KEYS Z walk, X stop, Q/E yaw command", flush=True)
        with mj_viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = pelvis_id
            viewer.cam.distance = 2.2
            viewer.cam.azimuth = 90.0
            viewer.cam.elevation = -10.0
            run(viewer)
    else:
        run()

    peak = torque_peak
    rms = np.sqrt(torque_sq_sum / max(torque_count, 1))
    worst = np.argsort(peak / np.maximum(effort, 1.0e-9))[-8:][::-1]
    print(f"RESULT survived={not failed} sim_time={data.time:.3f} root_xy=({data.qpos[0]:.3f},{data.qpos[1]:.3f})")
    print(f"ROBOT_SELF_CONTACT_SAMPLES {robot_self_contact_samples}")
    for pair, count in robot_self_contact_pairs.most_common(10):
        print(f"ROBOT_SELF_CONTACT_PAIR {pair[0]} {pair[1]} count={count}")
    for index in worst:
        print(
            f"TORQUE {joint_names[index]} rms_nm={rms[index]:.3f} peak_abs_nm={peak[index]:.3f} "
            f"limit_nm={effort[index]:.3f} ratio={peak[index] / effort[index]:.3f}"
        )
    j4340_summary = {}
    for index, name in enumerate(j4340_names):
        motor_rms = math.sqrt(j4340_torque_sq_sum[index] / max(torque_count, 1))
        speed_rms = math.sqrt(j4340_speed_sq_sum[index] / max(torque_count, 1))
        over_rated_fraction = j4340_over_rated_count[index] / max(torque_count, 1)
        j4340_summary[name] = {
            "torque_rms_nm": motor_rms,
            "torque_peak_nm": float(j4340_torque_peak[index]),
            "rated_torque_nm": j4340_rated_torque,
            "peak_torque_nm": float(j4340_peak_torque[index]),
            "speed_rms_rad_s": speed_rms,
            "speed_peak_rad_s": float(j4340_speed_peak[index]),
            "rated_speed_rad_s": j4340_rated_speed,
            "no_load_speed_rad_s": float(j4340_no_load_speed[index]),
            "over_rated_fraction": float(over_rated_fraction),
            "torque_speed_envelope_peak_ratio": float(j4340_envelope_peak[index]),
        }
        print(
            f"J4340_MOTOR {name} torque_rms_nm={motor_rms:.3f} "
            f"torque_peak_nm={j4340_torque_peak[index]:.3f} "
            f"speed_rms_rad_s={speed_rms:.3f} speed_peak_rad_s={j4340_speed_peak[index]:.3f} "
            f"over_rated_fraction={over_rated_fraction:.6f} "
            f"envelope_peak_ratio={j4340_envelope_peak[index]:.3f}"
        )
    ankle_summary = {}
    for index, name in enumerate(ankle_motor_names):
        motor_rms = math.sqrt(ankle_motor_torque_sq_sum[index] / max(torque_count, 1))
        speed_rms = math.sqrt(ankle_motor_speed_sq_sum[index] / max(torque_count, 1))
        over_rated_fraction = ankle_motor_over_rated_count[index] / max(torque_count, 1)
        ankle_summary[name] = {
            "torque_rms_nm": motor_rms,
            "torque_peak_nm": float(ankle_motor_torque_peak[index]),
            "rated_torque_nm": ankle_rated_torque,
            "peak_torque_nm": ankle_peak_torque,
            "speed_rms_rad_s": speed_rms,
            "speed_peak_rad_s": float(ankle_motor_speed_peak[index]),
            "no_load_speed_rad_s": ankle_no_load_speed,
            "over_rated_fraction": float(over_rated_fraction),
            "torque_speed_envelope_peak_ratio": float(ankle_motor_envelope_peak[index]),
        }
        print(
            f"ANKLE_MOTOR {name} torque_rms_nm={motor_rms:.3f} "
            f"torque_peak_nm={ankle_motor_torque_peak[index]:.3f} "
            f"speed_rms_rad_s={speed_rms:.3f} speed_peak_rad_s={ankle_motor_speed_peak[index]:.3f} "
            f"over_rated_fraction={over_rated_fraction:.6f} "
            f"envelope_peak_ratio={ankle_motor_envelope_peak[index]:.3f}"
        )
    if args.trace_output:
        trace_path = Path(args.trace_output)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")
        print(f"WROTE_TRACE {trace_path} steps={len(trace)}")
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        divisor = max(metric_count, 1)
        summary = {
            "schema": "sprite0825_stage2_mujoco_eval_v3",
            "contract": str(contract_path),
            "survived": not failed,
            "sim_time_s": float(data.time),
            "final_root_pos": [float(value) for value in data.qpos[:3]],
            "automatic_cycle_protocol": {
                "enabled": auto_cycle_enabled,
                "walk_seconds": args.auto_walk_seconds,
                "stop_seconds": args.auto_stop_seconds,
                "completed_cycles": (
                    int(
                        max(data.time - args.stand_seconds, 0.0)
                        // (args.auto_walk_seconds + args.auto_stop_seconds)
                    )
                    if auto_cycle_enabled
                    else 0
                ),
            },
            "steady_state_tracking": {
                "sample_count": metric_count,
                "command_mean": (metric_command_sum / divisor).tolist(),
                "actual_body_velocity_mean": (metric_actual_sum / divisor).tolist(),
                "mae": (metric_abs_error_sum / divisor).tolist(),
                "rmse": np.sqrt(metric_sq_error_sum / divisor).tolist(),
                "layout": ["vx_m_s", "vy_m_s", "yaw_rate_rad_s"],
                "actor_uses_horizontal_base_velocity": False,
            },
            "robot_self_contacts": {
                "sample_count": robot_self_contact_samples,
                "top_pairs": [
                    {"body_a": pair[0], "body_b": pair[1], "count": count}
                    for pair, count in robot_self_contact_pairs.most_common(20)
                ],
                "disabled": args.disable_robot_self_collision,
            },
            "j4340p_physical_motors": j4340_summary,
            "ankle_physical_motors": ankle_summary,
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"SUMMARY_JSON {summary_path}")


if __name__ == "__main__":
    main()
