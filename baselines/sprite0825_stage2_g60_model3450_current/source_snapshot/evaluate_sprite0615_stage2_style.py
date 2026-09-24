#!/usr/bin/env python3
"""Compare Sprite0615 gait style with V38 using phase-invariant metrics."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--reference", required=True)
parser.add_argument("--reference-body-mode", choices=("archive", "none"), default="archive")
parser.add_argument("--output", required=True)
parser.add_argument("--mode", choices=("tracking", "velocity"), required=True)
parser.add_argument("--target-vx", type=float, default=0.30)
parser.add_argument("--target-yaw-rate", type=float, default=0.0)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--warmup-steps", type=int, default=100)
parser.add_argument("--sample-stride", type=int, default=5)
parser.add_argument("--cycle-frames", type=int, default=139)
parser.add_argument("--reference-inter-touchdown-s", type=float, default=0.63)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


INSTALLED_RSL_RL = metadata.version("rsl-rl-lib")
MAX_QUANTILE_SAMPLES = 1_000_000


def summary(values: torch.Tensor) -> dict[str, float]:
    values = values.float().flatten()
    quantile_values = values
    if values.numel() > MAX_QUANTILE_SAMPLES:
        stride = (values.numel() + MAX_QUANTILE_SAMPLES - 1) // MAX_QUANTILE_SAMPLES
        quantile_values = values[::stride]
    return {
        "mean": float(values.mean().item()),
        "rms": float(torch.sqrt(torch.mean(values.square())).item()),
        "p50": float(torch.quantile(quantile_values, 0.50).item()),
        "p90": float(torch.quantile(quantile_values, 0.90).item()),
        "p99": float(torch.quantile(quantile_values, 0.99).item()),
        "max": float(values.max().item()),
    }


def optional_summary(values: list[torch.Tensor]) -> dict[str, float] | None:
    nonempty = [value for value in values if value.numel()]
    return summary(torch.cat(nonempty)) if nonempty else None


def per_name_summary(values: torch.Tensor, names: list[str]) -> dict[str, dict[str, float]]:
    """Summarize the first dimension while preserving named trailing channels."""
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError(f"Expected [samples, {len(names)}], got {tuple(values.shape)}")
    return {name: summary(values[:, index]) for index, name in enumerate(names)}


def per_name_exceedance(values: torch.Tensor, names: list[str], threshold: float = 1.0) -> dict[str, float]:
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError(f"Expected [samples, {len(names)}], got {tuple(values.shape)}")
    return {
        name: float((values[:, index] > threshold).float().mean().item())
        for index, name in enumerate(names)
    }


def stable_touchdown_events(contact: torch.Tensor) -> torch.Tensor:
    """Return contact starts after three-frame off/on debounce windows."""
    if contact.shape[0] < 7:
        return torch.zeros((0, contact.shape[1]), dtype=torch.bool, device=contact.device)
    stable_off = ~contact[:-5] & ~contact[1:-4] & ~contact[2:-3]
    stable_on = contact[3:-2] & contact[4:-1] & contact[5:]
    return stable_off & stable_on


def stable_touchdown_rate(contact: torch.Tensor, dt: float) -> dict[str, float]:
    """Count contact starts with three-frame off/on debounce windows."""
    events = stable_touchdown_events(contact)
    duration = (contact.shape[0] - 1) * dt
    return summary(events.float().sum(dim=0) / duration)


def cadence_metrics(
    contact: torch.Tensor, dt: float, reference_inter_touchdown_s: float
) -> dict[str, object]:
    """Measure whole-gait cadence and left/right alternation after debounce."""
    left = stable_touchdown_events(contact[..., 0])
    right = stable_touchdown_events(contact[..., 1])
    duration = (contact.shape[0] - 1) * dt
    total_rate = (left.float().sum(dim=0) + right.float().sum(dim=0)) / duration
    intervals: list[torch.Tensor] = []
    alternation: list[torch.Tensor] = []
    for env_id in range(contact.shape[1]):
        left_times = torch.nonzero(left[:, env_id], as_tuple=False).flatten()
        right_times = torch.nonzero(right[:, env_id], as_tuple=False).flatten()
        times = torch.cat((left_times, right_times))
        sides = torch.cat(
            (
                torch.zeros_like(left_times, dtype=torch.bool),
                torch.ones_like(right_times, dtype=torch.bool),
            )
        )
        if times.numel() < 2:
            continue
        order = torch.argsort(times)
        ordered_times = times[order]
        ordered_sides = sides[order]
        intervals.append((ordered_times[1:] - ordered_times[:-1]).float() * dt)
        alternation.append((ordered_sides[1:] != ordered_sides[:-1]).float())

    reference_total_rate = 1.0 / reference_inter_touchdown_s
    return {
        "total_touchdown_rate_hz": summary(total_rate),
        "inter_touchdown_interval_s": optional_summary(intervals),
        "alternation_fraction": optional_summary(alternation),
        "reference_inter_touchdown_interval_s": reference_inter_touchdown_s,
        "reference_total_touchdown_rate_hz": reference_total_rate,
        "cadence_ratio_to_reference": summary(total_rate / reference_total_rate),
    }


def root_local_positions(body_pos: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
    delta = body_pos - body_pos[:, :1]
    quat = root_quat[:, None, :].expand(-1, body_pos.shape[1], -1)
    return quat_apply_inverse(quat.reshape(-1, 4), delta.reshape(-1, 3)).reshape_as(delta)


def nearest_reference_phase(
    joint_pos: torch.Tensor, reference_joint_pos: torch.Tensor
) -> torch.Tensor:
    scale = torch.clamp(reference_joint_pos.std(dim=0), min=0.05)
    query = joint_pos / scale
    reference = reference_joint_pos / scale
    reference_norm = reference.square().sum(dim=1)
    indices = []
    for chunk in query.split(1024):
        distance = (
            chunk.square().sum(dim=1, keepdim=True)
            + reference_norm.unsqueeze(0)
            - 2.0 * chunk @ reference.T
        )
        indices.append(torch.argmin(distance, dim=1))
    return torch.cat(indices)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg) -> None:
    installed_version = INSTALLED_RSL_RL
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device or agent_cfg.device
    agent_cfg.device = env_cfg.sim.device
    env_cfg.episode_length_s = max(
        float(getattr(env_cfg, "episode_length_s", 20.0)),
        (args_cli.steps + 100) * float(env_cfg.sim.dt) * float(env_cfg.decimation),
    )
    for group_name in ("policy", "critic"):
        group = getattr(env_cfg.observations, group_name, None)
        if group is not None and hasattr(group, "enable_corruption"):
            group.enable_corruption = False
    if hasattr(env_cfg.scene, "contact_forces"):
        env_cfg.scene.contact_forces.debug_vis = False
    if args_cli.mode == "tracking":
        env_cfg.commands.motion.debug_vis = False
        if hasattr(env_cfg.commands.motion, "speed_scale_range"):
            env_cfg.commands.motion.speed_scale_range = (1.0, 1.0)
        for event_name in ("push_robot", "physics_material", "add_joint_default_pos", "base_com"):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    checkpoint = handle_deprecated_rsl_rl_checkpoint(args_cli.checkpoint, installed_version)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(
        checkpoint,
        load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": True},
    )
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    robot = base.scene["robot"]
    contact_sensor = base.scene["contact_forces"]
    action_term = base.action_manager.get_term("joint_pos")
    dt = float(base.step_dt)
    observation_terms = list(base.observation_manager.active_terms["policy"])
    if "base_lin_vel" in observation_terms:
        raise RuntimeError("Actor observation unexpectedly contains base_lin_vel")

    reference_npz = np.load(args_cli.reference)
    reference_joint_names = [str(name) for name in reference_npz["joint_names"]]
    reference_body_names = (
        [str(name) for name in reference_npz["body_names"]]
        if args_cli.reference_body_mode == "archive"
        else []
    )
    if args_cli.cycle_frames > len(reference_npz["joint_pos"]):
        raise ValueError("--cycle-frames exceeds reference length")
    joint_ids = [robot.data.joint_names.index(name) for name in reference_joint_names]
    body_ids = [robot.body_names.index(name) for name in reference_body_names]
    j4340_names = [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee")
    ]
    j4340_ids = [robot.data.joint_names.index(name) for name in j4340_names]
    ankle_joint_ids = {
        side: (
            robot.data.joint_names.index(f"{side}_ankle_pitch_joint"),
            robot.data.joint_names.index(f"{side}_ankle_roll_joint"),
        )
        for side in ("left", "right")
    }
    ankle_motor_names = [
        f"{side}_motor_{motor}"
        for side in ("left", "right")
        for motor in ("a", "b")
    ]
    actuator_types = {name: type(actuator).__name__ for name, actuator in robot.actuators.items()}
    legs_actuator = robot.actuators.get("legs_j4340")
    ankle_actuator = robot.actuators.get("feet_j4310_pair")
    ankle_actuator_pairs = (
        [
            (
                ankle_actuator._joint_names.index(f"{side}_ankle_pitch_joint"),
                ankle_actuator._joint_names.index(f"{side}_ankle_roll_joint"),
            )
            for side in ("left", "right")
        ]
        if ankle_actuator is not None
        else []
    )
    foot_body_ids = [
        robot.body_names.index("left_foot_link"),
        robot.body_names.index("right_foot_link"),
    ]
    sensor_foot_ids = [
        contact_sensor.body_names.index("left_foot_link"),
        contact_sensor.body_names.index("right_foot_link"),
    ]
    reference_joint_pos = torch.as_tensor(
        reference_npz["joint_pos"][: args_cli.cycle_frames], device=base.device
    )
    reference_body_local = None
    if args_cli.reference_body_mode == "archive":
        reference_body_pos = torch.as_tensor(
            reference_npz["body_pos_w"][: args_cli.cycle_frames], device=base.device
        )
        reference_body_quat = torch.as_tensor(
            reference_npz["body_quat_w"][: args_cli.cycle_frames], device=base.device
        )
        reference_body_local = root_local_positions(reference_body_pos, reference_body_quat[:, 0])

    command_term = None
    if args_cli.mode == "velocity":
        command_term = base.command_manager.get_term("base_velocity")
        command_term.is_standing_env[:] = False

    actions_all = []
    targets_all = []
    joint_vel_all = []
    root_vx_all = []
    contact_all = []
    contact_slip = [[], []]
    contact_tilt = [[], []]
    swing_height = [[], []]
    j4340_envelope_all = []
    ankle_motor_envelope_all = []
    internal_j4340_envelope_all = []
    internal_ankle_motor_envelope_all = []
    aligned_j4340_envelope_all = []
    aligned_ankle_motor_envelope_all = []
    sample_joint = []
    sample_body_local = []
    sample_contact = []
    sample_foot_z = []
    failed = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=base.device)
    obs = env.get_observations()

    with torch.inference_mode():
        for step in range(args_cli.steps):
            if command_term is not None:
                command_term.is_standing_env[:] = False
                command_term.vel_command_b[:, 0] = args_cli.target_vx
                command_term.vel_command_b[:, 1] = 0.0
                command_term.vel_command_b[:, 2] = args_cli.target_yaw_rate
            obs = env.get_observations()
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            failed |= dones.bool()
            if step < args_cli.warmup_steps:
                continue

            actions_all.append(actions.detach().clone())
            scale = action_term._scale
            if isinstance(scale, torch.Tensor):
                if scale.ndim == 1:
                    scale = scale.unsqueeze(0)
                targets_all.append((actions * scale).detach().clone())
            else:
                targets_all.append((actions * float(scale)).detach().clone())
            joint_vel_all.append(robot.data.joint_vel[:, joint_ids].detach().clone())
            root_vx_all.append(robot.data.root_lin_vel_b[:, 0].detach().clone())

            j4340_effort = robot.data.applied_torque[:, j4340_ids]
            j4340_velocity = robot.data.joint_vel[:, j4340_ids]
            j4340_torque = torch.abs(j4340_effort)
            j4340_speed = torch.abs(j4340_velocity)
            j4340_motoring = (j4340_effort * j4340_velocity) > 0.0
            j4340_directional_envelope = (
                j4340_torque / 40.0
                + torch.where(j4340_motoring, j4340_speed / 9.3, 0.0)
            )
            j4340_envelope_all.append(
                torch.stack(
                    (
                        j4340_torque / 14.0,
                        j4340_torque / 40.0,
                        j4340_speed / 3.7699111843,
                        j4340_speed / 9.3,
                        j4340_directional_envelope,
                    ),
                    dim=-1,
                ).detach().clone()
            )
            ankle_motor_values = []
            for side in ("left", "right"):
                pitch_id, roll_id = ankle_joint_ids[side]
                pitch_torque = robot.data.applied_torque[:, pitch_id]
                roll_torque = robot.data.applied_torque[:, roll_id]
                pitch_speed = robot.data.joint_vel[:, pitch_id]
                roll_speed = robot.data.joint_vel[:, roll_id]
                for motor_torque, motor_speed in (
                    (0.5 * (pitch_torque + roll_torque), pitch_speed + roll_speed),
                    (0.5 * (pitch_torque - roll_torque), pitch_speed - roll_speed),
                ):
                    torque = torch.abs(motor_torque)
                    speed = torch.abs(motor_speed)
                    motoring = (motor_torque * motor_speed) > 0.0
                    directional_envelope = (
                        torque / 12.5 + torch.where(motoring, speed / 36.2, 0.0)
                    )
                    ankle_motor_values.append(
                        torch.stack(
                            (
                                torque / 3.5,
                                torque / 12.5,
                                speed / 12.5663706144,
                                speed / 36.2,
                                directional_envelope,
                            ),
                            dim=-1,
                        )
                    )
            ankle_motor_envelope_all.append(
                torch.stack(ankle_motor_values, dim=1).detach().clone()
            )

            if legs_actuator is not None and hasattr(legs_actuator, "_joint_vel"):
                effort = legs_actuator.applied_effort
                velocity = legs_actuator._joint_vel
                torque = torch.abs(effort)
                speed = torch.abs(velocity)
                directional_envelope = torque / 40.0 + torch.where(
                    effort * velocity > 0.0, speed / 9.3, 0.0
                )
                internal_j4340_envelope_all.append(directional_envelope.detach().clone())
                aligned_j4340_envelope_all.append(
                    torch.stack(
                        (
                            torque / 14.0,
                            torque / 40.0,
                            speed / 3.7699111843,
                            speed / 9.3,
                            directional_envelope,
                        ),
                        dim=-1,
                    ).detach().clone()
                )
            if ankle_actuator is not None and hasattr(ankle_actuator, "_joint_vel"):
                effort = ankle_actuator.applied_effort
                velocity = ankle_actuator._joint_vel
                internal_motor_values = []
                aligned_motor_values = []
                for pitch_index, roll_index in ankle_actuator_pairs:
                    pitch_effort = effort[:, pitch_index]
                    roll_effort = effort[:, roll_index]
                    pitch_velocity = velocity[:, pitch_index]
                    roll_velocity = velocity[:, roll_index]
                    for motor_effort, motor_velocity in (
                        (0.5 * (pitch_effort + roll_effort), pitch_velocity + roll_velocity),
                        (0.5 * (pitch_effort - roll_effort), pitch_velocity - roll_velocity),
                    ):
                        torque = torch.abs(motor_effort)
                        speed = torch.abs(motor_velocity)
                        directional_envelope = torque / 12.5 + torch.where(
                            motor_effort * motor_velocity > 0.0, speed / 36.2, 0.0
                        )
                        internal_motor_values.append(directional_envelope)
                        aligned_motor_values.append(
                            torch.stack(
                                (
                                    torque / 3.5,
                                    torque / 12.5,
                                    speed / 12.5663706144,
                                    speed / 36.2,
                                    directional_envelope,
                                ),
                                dim=-1,
                            )
                        )
                internal_ankle_motor_envelope_all.append(
                    torch.stack(internal_motor_values, dim=1).detach().clone()
                )
                aligned_ankle_motor_envelope_all.append(
                    torch.stack(aligned_motor_values, dim=1).detach().clone()
                )

            force = contact_sensor.data.net_forces_w_history[:, :, sensor_foot_ids, :]
            force = force.norm(dim=-1).max(dim=1).values
            contact = force > 10.0
            contact_all.append(contact.detach().clone())
            foot_xy_speed = torch.linalg.vector_norm(
                robot.data.body_lin_vel_w[:, foot_body_ids, :2], dim=-1
            )
            gravity_w = torch.zeros((args_cli.num_envs, 2, 3), device=base.device)
            gravity_w[..., 2] = -1.0
            foot_gravity_b = quat_apply_inverse(
                robot.data.body_quat_w[:, foot_body_ids].reshape(-1, 4), gravity_w.reshape(-1, 3)
            ).reshape(args_cli.num_envs, 2, 3)
            foot_tilt = torch.acos(torch.clamp(-foot_gravity_b[..., 2], -1.0, 1.0))
            foot_z = robot.data.body_pos_w[:, foot_body_ids, 2]
            for side in range(2):
                if torch.any(contact[:, side]):
                    contact_slip[side].append(foot_xy_speed[:, side][contact[:, side]].detach().clone())
                    contact_tilt[side].append(foot_tilt[:, side][contact[:, side]].detach().clone())
                if torch.any(~contact[:, side]):
                    swing_height[side].append(foot_z[:, side][~contact[:, side]].detach().clone())

            if (step - args_cli.warmup_steps) % args_cli.sample_stride == 0:
                sample_joint.append(robot.data.joint_pos[:, joint_ids].detach().clone())
                sample_contact.append(contact.detach().clone())
                sample_foot_z.append(foot_z.detach().clone())
                if reference_body_local is not None:
                    body_pos = robot.data.body_pos_w[:, body_ids]
                    root_quat = robot.data.body_quat_w[:, body_ids[0]]
                    sample_body_local.append(root_local_positions(body_pos, root_quat).detach().clone())

    action = torch.stack(actions_all)
    target = torch.stack(targets_all)
    joint_vel = torch.stack(joint_vel_all)
    root_vx = torch.stack(root_vx_all)
    j4340_envelope = torch.stack(j4340_envelope_all).flatten(0, 1)
    ankle_motor_envelope = torch.stack(ankle_motor_envelope_all).flatten(0, 1)
    internal_j4340_envelope = (
        torch.stack(internal_j4340_envelope_all).flatten(0, 1)
        if internal_j4340_envelope_all
        else None
    )
    internal_ankle_motor_envelope = (
        torch.stack(internal_ankle_motor_envelope_all).flatten(0, 1)
        if internal_ankle_motor_envelope_all
        else None
    )
    aligned_j4340_envelope = (
        torch.stack(aligned_j4340_envelope_all).flatten(0, 1)
        if aligned_j4340_envelope_all
        else None
    )
    aligned_ankle_motor_envelope = (
        torch.stack(aligned_ankle_motor_envelope_all).flatten(0, 1)
        if aligned_ankle_motor_envelope_all
        else None
    )
    contact = torch.stack(contact_all)
    sampled_joint = torch.stack(sample_joint)
    sample_steps, sample_envs = sampled_joint.shape[:2]
    flat_joint = sampled_joint.reshape(-1, sampled_joint.shape[-1])
    phase = nearest_reference_phase(flat_joint, reference_joint_pos)
    matched_joint = reference_joint_pos[phase]
    joint_error = flat_joint - matched_joint
    body_error_norm = None
    body_error_by_body = None
    body_error_vector = None
    if reference_body_local is not None:
        sampled_body_local = torch.stack(sample_body_local)
        flat_body = sampled_body_local.reshape(-1, *sampled_body_local.shape[2:])
        matched_body = reference_body_local[phase]
        body_error_vector = flat_body - matched_body
        body_error_by_body = torch.linalg.vector_norm(body_error_vector, dim=-1)
        body_error_norm = body_error_by_body.mean(dim=-1)
    phase_by_env = phase.reshape(sample_steps, sample_envs)
    sampled_contact = torch.stack(sample_contact).reshape(-1, 2)
    sampled_foot_z = torch.stack(sample_foot_z).reshape(-1, 2)
    phase_delta = torch.remainder(phase_by_env[1:] - phase_by_env[:-1], args_cli.cycle_frames)
    backward = phase_delta > (args_cli.cycle_frames // 2)
    forward_phase_delta = phase_delta[~backward].float()

    left_contact = contact[..., 0].float()
    right_contact = contact[..., 1].float()
    cadence = cadence_metrics(contact, dt, args_cli.reference_inter_touchdown_s)
    action_joint_names = list(robot.data.joint_names)
    action_scale = action_term._scale
    if isinstance(action_scale, torch.Tensor):
        action_scale_values = action_scale[0] if action_scale.ndim > 1 else action_scale
    else:
        action_scale_values = torch.full(
            (len(action_joint_names),), float(action_scale), device=base.device
        )
    raw_action_joint_rms = torch.sqrt(torch.mean(action.square(), dim=(0, 1)))
    target_offset_joint_rms = torch.sqrt(torch.mean(target.square(), dim=(0, 1)))
    raw_action_second_delta = action[2:] - 2.0 * action[1:-1] + action[:-2]
    raw_action_second_delta_joint_rms = torch.sqrt(
        torch.mean(raw_action_second_delta.square(), dim=(0, 1))
    )
    physical_target_delta = target[1:] - target[:-1]
    physical_target_delta_joint_rms = torch.sqrt(
        torch.mean(physical_target_delta.square(), dim=(0, 1))
    )
    phase_bins = []
    for bin_index in range(8):
        start = bin_index * args_cli.cycle_frames // 8
        end = (bin_index + 1) * args_cli.cycle_frames // 8
        mask = (phase >= start) & (phase < end)
        entry = {
            "bin": bin_index,
            "start_frame": start,
            "end_frame_exclusive": end,
            "sample_count": int(mask.sum().item()),
        }
        if torch.any(mask):
            entry["joint_rmse_mean"] = float(
                torch.sqrt(torch.mean(joint_error[mask].square(), dim=-1)).mean().item()
            )
            if body_error_norm is not None:
                entry["body_mean_l2_m"] = float(body_error_norm[mask].mean().item())
            for side, name in enumerate(("left", "right")):
                swing_mask = mask & ~sampled_contact[:, side]
                entry[f"{name}_swing_count"] = int(swing_mask.sum().item())
                entry[f"{name}_swing_height_mean_m"] = (
                    float(sampled_foot_z[swing_mask, side].mean().item()) if torch.any(swing_mask) else None
                )
        phase_bins.append(entry)

    support_masks = {
        "double": sampled_contact[:, 0] & sampled_contact[:, 1],
        "left_only": sampled_contact[:, 0] & ~sampled_contact[:, 1],
        "right_only": ~sampled_contact[:, 0] & sampled_contact[:, 1],
        "flight": ~sampled_contact[:, 0] & ~sampled_contact[:, 1],
    }
    error_by_support_state = {}
    for name, mask in support_masks.items():
        entry = {"sample_count": int(mask.sum().item())}
        if torch.any(mask):
            entry["joint_rmse_mean"] = float(
                torch.sqrt(torch.mean(joint_error[mask].square(), dim=-1)).mean().item()
            )
            if body_error_norm is not None:
                entry["body_mean_l2_m"] = float(body_error_norm[mask].mean().item())
        error_by_support_state[name] = entry
    result = {
        "schema": "sprite0615_stage2_style_v2",
        "mode": args_cli.mode,
        "target_vx_m_s": args_cli.target_vx,
        "target_yaw_rate_rad_s": args_cli.target_yaw_rate,
        "task": args_cli.task,
        "checkpoint": str(Path(args_cli.checkpoint).resolve()),
        "reference": str(Path(args_cli.reference).resolve()),
        "reference_body_mode": args_cli.reference_body_mode,
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "steps": args_cli.steps,
        "warmup_steps": args_cli.warmup_steps,
        "sample_stride": args_cli.sample_stride,
        "cycle_frames": args_cli.cycle_frames,
        "step_dt": dt,
        "survival_rate": float((~failed).float().mean().item()),
        "observation_terms": observation_terms,
        "contains_base_lin_vel": "base_lin_vel" in observation_terms,
        "root_vx": summary(root_vx),
        "motor_envelope_contract": {
            "actuator_types": actuator_types,
            "j4340p": {
                "rated_torque_nm": 14.0,
                "peak_torque_nm": 40.0,
                "rated_speed_rad_s": 3.7699111843,
                "working_voltage_no_load_speed_rad_s": 9.3,
                "peak_torque_speed_metric": "directional: torque/peak + speed/no_load only while motoring",
            },
            "j4310p_differential_ankle": {
                "motor_torque_mapping": "0.5 * (joint_pitch_torque +/- joint_roll_torque)",
                "motor_speed_mapping": "joint_pitch_speed +/- joint_roll_speed",
                "rated_torque_nm": 3.5,
                "peak_torque_nm": 12.5,
                "rated_speed_rad_s": 12.5663706144,
                "working_voltage_no_load_speed_rad_s": 36.2,
                "peak_torque_speed_metric": "directional: torque/peak + speed/no_load only while motoring",
            },
        },
        "j4340p_motor_ratio_by_joint": {
            metric: per_name_summary(j4340_envelope[:, :, index], j4340_names)
            for index, metric in enumerate(
                ("rated_torque", "peak_torque", "rated_speed", "max_speed", "peak_torque_speed")
            )
        },
        "j4340p_motor_ratio_aggregate": {
            metric: summary(j4340_envelope[:, :, index])
            for index, metric in enumerate(
                ("rated_torque", "peak_torque", "rated_speed", "max_speed", "peak_torque_speed")
            )
        },
        "j4340p_peak_torque_speed_exceedance_fraction_by_joint": per_name_exceedance(
            j4340_envelope[:, :, 4], j4340_names
        ),
        "j4310p_ankle_motor_ratio_by_motor": {
            metric: per_name_summary(ankle_motor_envelope[:, :, index], ankle_motor_names)
            for index, metric in enumerate(
                ("rated_torque", "peak_torque", "rated_speed", "max_speed", "peak_torque_speed")
            )
        },
        "j4310p_ankle_motor_ratio_aggregate": {
            metric: summary(ankle_motor_envelope[:, :, index])
            for index, metric in enumerate(
                ("rated_torque", "peak_torque", "rated_speed", "max_speed", "peak_torque_speed")
            )
        },
        "j4310p_peak_torque_speed_exceedance_fraction_by_motor": per_name_exceedance(
            ankle_motor_envelope[:, :, 4], ankle_motor_names
        ),
        "aligned_motor_envelope_provenance": (
            "actuator.applied_effort paired with actuator._joint_vel from the same physics substep; "
            "legacy j4340p/j4310p fields pair applied torque with post-step articulation velocity"
        ),
        "aligned_j4340p_motor_ratio_by_joint": (
            {
                metric: per_name_summary(
                    aligned_j4340_envelope[:, :, index], list(legs_actuator._joint_names)
                )
                for index, metric in enumerate(
                    ("rated_torque", "peak_torque", "rated_speed", "max_speed", "peak_torque_speed")
                )
            }
            if aligned_j4340_envelope is not None
            else None
        ),
        "aligned_j4340p_peak_torque_speed_exceedance_fraction_by_joint": (
            per_name_exceedance(
                aligned_j4340_envelope[:, :, 4], list(legs_actuator._joint_names)
            )
            if aligned_j4340_envelope is not None
            else None
        ),
        "aligned_j4310p_motor_ratio_by_motor": (
            {
                metric: per_name_summary(aligned_ankle_motor_envelope[:, :, index], ankle_motor_names)
                for index, metric in enumerate(
                    ("rated_torque", "peak_torque", "rated_speed", "max_speed", "peak_torque_speed")
                )
            }
            if aligned_ankle_motor_envelope is not None
            else None
        ),
        "aligned_j4310p_peak_torque_speed_exceedance_fraction_by_motor": (
            per_name_exceedance(aligned_ankle_motor_envelope[:, :, 4], ankle_motor_names)
            if aligned_ankle_motor_envelope is not None
            else None
        ),
        "actuator_internal_j4340p_peak_torque_speed": (
            summary(internal_j4340_envelope) if internal_j4340_envelope is not None else None
        ),
        "actuator_internal_j4310p_peak_torque_speed": (
            summary(internal_ankle_motor_envelope)
            if internal_ankle_motor_envelope is not None
            else None
        ),
        "phase_invariant_joint_l2": summary(torch.linalg.vector_norm(joint_error, dim=-1)),
        "phase_invariant_joint_rmse": summary(torch.sqrt(torch.mean(joint_error.square(), dim=-1))),
        "phase_invariant_body_mean_l2_m": summary(body_error_norm) if body_error_norm is not None else None,
        "phase_invariant_body_l2_by_name_m": (
            per_name_summary(body_error_by_body, reference_body_names)
            if body_error_by_body is not None
            else None
        ),
        "phase_invariant_body_abs_xyz_mean_by_name_m": (
            {
                name: [float(value) for value in body_error_vector[:, index].abs().mean(dim=0).tolist()]
                for index, name in enumerate(reference_body_names)
            }
            if body_error_vector is not None
            else None
        ),
        "phase_invariant_joint_abs_by_name_rad": per_name_summary(joint_error.abs(), reference_joint_names),
        "phase_bin_diagnostics": phase_bins,
        "error_by_support_state": error_by_support_state,
        "phase_backward_jump_rate": float(backward.float().mean().item()),
        "forward_phase_delta_frames": summary(forward_phase_delta),
        "phase_speed_ratio_to_reference": float(
            forward_phase_delta.mean().item() / args_cli.sample_stride
        ),
        "phase_estimator_status": "diagnostic_only_pose_nearest_is_ambiguous_for_multisegment_reference",
        "action_joint_names": action_joint_names,
        "action_scale_by_joint_rad": {
            name: float(value.item()) for name, value in zip(action_joint_names, action_scale_values)
        },
        "raw_action_rms": summary(torch.sqrt(torch.mean(action.square(), dim=-1))),
        "raw_action_abs": summary(torch.abs(action)),
        "raw_action_rms_by_joint": {
            name: float(value.item()) for name, value in zip(action_joint_names, raw_action_joint_rms)
        },
        "physical_target_offset_rms_rad": summary(
            torch.sqrt(torch.mean(target.square(), dim=-1))
        ),
        "physical_target_offset_abs_rad": summary(torch.abs(target)),
        "physical_target_offset_rms_by_joint_rad": {
            name: float(value.item())
            for name, value in zip(action_joint_names, target_offset_joint_rms)
        },
        "raw_action_delta_rms": summary(
            torch.sqrt(torch.mean((action[1:] - action[:-1]).square(), dim=-1))
        ),
        "raw_action_second_delta_rms": summary(
            torch.sqrt(torch.mean(raw_action_second_delta.square(), dim=-1))
        ),
        "raw_action_second_delta_rms_by_joint": {
            name: float(value.item())
            for name, value in zip(action_joint_names, raw_action_second_delta_joint_rms)
        },
        "physical_target_delta_rms_rad": summary(
            torch.sqrt(torch.mean(physical_target_delta.square(), dim=-1))
        ),
        "physical_target_delta_rms_by_joint_rad": {
            name: float(value.item())
            for name, value in zip(action_joint_names, physical_target_delta_joint_rms)
        },
        "joint_velocity_rms_rad_s": summary(torch.sqrt(torch.mean(joint_vel.square(), dim=-1))),
        "left_contact_duty": float(left_contact.mean().item()),
        "right_contact_duty": float(right_contact.mean().item()),
        "contact_duty_gap": float(torch.abs(left_contact.mean() - right_contact.mean()).item()),
        "left_touchdown_rate_hz": stable_touchdown_rate(contact[..., 0], dt),
        "right_touchdown_rate_hz": stable_touchdown_rate(contact[..., 1], dt),
        "cadence": cadence,
        "left_contact_slip_m_s": optional_summary(contact_slip[0]),
        "right_contact_slip_m_s": optional_summary(contact_slip[1]),
        "left_contact_tilt_rad": optional_summary(contact_tilt[0]),
        "right_contact_tilt_rad": optional_summary(contact_tilt[1]),
        "left_swing_link_height_m": optional_summary(swing_height[0]),
        "right_swing_link_height_m": optional_summary(swing_height[1]),
    }
    for output_name, left_name, right_name in (
        ("contact_slip_mean_gap_m_s", "left_contact_slip_m_s", "right_contact_slip_m_s"),
        ("contact_tilt_mean_gap_rad", "left_contact_tilt_rad", "right_contact_tilt_rad"),
        ("swing_height_mean_gap_m", "left_swing_link_height_m", "right_swing_link_height_m"),
    ):
        left = result[left_name]
        right = result[right_name]
        result[output_name] = abs(left["mean"] - right["mean"]) if left and right else None

    payload = json.dumps(result, indent=2, sort_keys=True)
    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n", encoding="utf-8")
    print("STAGE2_STYLE_JSON")
    print(payload)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
