from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Deterministic Sprite0615 Stage 2 command evaluation")
parser.add_argument("--task", default="Isaac-Sprite0615-Stage2-AMP-Forward-Play-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", default="")
parser.add_argument(
    "--teacher-policy",
    default="",
    help="Evaluate a frozen V38 TorchScript tracking actor instead of an RSL-RL checkpoint.",
)
parser.add_argument("--reference-vx", type=float, default=0.307)
parser.add_argument("--motion-file", default="")
parser.add_argument("--motion-start-at-zero", action="store_true")
parser.add_argument(
    "--fixed-phase-scale",
    type=float,
    default=-1.0,
    help="Use a fixed motion-frame increment per control step; negative derives it from vx.",
)
parser.add_argument(
    "--protocol",
    choices=("grid", "transitions", "stand", "gait", "yaw", "yaw_transitions"),
    default="grid",
)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--cycles", type=int, default=2)
parser.add_argument("--duration", type=float, default=30.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--command-ramp-rate",
    type=float,
    default=0.0,
    help="Optional vx slew rate in m/s^2; zero keeps the original step command.",
)
parser.add_argument(
    "--yaw-ramp-rate",
    type=float,
    default=0.0,
    help="Optional yaw-command slew rate in rad/s^2; zero keeps a step command.",
)
parser.add_argument(
    "--pre-yaw-seconds",
    type=float,
    default=8.0,
    help="Straight-walking duration before yaw_transitions applies the turn command.",
)
parser.add_argument(
    "--heading-hold",
    action="store_true",
    help="Use the PM01 outer heading controller for zero-yaw stand/straight commands.",
)
parser.add_argument("--output", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


def percentile(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x.float(), q).item())


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    if bool(args_cli.checkpoint) == bool(args_cli.teacher_policy):
        raise ValueError("Specify exactly one of --checkpoint or --teacher-policy")
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.observations.critic.enable_corruption = False
    if hasattr(env_cfg.observations, "teacher"):
        env_cfg.observations.teacher.enable_corruption = False
    if args_cli.motion_file:
        env_cfg.commands.motion.motion_file = args_cli.motion_file
    if args_cli.motion_start_at_zero:
        env_cfg.commands.motion.start_at_zero = True
    if args_cli.cycles < 1:
        raise ValueError("--cycles must be at least 1")
    if args_cli.protocol == "grid":
        protocol_duration = 6.0
    elif args_cli.protocol == "transitions":
        protocol_duration = 2.0 + args_cli.cycles * 7.0
    elif args_cli.protocol in ("stand", "gait", "yaw"):
        if args_cli.duration <= 0.0:
            raise ValueError("--duration must be positive")
        protocol_duration = args_cli.duration
    else:
        if args_cli.duration <= 0.0 or args_cli.pre_yaw_seconds <= 0.0:
            raise ValueError("yaw_transitions durations must be positive")
        protocol_duration = args_cli.pre_yaw_seconds + args_cli.duration
    env_cfg.episode_length_s = max(60.0, protocol_duration + 5.0)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    base = env.unwrapped
    if args_cli.teacher_policy:
        from isaaclab_tasks.manager_based.locomotion.velocity.config.sprite0615.flat_env_cfg import (
            v38_to_stage2_action_ratio,
        )

        teacher = torch.jit.load(args_cli.teacher_policy, map_location=base.device)
        teacher.eval()
        action_ratio = v38_to_stage2_action_ratio(base).detach()

        def policy(observations):
            return teacher(observations["teacher"]) * action_ratio

        policy_source = str(Path(args_cli.teacher_policy).resolve())
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        checkpoint = handle_deprecated_rsl_rl_checkpoint(args_cli.checkpoint, installed_version)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=base.device)
        policy_source = str(Path(args_cli.checkpoint).resolve())

    robot = base.scene["robot"]
    command_term = base.command_manager.get_term("base_velocity")
    motion_term = (
        base.command_manager.get_term("motion") if args_cli.teacher_policy else None
    )
    command_term.is_standing_env[:] = False
    if hasattr(command_term, "is_heading_env"):
        command_term.is_heading_env[:] = False
    dt = float(base.step_dt)
    obs = env.get_observations()
    failed = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=base.device)

    vx_samples: list[torch.Tensor] = []
    wz_samples: list[torch.Tensor] = []
    yaw_target_samples: list[torch.Tensor] = []
    target_samples: list[torch.Tensor] = []
    segment_samples: list[torch.Tensor] = []
    start_latency = torch.full((args_cli.cycles, args_cli.num_envs), float("nan"), device=base.device)
    stop_latency = torch.full((args_cli.cycles, args_cli.num_envs), float("nan"), device=base.device)
    gait_joint_names = (
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
    )
    gait_joint_ids, _ = robot.find_joints(list(gait_joint_names), preserve_order=True)
    gait_joint_samples: list[torch.Tensor] = []
    gait_root_pos_samples: list[torch.Tensor] = []
    gait_heading_samples: list[torch.Tensor] = []

    if args_cli.protocol == "grid":
        speeds = torch.tensor([0.0, 0.15, 0.30, 0.45], device=base.device)
        groups = torch.arange(args_cli.num_envs, device=base.device) % len(speeds)
        schedule = [(6.0, speeds[groups])]
    elif args_cli.protocol == "transitions":
        schedule = [(2.0, torch.zeros(args_cli.num_envs, device=base.device))]
        for _ in range(args_cli.cycles):
            schedule.extend(
                (
                    (4.0, torch.full((args_cli.num_envs,), 0.30, device=base.device)),
                    (3.0, torch.zeros(args_cli.num_envs, device=base.device)),
                )
            )
    elif args_cli.protocol == "stand":
        schedule = [(args_cli.duration, torch.zeros(args_cli.num_envs, device=base.device))]
    elif args_cli.protocol == "gait":
        schedule = [(args_cli.duration, torch.full((args_cli.num_envs,), 0.30, device=base.device))]
    elif args_cli.protocol == "yaw":
        yaw_levels = torch.tensor([-0.20, 0.0, 0.20], device=base.device)
        yaw_groups = torch.arange(args_cli.num_envs, device=base.device) % len(yaw_levels)
        yaw_targets = yaw_levels[yaw_groups]
        schedule = [(args_cli.duration, torch.full((args_cli.num_envs,), 0.30, device=base.device))]
    else:
        yaw_levels = torch.tensor([-0.20, 0.20], device=base.device)
        yaw_groups = torch.arange(args_cli.num_envs, device=base.device) % len(yaw_levels)
        yaw_targets = yaw_levels[yaw_groups]
        forward = torch.full((args_cli.num_envs,), 0.30, device=base.device)
        schedule = [(args_cli.pre_yaw_seconds, forward), (args_cli.duration, forward)]

    stand_start_xy = robot.data.root_pos_w[:, :2].detach().clone()
    stand_start_heading = robot.data.heading_w.detach().clone()
    integrated_yaw = torch.zeros(args_cli.num_envs, device=base.device)
    first_failure_segment = torch.full(
        (args_cli.num_envs,), -1, dtype=torch.long, device=base.device
    )
    first_failure_local_step = torch.full(
        (args_cli.num_envs,), -1, dtype=torch.long, device=base.device
    )

    global_step = 0
    segment_start = 0
    applied_vx = torch.zeros(args_cli.num_envs, device=base.device)
    applied_yaw = torch.zeros(args_cli.num_envs, device=base.device)
    with torch.inference_mode():
        for segment_id, (duration, target_vx) in enumerate(schedule):
            steps = round(duration / dt)
            segment_start = global_step
            for local_step in range(steps):
                if args_cli.command_ramp_rate > 0.0:
                    max_delta = args_cli.command_ramp_rate * dt
                    applied_vx += torch.clamp(target_vx - applied_vx, -max_delta, max_delta)
                else:
                    applied_vx.copy_(target_vx)
                if args_cli.protocol == "yaw":
                    target_yaw = yaw_targets
                elif args_cli.protocol == "yaw_transitions" and segment_id > 0:
                    target_yaw = yaw_targets
                else:
                    target_yaw = torch.zeros_like(applied_yaw)
                if args_cli.yaw_ramp_rate > 0.0:
                    max_yaw_delta = args_cli.yaw_ramp_rate * dt
                    applied_yaw += torch.clamp(
                        target_yaw - applied_yaw, -max_yaw_delta, max_yaw_delta
                    )
                else:
                    applied_yaw.copy_(target_yaw)
                if motion_term is not None:
                    if args_cli.fixed_phase_scale >= 0.0:
                        phase_scale = torch.full_like(applied_vx, args_cli.fixed_phase_scale)
                    else:
                        phase_scale = 0.5 * applied_vx / args_cli.reference_vx
                    motion_term.phase_scale.copy_(phase_scale)
                    motion_term.target_phase_scale.copy_(phase_scale)
                command_term.is_standing_env[:] = False
                command_term.vel_command_b[:, 0] = applied_vx
                command_term.vel_command_b[:, 1] = 0.0
                if args_cli.heading_hold:
                    command_term.is_heading_env[:] = True
                    command_term.heading_target[:] = stand_start_heading
                    command_term._update_command()
                else:
                    if hasattr(command_term, "is_heading_env"):
                        command_term.is_heading_env[:] = False
                    if args_cli.protocol in ("yaw", "yaw_transitions"):
                        command_term.vel_command_b[:, 2] = applied_yaw
                    else:
                        command_term.vel_command_b[:, 2] = 0.0
                obs = env.get_observations()
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if hasattr(policy, "reset"):
                    policy.reset(dones)
                newly_failed = dones.bool() & ~failed
                first_failure_segment[newly_failed] = segment_id
                first_failure_local_step[newly_failed] = local_step
                failed |= dones.bool()
                if args_cli.protocol == "stand":
                    integrated_yaw += robot.data.root_ang_vel_b[:, 2] * dt
                elif args_cli.protocol == "gait":
                    gait_joint_samples.append(robot.data.joint_pos[:, gait_joint_ids].detach().clone())
                    gait_root_pos_samples.append(robot.data.root_pos_w[:, :2].detach().clone())
                    gait_heading_samples.append(robot.data.heading_w.detach().clone())

                vx = robot.data.root_lin_vel_b[:, 0].detach().clone()
                vx_samples.append(vx)
                wz_samples.append(robot.data.root_ang_vel_b[:, 2].detach().clone())
                if args_cli.protocol in ("yaw", "yaw_transitions"):
                    yaw_target_samples.append(applied_yaw.detach().clone())
                else:
                    yaw_target_samples.append(torch.zeros_like(vx))
                target_samples.append(target_vx.detach().clone())
                segment_samples.append(
                    torch.full((args_cli.num_envs,), segment_id, device=base.device, dtype=torch.long)
                )

                elapsed = local_step * dt
                if args_cli.protocol == "transitions" and segment_id % 2 == 1:
                    cycle = (segment_id - 1) // 2
                    reached = torch.isnan(start_latency[cycle]) & (vx > 0.10) & ~failed
                    start_latency[cycle, reached] = elapsed
                if args_cli.protocol == "transitions" and segment_id > 0 and segment_id % 2 == 0:
                    cycle = (segment_id - 2) // 2
                    settled = torch.isnan(stop_latency[cycle]) & (torch.abs(vx) < 0.08) & ~failed
                    stop_latency[cycle, settled] = elapsed
                global_step += 1

    vx_all = torch.stack(vx_samples)
    wz_all = torch.stack(wz_samples)
    yaw_target_all = torch.stack(yaw_target_samples)
    target_all = torch.stack(target_samples)
    segment_all = torch.stack(segment_samples)
    result = {
        "task": args_cli.task,
        "checkpoint": policy_source,
        "policy_kind": "v38_tracking_teacher" if args_cli.teacher_policy else "rsl_rl_checkpoint",
        "protocol": args_cli.protocol,
        "num_envs": args_cli.num_envs,
        "step_dt": dt,
        "seed": args_cli.seed,
        "heading_hold": args_cli.heading_hold,
        "command_ramp_rate_mps2": args_cli.command_ramp_rate,
        "survival_rate": float((~failed).float().mean().item()),
    }

    if args_cli.protocol == "grid":
        grid = {}
        for speed in (0.0, 0.15, 0.30, 0.45):
            env_mask = torch.isclose(target_all[0], torch.tensor(speed, device=base.device))
            steady = vx_all[round(1.0 / dt) :, env_mask]
            per_env_mean = steady.mean(dim=0)
            grid[f"{speed:.2f}"] = {
                "mean_vx": float(per_env_mean.mean().item()),
                "mae_vx": float(torch.abs(per_env_mean - speed).mean().item()),
                "p10_vx": percentile(per_env_mean, 0.10),
                "p90_vx": percentile(per_env_mean, 0.90),
            }
        result["grid"] = grid
    elif args_cli.protocol == "transitions":
        start_ok = start_latency <= 1.0
        stop_ok = stop_latency <= 1.5
        restart = start_ok[1:] if args_cli.cycles > 1 else start_ok[:0]
        complete_per_env = (~failed) & torch.all(start_ok, dim=0) & torch.all(stop_ok, dim=0)
        result.update(
            {
                "cycles": args_cli.cycles,
                "first_start_within_1s_rate": float((start_ok[0] & ~failed).float().mean().item()),
                "restart_within_1s_rate": float(restart.float().mean().item()) if restart.numel() else None,
                "all_starts_within_1s_rate": float(start_ok.float().mean().item()),
                "all_stops_within_1p5s_rate": float(stop_ok.float().mean().item()),
                "complete_cycle_protocol_rate": float(complete_per_env.float().mean().item()),
                "start_latency_median_s": float(torch.nanmedian(start_latency).item()),
                "stop_latency_median_s": float(torch.nanmedian(stop_latency).item()),
            }
        )
        failure_histogram = {}
        for segment_id in range(len(schedule)):
            count = int((first_failure_segment == segment_id).sum().item())
            if count:
                failure_histogram[str(segment_id)] = {
                    "command_vx_mps": float(schedule[segment_id][1][0].item()),
                    "failure_count": count,
                    "mean_time_from_segment_start_s": float(
                        first_failure_local_step[first_failure_segment == segment_id].float().mean().item()
                        * dt
                    ),
                }
        result["first_failure_by_segment"] = failure_histogram
        segments = {}
        for segment_id in range(len(schedule)):
            data = vx_all[segment_all[:, 0] == segment_id]
            trim = min(round(0.5 / dt), max(data.shape[0] - 1, 0))
            segments[str(segment_id)] = float(data[trim:].mean().item())
        result["segment_mean_vx"] = segments
    elif args_cli.protocol == "stand":
        displacement = torch.linalg.vector_norm(robot.data.root_pos_w[:, :2] - stand_start_xy, dim=1)
        abs_yaw = integrated_yaw.abs()
        final_heading_error = torch.atan2(
            torch.sin(robot.data.heading_w - stand_start_heading),
            torch.cos(robot.data.heading_w - stand_start_heading),
        ).abs()
        stand_ok = (~failed) & (displacement <= 0.10) & (
            final_heading_error <= torch.deg2rad(torch.tensor(5.0, device=base.device))
        )
        result.update(
            {
                "duration_s": args_cli.duration,
                "mean_horizontal_displacement_m": float(displacement.mean().item()),
                "p90_horizontal_displacement_m": percentile(displacement, 0.90),
                "max_horizontal_displacement_m": float(displacement.max().item()),
                # Body-frame angular velocity z is not the world-yaw derivative when roll/pitch are non-zero.
                "diagnostic_mean_abs_integrated_body_z_ang_vel_deg": float(
                    torch.rad2deg(abs_yaw).mean().item()
                ),
                "diagnostic_p90_abs_integrated_body_z_ang_vel_deg": float(
                    torch.rad2deg(torch.quantile(abs_yaw, 0.90)).item()
                ),
                "diagnostic_max_abs_integrated_body_z_ang_vel_deg": float(
                    torch.rad2deg(abs_yaw.max()).item()
                ),
                "mean_abs_final_heading_error_deg": float(torch.rad2deg(final_heading_error).mean().item()),
                "p90_abs_final_heading_error_deg": float(
                    torch.rad2deg(torch.quantile(final_heading_error, 0.90)).item()
                ),
                "max_abs_final_heading_error_deg": float(torch.rad2deg(final_heading_error.max()).item()),
                "stand_gate_pass_rate": float(stand_ok.float().mean().item()),
            }
        )
    elif args_cli.protocol in ("yaw", "yaw_transitions"):
        if args_cli.protocol == "yaw_transitions":
            skip = min(round((args_cli.pre_yaw_seconds + 2.0) / dt), wz_all.shape[0] - 1)
        else:
            skip = min(round(2.0 / dt), wz_all.shape[0] - 1)
        steady_wz = wz_all[skip:]
        steady_vx = vx_all[skip:]
        targets = yaw_target_all[-1]
        yaw_grid = {}
        mean_by_level = {}
        levels = (-0.20, 0.20) if args_cli.protocol == "yaw_transitions" else (-0.20, 0.0, 0.20)
        for level in levels:
            mask = torch.isclose(targets, torch.tensor(level, device=base.device))
            per_env_wz = steady_wz[:, mask].mean(dim=0)
            per_env_vx = steady_vx[:, mask].mean(dim=0)
            mean_by_level[level] = per_env_wz.mean()
            yaw_grid[f"{level:+.2f}"] = {
                "mean_wz_rad_s": float(per_env_wz.mean().item()),
                "mae_wz_rad_s": float(torch.abs(per_env_wz - level).mean().item()),
                "p10_wz_rad_s": percentile(per_env_wz, 0.10),
                "p90_wz_rad_s": percentile(per_env_wz, 0.90),
                "mean_vx_m_s": float(per_env_vx.mean().item()),
            }
        left_magnitude = torch.abs(mean_by_level[0.20])
        right_magnitude = torch.abs(mean_by_level[-0.20])
        result["yaw_grid"] = yaw_grid
        result["yaw_symmetry"] = {
            "magnitude_gap_rad_s": float(torch.abs(left_magnitude - right_magnitude).item()),
            "magnitude_ratio_min_over_max": float(
                (torch.minimum(left_magnitude, right_magnitude) / torch.clamp(
                    torch.maximum(left_magnitude, right_magnitude), min=1.0e-6
                )).item()
            ),
            "zero_command_abs_wz_rad_s": (
                float(torch.abs(mean_by_level[0.0]).item()) if 0.0 in mean_by_level else None
            ),
        }
        if args_cli.protocol == "yaw_transitions":
            result["pre_yaw_seconds"] = args_cli.pre_yaw_seconds
            result["turn_duration_s"] = args_cli.duration
            result["yaw_ramp_rate_rad_s2"] = args_cli.yaw_ramp_rate
            result["failure_count_during_straight"] = int(
                ((first_failure_segment == 0) & failed).sum().item()
            )
            result["failure_count_during_turn"] = int(
                ((first_failure_segment == 1) & failed).sum().item()
            )
    else:
        joint_series = torch.stack(gait_joint_samples)
        root_xy = torch.stack(gait_root_pos_samples)
        headings = torch.stack(gait_heading_samples)
        skip = min(round(4.0 / dt), joint_series.shape[0] - 2)
        joint_series = joint_series[skip:]
        root_xy = root_xy[skip:]
        headings = headings[skip:]
        centered = joint_series - joint_series.mean(dim=0, keepdim=True)
        frequencies = torch.fft.rfftfreq(centered.shape[0], d=dt, device=base.device)
        spectrum = torch.abs(torch.fft.rfft(centered, dim=0)).square()
        frequency_mask = (frequencies >= 0.4) & (frequencies <= 4.0)
        masked_frequencies = frequencies[frequency_mask]
        dominant_indexes = spectrum[frequency_mask].argmax(dim=0)
        dominant_hz = masked_frequencies[dominant_indexes]
        joint_ranges = joint_series.amax(dim=0) - joint_series.amin(dim=0)
        hip_cycle_hz = dominant_hz[:, :2].mean(dim=1)
        path_length = torch.linalg.vector_norm(root_xy[1:] - root_xy[:-1], dim=-1).sum(dim=0)
        analyzed_duration = (root_xy.shape[0] - 1) * dt
        path_speed = path_length / analyzed_duration
        cadence_spm = 2.0 * hip_cycle_hz * 60.0
        estimated_step_length = path_speed / torch.clamp(2.0 * hip_cycle_hz, min=1.0e-6)
        heading_delta = torch.atan2(
            torch.sin(headings[-1] - headings[0]), torch.cos(headings[-1] - headings[0])
        )
        result["gait"] = {
            "command_vx_mps": 0.30,
            "analyzed_duration_s": analyzed_duration,
            "mean_path_speed_mps": float(path_speed.mean().item()),
            "mean_cadence_steps_per_min": float(cadence_spm.mean().item()),
            "p10_cadence_steps_per_min": percentile(cadence_spm, 0.10),
            "p90_cadence_steps_per_min": percentile(cadence_spm, 0.90),
            "mean_estimated_step_length_m": float(estimated_step_length.mean().item()),
            "mean_heading_drift_deg": float(torch.rad2deg(heading_delta).mean().item()),
            "mean_abs_heading_drift_deg": float(torch.rad2deg(heading_delta).abs().mean().item()),
            "joints": {
                name: {
                    "mean_range_rad": float(joint_ranges[:, index].mean().item()),
                    "mean_cycle_hz": float(dominant_hz[:, index].mean().item()),
                }
                for index, name in enumerate(gait_joint_names)
            },
        }

    payload = json.dumps(result, indent=2, sort_keys=True)
    print("STAGE2_EVAL_JSON")
    print(payload)
    if args_cli.output:
        output = Path(args_cli.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
