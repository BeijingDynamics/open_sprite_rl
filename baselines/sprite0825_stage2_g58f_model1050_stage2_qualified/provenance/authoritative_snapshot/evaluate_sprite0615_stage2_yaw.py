from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Sprite0615 Stage 2 moving-turn yaw grid")
parser.add_argument("--task", required=True)
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=160)
parser.add_argument("--target_vx", type=float, default=0.30)
parser.add_argument("--duration", type=float, default=8.0)
parser.add_argument("--warmup", type=float, default=2.0)
parser.add_argument("--seed", type=int, default=42)
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


YAW_TARGETS = (-0.40, -0.20, 0.0, 0.20, 0.40)


def percentile(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x.float(), q).item())


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    if args_cli.num_envs < len(YAW_TARGETS):
        raise ValueError(f"--num_envs must be at least {len(YAW_TARGETS)}")
    if not 0.0 <= args_cli.warmup < args_cli.duration:
        raise ValueError("--warmup must be non-negative and less than --duration")

    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.observations.critic.enable_corruption = False
    env_cfg.episode_length_s = max(60.0, args_cli.duration + 5.0)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    checkpoint = handle_deprecated_rsl_rl_checkpoint(args_cli.checkpoint, installed_version)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    robot = base.scene["robot"]
    command_term = base.command_manager.get_term("base_velocity")
    command_term.is_standing_env[:] = False
    if hasattr(command_term, "is_heading_env"):
        command_term.is_heading_env[:] = False

    device = base.device
    dt = float(base.step_dt)
    targets = torch.tensor(YAW_TARGETS, device=device)
    groups = torch.arange(args_cli.num_envs, device=device) % len(YAW_TARGETS)
    target_yaw = targets[groups]
    steps = round(args_cli.duration / dt)
    warmup_steps = round(args_cli.warmup / dt)
    failed = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
    vx_samples: list[torch.Tensor] = []
    vy_samples: list[torch.Tensor] = []
    yaw_samples: list[torch.Tensor] = []

    with torch.inference_mode():
        for step in range(steps):
            command_term.is_standing_env[:] = False
            if hasattr(command_term, "is_heading_env"):
                command_term.is_heading_env[:] = False
            command_term.vel_command_b[:, 0] = args_cli.target_vx
            command_term.vel_command_b[:, 1] = 0.0
            command_term.vel_command_b[:, 2] = target_yaw
            obs = env.get_observations()
            actions = policy(obs)
            _, _, dones, _ = env.step(actions)
            if hasattr(policy, "reset"):
                policy.reset(dones)
            failed |= dones.bool()
            if step >= warmup_steps:
                vx_samples.append(robot.data.root_lin_vel_b[:, 0].detach().clone())
                vy_samples.append(robot.data.root_lin_vel_b[:, 1].detach().clone())
                yaw_samples.append(robot.data.root_ang_vel_b[:, 2].detach().clone())

    vx_all = torch.stack(vx_samples)
    vy_all = torch.stack(vy_samples)
    yaw_all = torch.stack(yaw_samples)
    grid = {}
    means = {}
    for target in YAW_TARGETS:
        mask = torch.isclose(target_yaw, torch.tensor(target, device=device))
        mean_yaw_per_env = yaw_all[:, mask].mean(dim=0)
        mean_vx_per_env = vx_all[:, mask].mean(dim=0)
        mean_abs_vy_per_env = vy_all[:, mask].abs().mean(dim=0)
        surviving = ~failed[mask]
        means[target] = float(mean_yaw_per_env.mean().item())
        grid[f"{target:+.2f}"] = {
            "mean_yaw_rate": means[target],
            "mae_yaw_rate": float(torch.abs(mean_yaw_per_env - target).mean().item()),
            "p10_yaw_rate": percentile(mean_yaw_per_env, 0.10),
            "p90_yaw_rate": percentile(mean_yaw_per_env, 0.90),
            "mean_vx": float(mean_vx_per_env.mean().item()),
            "mae_vx": float(torch.abs(mean_vx_per_env - args_cli.target_vx).mean().item()),
            "mean_abs_vy": float(mean_abs_vy_per_env.mean().item()),
            "survival_rate": float(surviving.float().mean().item()),
        }

    symmetry = {
        "0.20_abs_rate_difference": abs(abs(means[-0.20]) - abs(means[0.20])),
        "0.40_abs_rate_difference": abs(abs(means[-0.40]) - abs(means[0.40])),
    }
    result = {
        "task": args_cli.task,
        "checkpoint": str(Path(args_cli.checkpoint).resolve()),
        "protocol": "moving_yaw_grid",
        "num_envs": args_cli.num_envs,
        "step_dt": dt,
        "duration_s": args_cli.duration,
        "warmup_s": args_cli.warmup,
        "target_vx": args_cli.target_vx,
        "seed": args_cli.seed,
        "survival_rate": float((~failed).float().mean().item()),
        "grid": grid,
        "symmetry": symmetry,
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print("STAGE2_YAW_EVAL_JSON")
    print(payload)
    if args_cli.output:
        output = Path(args_cli.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
