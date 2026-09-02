from __future__ import annotations

import argparse
import importlib.metadata as metadata
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Review a deterministic G57 stand/walk command sequence.")
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--repeats", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument(
    "--no-real-time",
    action="store_true",
    help="Disable wall-clock throttling for headless smoke tests.",
)
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


SEQUENCE = (
    ("stand", 4.0, 0.00),
    ("slow", 6.0, 0.15),
    ("nominal", 8.0, 0.30),
    ("fast", 6.0, 0.45),
    ("stop", 5.0, 0.00),
    ("restart", 8.0, 0.30),
    ("final stand", 5.0, 0.00),
)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    if args_cli.repeats < 1:
        raise ValueError("--repeats must be at least one")

    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = max(
        120.0,
        args_cli.repeats * sum(duration for _, duration, _ in SEQUENCE) + 5.0,
    )
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.observations.critic.enable_corruption = False

    env_cfg.viewer.eye = (0.0, -3.0, 0.85)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.45)
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.asset_name = "robot"

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    checkpoint = handle_deprecated_rsl_rl_checkpoint(args_cli.checkpoint, installed_version)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    command = base.command_manager.get_term("base_velocity")
    command.is_standing_env[:] = False
    if hasattr(command, "is_heading_env"):
        command.is_heading_env[:] = False

    dt = float(base.step_dt)
    obs = env.get_observations()
    failed = False
    try:
        with torch.inference_mode():
            for repeat in range(args_cli.repeats):
                for name, duration, vx in SEQUENCE:
                    if not simulation_app.is_running():
                        return
                    print(
                        f"COMMAND repeat={repeat + 1}/{args_cli.repeats} "
                        f"segment={name} vx={vx:.2f} duration={duration:.1f}s",
                        flush=True,
                    )
                    for _ in range(round(duration / dt)):
                        if not simulation_app.is_running():
                            return
                        start = time.time()
                        command.is_standing_env[:] = False
                        if hasattr(command, "is_heading_env"):
                            command.is_heading_env[:] = False
                        command.vel_command_b[:, 0] = vx
                        command.vel_command_b[:, 1] = 0.0
                        command.vel_command_b[:, 2] = 0.0
                        obs = env.get_observations()
                        actions = policy(obs)
                        obs, _, dones, _ = env.step(actions)
                        policy.reset(dones)
                        if bool(dones.any().item()):
                            failed = True
                            print(f"RESET during segment={name}", flush=True)
                        sleep_time = dt - (time.time() - start)
                        if not args_cli.no_real_time and sleep_time > 0.0:
                            time.sleep(sleep_time)
    finally:
        print(f"SEQUENCE_COMPLETE reset_seen={failed}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
