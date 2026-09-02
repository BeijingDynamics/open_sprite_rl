#!/usr/bin/env python3
"""Export a Sprite0825 Stage 2 actor and its runtime-derived contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--asset-urdf", required=True)
parser.add_argument("--label", default="sprite0825_stage2_g4")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument(
    "--expected-observation-dim",
    type=int,
    default=795,
    help="Fail if the resolved actor observation width differs (G8/G9 use 1488).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flat_list(value) -> list[float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value).reshape(-1).astype(float).tolist()


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg) -> None:
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.observations.critic.enable_corruption = False
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    env_cfg.seed = agent_cfg.seed

    checkpoint = Path(
        handle_deprecated_rsl_rl_checkpoint(args_cli.checkpoint, installed_version)
    ).resolve()
    asset_urdf = Path(args_cli.asset_urdf).resolve()
    output_dir = Path(args_cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not asset_urdf.is_file():
        raise FileNotFoundError(asset_urdf)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": True},
    )
    runner.export_policy_to_onnx(path=str(output_dir), filename="policy.onnx")
    runner.export_policy_to_jit(path=str(output_dir), filename="policy.pt")

    base = env.unwrapped
    robot = base.scene["robot"]
    action_term = base.action_manager.get_term("joint_pos")
    observation_manager = base.observation_manager
    policy_names = list(observation_manager.active_terms["policy"])
    expected_names = [
        "joint_pos",
        "joint_vel",
        "actions",
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
    ]
    if policy_names != expected_names:
        raise RuntimeError(f"Unexpected Stage2 policy terms: {policy_names}")
    forbidden_actor_terms = {"base_lin_vel", "root_lin_vel", "base_velocity"}
    unexpected_velocity_terms = forbidden_actor_terms.intersection(policy_names)
    if unexpected_velocity_terms:
        raise RuntimeError(
            f"Deployment actor unexpectedly contains horizontal velocity terms: "
            f"{sorted(unexpected_velocity_terms)}"
        )

    term_cfgs = observation_manager._group_obs_term_cfgs["policy"]
    term_dims = observation_manager.group_obs_term_dim["policy"]
    observation_terms = []
    cursor = 0
    for name, cfg, shape in zip(policy_names, term_cfgs, term_dims):
        width = int(np.prod(shape))
        frame_dim = width // max(int(cfg.history_length), 1)
        observation_terms.append(
            {
                "name": name,
                "start": cursor,
                "end": cursor + width,
                "shape": [int(value) for value in shape],
                "frame_dim": frame_dim,
                "history_length": int(cfg.history_length),
                "history_order": "oldest_to_newest" if cfg.history_length > 0 else "current",
                "flatten_history_dim": bool(cfg.flatten_history_dim),
                "scale": cfg.scale,
                "clip": cfg.clip,
            }
        )
        cursor += width
    if cursor != args_cli.expected_observation_dim:
        raise RuntimeError(
            f"Expected {args_cli.expected_observation_dim} actor observations, got {cursor}"
        )

    joint_names = list(robot.data.joint_names)
    action_joint_names = list(action_term._joint_names)
    if action_joint_names != joint_names:
        raise RuntimeError("Action and articulation joint orders differ")
    action_scale = flat_list(action_term._scale)
    action_offset = flat_list(action_term._offset)
    if len(action_scale) == 1:
        action_scale *= len(joint_names)
    if len(action_offset) == 1:
        action_offset *= len(joint_names)

    obs = env.get_observations()
    policy_obs = obs["policy"] if hasattr(obs, "keys") and "policy" in obs.keys() else obs
    policy = runner.get_inference_policy(device=base.device)
    with torch.inference_mode():
        torch_action = policy(obs).detach().cpu().numpy()
    onnx_path = output_dir / "policy.onnx"
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_action = session.run(None, {input_name: policy_obs.detach().cpu().numpy()})[0]
    parity_max_abs = float(np.max(np.abs(torch_action - onnx_action)))
    if parity_max_abs > 1.0e-4:
        raise RuntimeError(f"ONNX parity failed: max abs error {parity_max_abs}")

    asset_usd = Path(base.cfg.scene.robot.spawn.usd_path).resolve()
    contract = {
        "schema": "sprite0825_stage2_command_locomotion_contract_v1",
        "label": args_cli.label,
        "task": args_cli.task,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "policy_onnx": str(onnx_path),
        "policy_onnx_sha256": sha256(onnx_path),
        "policy_jit": str(output_dir / "policy.pt"),
        "asset_usd": str(asset_usd),
        "asset_usd_sha256": sha256(asset_usd),
        "asset_urdf": str(asset_urdf),
        "asset_urdf_sha256": sha256(asset_urdf),
        "policy_dt": float(base.step_dt),
        "physics_dt": float(base.physics_dt),
        "decimation": int(round(base.step_dt / base.physics_dt)),
        "actor_observation_dim": cursor,
        "observation_terms": observation_terms,
        "observation_history_reset": "replicate_first_sample_across_all_history_frames",
        "observation_has_horizontal_base_velocity": False,
        "observation_has_global_position": False,
        "observation_has_global_yaw": False,
        "command_layout": ["vx", "vy", "yaw_rate"],
        "command_units": ["m/s", "m/s", "rad/s"],
        "action_dim": len(joint_names),
        "action_clip": None if agent_cfg.clip_actions is None else float(agent_cfg.clip_actions),
        "action_formula": (
            "target_joint_pos = action_offset + action_scale * policy_action"
            if agent_cfg.clip_actions is None
            else "target_joint_pos = action_offset + action_scale * clipped_policy_action"
        ),
        "joint_names": joint_names,
        "default_joint_pos": flat_list(robot.data.default_joint_pos[0]),
        "action_scale": action_scale,
        "action_offset": action_offset,
        "stiffness": flat_list(robot.data.default_joint_stiffness[0]),
        "damping": flat_list(robot.data.default_joint_damping[0]),
        "effort_limit": flat_list(robot.data.joint_effort_limits[0]),
        "velocity_limit": flat_list(robot.data.joint_vel_limits[0]),
        "physical_ankle_differential": {
            "enabled": True,
            "motor_model": "DM-J4310P-2EC_at_approximately_38V",
            "joint_pairs": {
                side: [f"{side}_ankle_pitch_joint", f"{side}_ankle_roll_joint"]
                for side in ("left", "right")
            },
            "motor_torque_formula": [
                "0.5 * (tau_pitch + tau_roll)",
                "0.5 * (tau_pitch - tau_roll)",
            ],
            "motor_velocity_formula": [
                "qd_pitch + qd_roll",
                "qd_pitch - qd_roll",
            ],
            "rated_torque_nm": 3.5,
            "peak_torque_nm": 12.5,
            "rated_speed_rad_s": 12.56,
            "no_load_speed_rad_s": 36.2,
            "joint_to_motor_ratio": 1.0,
        },
        "deployment_pd_scale": 1.0,
        "deployment_handoff_seconds": 0.04,
        "deployment_handoff_mode": "smoothstep_from_pose_equivalent_action",
        "deployment_initial_velocity_mode": "zero",
        "deployment_initial_root_height_m": float(robot.data.default_root_state[0, 2]),
        "mujoco_contact": {
            "sliding_friction": 1.0,
            "torsional_friction": 0.005,
            "rolling_friction": 0.0001,
            "robot_self_collision_enabled": False,
            "source": "Isaac deterministic terrain material and articulation collision config",
        },
        "quaternion_order": "wxyz",
        "base_ang_vel_frame": "root_link_local",
        "mujoco_base_ang_vel_source": "freejoint_local",
        "mujoco_freejoint_angular_velocity_semantics": (
            "qvel[3:6] is already expressed in the free-joint local frame; do not rotate again"
        ),
        "projected_gravity_definition": "R_world_to_base @ [0, 0, -1]",
        "onnx_parity_max_abs": parity_max_abs,
    }
    contract_path = output_dir / "contract.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {contract_path}")
    print(f"OBS {cursor} ACTIONS {len(joint_names)} DT {base.step_dt}")
    print(f"ONNX_PARITY_MAX_ABS {parity_max_abs:.9g}")
    print(f"ASSET_USD {asset_usd} sha256={contract['asset_usd_sha256']}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
