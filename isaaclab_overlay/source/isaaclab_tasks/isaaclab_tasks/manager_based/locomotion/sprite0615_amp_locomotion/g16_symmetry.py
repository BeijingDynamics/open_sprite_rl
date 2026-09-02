from __future__ import annotations

import torch
from tensordict import TensorDict


# Runtime response tests on Sprite0825 verify these signs.  A mirrored ankle-pitch
# uses the same coordinate sign; all other paired joints in this asset use the
# opposite sign.  Unpaired roll/yaw joints also negate under XZ-plane reflection.
_FULL_JOINT_SWAP = torch.tensor(
    [1, 0, 2, 4, 3, 5, 7, 6, 8, 10, 9, 12, 11, 13, 15, 14,
     17, 16, 18, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27, 30, 29],
    dtype=torch.long,
)
_FULL_JOINT_SIGN = torch.tensor(
    [-1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1,
     -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1,
     -1, 1, 1, -1, -1],
    dtype=torch.float32,
)

# Joint order used by PM01_HOMOLOGOUS_JOINT_NAMES in flat_env_cfg.py.
_AMP_JOINT_SWAP = torch.tensor(
    [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
     17, 16, 19, 18, 21, 20, 22],
    dtype=torch.long,
)
_AMP_JOINT_SIGN = torch.tensor(
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
     1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    dtype=torch.float32,
)


def _mirror_joints(values: torch.Tensor, swap: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    return values[..., swap.to(values.device)] * sign.to(device=values.device, dtype=values.dtype)


def mirror_full_joints(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] != 31:
        raise ValueError(f"expected 31 Sprite joints, got {values.shape}")
    return _mirror_joints(values, _FULL_JOINT_SWAP, _FULL_JOINT_SIGN)


def mirror_amp_history(values: torch.Tensor) -> torch.Tensor:
    """Mirror AMP history for either the PM01-23 or Sprite-31 contract."""
    supported_history_lengths = (5, 15, 65)
    if values.shape[-1] % 35 == 0 and values.shape[-1] // 35 in supported_history_lengths:
        frame_dim = 35
        joint_count = 31
    elif values.shape[-1] % 34 == 0 and values.shape[-1] // 34 in supported_history_lengths:
        frame_dim = 34
        joint_count = 31
    elif values.shape[-1] % 27 == 0 and values.shape[-1] // 27 in supported_history_lengths:
        frame_dim = 27
        joint_count = 23
    elif values.shape[-1] % 26 == 0 and values.shape[-1] // 26 in supported_history_lengths:
        frame_dim = 26
        joint_count = 23
    else:
        raise ValueError(f"unsupported AMP history shape {values.shape}")
    frames = values.reshape(*values.shape[:-1], -1, frame_dim).clone()
    if joint_count == 31:
        frames[..., :31] = mirror_full_joints(frames[..., :31])
    else:
        frames[..., :23] = _mirror_joints(frames[..., :23], _AMP_JOINT_SWAP, _AMP_JOINT_SIGN)
    frames[..., joint_count : joint_count + 3] *= frames.new_tensor([1.0, -1.0, 1.0])
    return frames.reshape_as(values)


def mirror_policy_history(values: torch.Tensor) -> torch.Tensor:
    """Mirror G16/G17 actor history without introducing undeployable state."""
    if values.shape[-1] != 1488:
        raise ValueError(f"expected 1488-D G16 policy observation, got {values.shape}")
    mirrored = values.clone()
    for start in (0, 465, 930):
        block = values[..., start : start + 465].reshape(*values.shape[:-1], 15, 31)
        mirrored[..., start : start + 465] = mirror_full_joints(block).reshape(
            *values.shape[:-1], 465
        )
    angular = values[..., 1395:1440].reshape(*values.shape[:-1], 15, 3)
    mirrored[..., 1395:1440] = (angular * angular.new_tensor([-1.0, 1.0, -1.0])).reshape(
        *values.shape[:-1], 45
    )
    gravity = values[..., 1440:1485].reshape(*values.shape[:-1], 15, 3)
    mirrored[..., 1440:1485] = (gravity * gravity.new_tensor([1.0, -1.0, 1.0])).reshape(
        *values.shape[:-1], 45
    )
    mirrored[..., 1485:1488] = values[..., 1485:1488] * values.new_tensor([1.0, -1.0, -1.0])
    return mirrored


@torch.no_grad()
def compute_symmetric_states(env, obs: TensorDict | None = None, actions: torch.Tensor | None = None):
    """RSL-RL callback for the G16/G17 policy, critic, AMP, and action contracts."""
    if obs is None:
        obs_aug = None
    else:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        for key in obs.keys():
            obs_aug[key][:batch_size] = obs[key]
            if key in ("policy", "critic"):
                obs_aug[key][batch_size:] = mirror_policy_history(obs[key])
            elif key == "amp":
                obs_aug[key][batch_size:] = mirror_amp_history(obs[key])
            else:
                raise KeyError(f"G17 symmetry has no transform for observation group {key!r}")

    if actions is None:
        actions_aug = None
    else:
        batch_size = actions.shape[0]
        actions_aug = torch.empty(
            (batch_size * 2, *actions.shape[1:]), device=actions.device, dtype=actions.dtype
        )
        actions_aug[:batch_size] = actions
        actions_aug[batch_size:] = mirror_full_joints(actions)
    return obs_aug, actions_aug
