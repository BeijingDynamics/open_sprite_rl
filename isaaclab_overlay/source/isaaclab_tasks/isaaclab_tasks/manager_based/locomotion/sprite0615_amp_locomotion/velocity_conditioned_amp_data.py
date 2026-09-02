from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

try:
    from .g16_symmetry import mirror_amp_history
except ImportError:  # Support the standalone loader contract test.
    from g16_symmetry import mirror_amp_history


class VelocityConditionedAMPDataLoader:
    """Sample expert motion clips conditional on a continuous forward command."""

    def __init__(
        self,
        motion_directory: str,
        device: str = "cpu",
        history_length: int = 5,
        stand_probability: float = 0.1,
        command_min: float = 0.15,
        command_max: float = 0.45,
        mirror_probability: float = 0.0,
        condition_source: str = "target",
    ) -> None:
        root = Path(motion_directory).expanduser().resolve()
        paths = sorted(root.glob("*.npz"))
        if not paths:
            raise AssertionError(f"No conditioned expert clips in {root}")
        if history_length < 1:
            raise ValueError("history_length must be positive")
        if not 0.0 <= stand_probability < 1.0:
            raise ValueError("stand_probability must be in [0, 1)")
        if not 0.0 < command_min < command_max:
            raise ValueError("invalid forward command range")
        if not 0.0 <= mirror_probability <= 1.0:
            raise ValueError("mirror_probability must be in [0, 1]")
        if condition_source not in {"target", "selected_clip"}:
            raise ValueError("condition_source must be 'target' or 'selected_clip'")

        self.device = device
        self.history_length = history_length
        self.stand_probability = stand_probability
        self.command_min = command_min
        self.command_max = command_max
        self.mirror_probability = mirror_probability
        self.condition_source = condition_source
        self.clips: list[dict[str, torch.Tensor | float]] = []
        expected_names: tuple[str, ...] | None = None
        expected_fps: float | None = None

        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                required = {"fps", "joint_names", "joint_pos", "base_lin_vel_b", "command_vx"}
                missing = required - set(data.files)
                if missing:
                    raise KeyError(f"{path} missing {sorted(missing)}")
                names = tuple(str(name) for name in data["joint_names"].tolist())
                fps = float(data["fps"])
                joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
                base_lin_vel_b = np.asarray(data["base_lin_vel_b"], dtype=np.float32)
                command_vx = float(np.asarray(data["command_vx"]).reshape(-1)[0])

            if expected_names is None:
                expected_names = names
                expected_fps = fps
            if names != expected_names:
                raise ValueError(f"joint order differs in {path}")
            if not math.isclose(fps, expected_fps or fps, rel_tol=0.0, abs_tol=1.0e-6):
                raise ValueError(f"mixed expert FPS in {path}: {fps} vs {expected_fps}")
            if joint_pos.ndim != 2 or joint_pos.shape[1] != len(names):
                raise ValueError(f"{path} joint_pos/name mismatch: {joint_pos.shape} vs {len(names)}")
            if joint_pos.shape[1] not in (23, 31):
                raise ValueError(f"{path} must use the 23- or 31-joint Sprite contract")
            if base_lin_vel_b.shape != (joint_pos.shape[0], 3):
                raise ValueError(f"{path} base_lin_vel_b shape mismatch")
            if joint_pos.shape[0] < history_length:
                raise ValueError(f"{path} is shorter than AMP history")
            if not np.isfinite(joint_pos).all() or not np.isfinite(base_lin_vel_b).all():
                raise ValueError(f"non-finite expert values in {path}")

            self.clips.append(
                {
                    "joint_pos": torch.as_tensor(joint_pos, device=device),
                    "base_lin_vel_b": torch.as_tensor(base_lin_vel_b, device=device),
                    "command_vx": command_vx,
                }
            )

        self.clips.sort(key=lambda clip: float(clip["command_vx"]))
        labels = [float(clip["command_vx"]) for clip in self.clips]
        if labels[0] != 0.0 or labels[-1] < command_max - 1.0e-6:
            raise ValueError(f"expert labels must cover stand through {command_max}: {labels}")
        if any(right <= left for left, right in zip(labels, labels[1:])):
            raise ValueError(f"expert command labels must be strictly increasing: {labels}")
        self.labels = torch.tensor(labels, dtype=torch.float32, device=device)
        self.joint_count = len(expected_names or ())
        self.frame_dim = self.joint_count + 4
        self.time_step_total = sum(int(clip["joint_pos"].shape[0]) for clip in self.clips)

    def _sample_targets(self, count: int) -> torch.Tensor:
        targets = torch.empty(count, device=self.device).uniform_(self.command_min, self.command_max)
        stand = torch.rand(count, device=self.device) < self.stand_probability
        targets[stand] = 0.0
        return targets

    def _select_clips(self, targets: torch.Tensor) -> torch.Tensor:
        upper = torch.searchsorted(self.labels, targets).clamp(max=len(self.clips) - 1)
        lower = (upper - 1).clamp(min=0)
        low_value = self.labels[lower]
        high_value = self.labels[upper]
        span = high_value - low_value
        probability_upper = torch.where(
            span > 1.0e-8, (targets - low_value) / span, torch.zeros_like(targets)
        )
        return torch.where(torch.rand_like(targets) < probability_upper, upper, lower)

    def _sample_batch(self, count: int) -> torch.Tensor:
        targets = self._sample_targets(count)
        selected = self._select_clips(targets)
        output = torch.empty(
            (count, self.history_length * self.frame_dim), dtype=torch.float32, device=self.device
        )
        offsets = torch.arange(
            -(self.history_length - 1), 1, dtype=torch.long, device=self.device
        )

        for clip_index, clip in enumerate(self.clips):
            rows = torch.nonzero(selected == clip_index, as_tuple=False).squeeze(-1)
            if rows.numel() == 0:
                continue
            joint_pos = clip["joint_pos"]
            base_lin_vel_b = clip["base_lin_vel_b"]
            current = torch.randint(0, joint_pos.shape[0], (rows.numel(),), device=self.device)
            history = (current[:, None] + offsets[None, :]).clamp(min=0)
            if self.condition_source == "selected_clip":
                condition_values = self.labels[selected[rows]]
            else:
                condition_values = targets[rows]
            condition = (condition_values / self.command_max).view(-1, 1, 1)
            condition = condition.expand(-1, self.history_length, 1)
            features = torch.cat(
                (joint_pos[history] * 9.0, base_lin_vel_b[history] * 7.0, condition), dim=-1
            )
            output[rows] = features.reshape(rows.numel(), -1)
        if self.mirror_probability > 0.0:
            mirrored = torch.rand(count, device=self.device) < self.mirror_probability
            output[mirrored] = mirror_amp_history(output[mirrored])
        return output

    def mini_batch_generator(self, num_mini_batches: int, num_epoches: int) -> Iterator[torch.Tensor]:
        batch_size = math.ceil(self.time_step_total / num_mini_batches)
        for _ in range(num_epoches):
            for _ in range(num_mini_batches):
                yield self._sample_batch(batch_size)
