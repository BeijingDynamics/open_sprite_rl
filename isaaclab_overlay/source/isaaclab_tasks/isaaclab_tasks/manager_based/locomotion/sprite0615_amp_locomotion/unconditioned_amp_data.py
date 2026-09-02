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


class UnconditionedAMPDataLoader:
    """Sample a PM01-style unconditioned AMP distribution from multiple clips."""

    def __init__(
        self,
        motion_directory: str,
        device: str = "cpu",
        history_length: int = 5,
        mirror_probability: float = 0.0,
    ) -> None:
        root = Path(motion_directory).expanduser().resolve()
        paths = sorted(root.glob("*.npz"))
        if not paths:
            raise AssertionError(f"No expert clips in {root}")
        if history_length < 1:
            raise ValueError("history_length must be positive")
        if not 0.0 <= mirror_probability <= 1.0:
            raise ValueError("mirror_probability must be in [0, 1]")

        self.device = device
        self.history_length = history_length
        self.mirror_probability = mirror_probability
        self.clips: list[dict[str, torch.Tensor]] = []
        expected_names: tuple[str, ...] | None = None
        expected_fps: float | None = None

        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                required = {"fps", "joint_names", "joint_pos", "base_lin_vel_b"}
                missing = required - set(data.files)
                if missing:
                    raise KeyError(f"{path} missing {sorted(missing)}")
                names = tuple(str(name) for name in data["joint_names"].tolist())
                fps = float(data["fps"])
                joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
                base_lin_vel_b = np.asarray(data["base_lin_vel_b"], dtype=np.float32)

            if expected_names is None:
                expected_names = names
                expected_fps = fps
            if names != expected_names:
                raise ValueError(f"joint order differs in {path}")
            if not math.isclose(fps, expected_fps or fps, rel_tol=0.0, abs_tol=1.0e-6):
                raise ValueError(f"mixed expert FPS in {path}: {fps} vs {expected_fps}")
            if joint_pos.ndim != 2 or joint_pos.shape[1] != len(names):
                raise ValueError(f"{path} joint_pos/name mismatch: {joint_pos.shape} vs {len(names)}")
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
                }
            )

        self.joint_count = len(expected_names or ())
        if self.joint_count not in (23, 31):
            raise ValueError(f"unsupported joint count {self.joint_count}")
        self.frame_dim = self.joint_count + 3
        self.clip_lengths = torch.tensor(
            [int(clip["joint_pos"].shape[0]) for clip in self.clips],
            dtype=torch.float32,
            device=device,
        )
        self.clip_probabilities = self.clip_lengths / self.clip_lengths.sum()
        self.time_step_total = int(self.clip_lengths.sum().item())

    def _sample_batch(self, count: int) -> torch.Tensor:
        selected = torch.multinomial(self.clip_probabilities, count, replacement=True)
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
            features = torch.cat(
                (joint_pos[history] * 9.0, base_lin_vel_b[history] * 7.0), dim=-1
            )
            output[rows] = features.reshape(rows.numel(), -1)

        if self.mirror_probability > 0.0:
            mirrored = torch.rand(count, device=self.device) < self.mirror_probability
            output[mirrored] = mirror_amp_history(output[mirrored])
        return output

    def mini_batch_generator(
        self, num_mini_batches: int, num_epoches: int
    ) -> Iterator[torch.Tensor]:
        batch_size = math.ceil(self.time_step_total / num_mini_batches)
        for _ in range(num_epoches):
            for _ in range(num_mini_batches):
                yield self._sample_batch(batch_size)
