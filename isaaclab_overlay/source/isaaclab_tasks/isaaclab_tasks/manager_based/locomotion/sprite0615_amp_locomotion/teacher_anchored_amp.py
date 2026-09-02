from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from .unconditioned_amp import UnconditionedAMPPPO


class TeacherAnchoredUnconditionedAMPPPO(UnconditionedAMPPPO):
    """PM01 AMP with a frozen-policy anchor on near-straight rollout states."""

    def __init__(
        self,
        *args,
        teacher_anchor_weight: float = 0.1,
        teacher_anchor_yaw_scale: float = 0.05,
        teacher_anchor_batch_size: int = 8192,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if teacher_anchor_weight < 0.0:
            raise ValueError("teacher_anchor_weight must be non-negative")
        if teacher_anchor_yaw_scale <= 0.0:
            raise ValueError("teacher_anchor_yaw_scale must be positive")
        if teacher_anchor_batch_size <= 0:
            raise ValueError("teacher_anchor_batch_size must be positive")

        self.teacher_anchor_weight = teacher_anchor_weight
        self.teacher_anchor_yaw_scale = teacher_anchor_yaw_scale
        self.teacher_anchor_batch_size = teacher_anchor_batch_size
        self.teacher_actor = None

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str):
        cfg["algorithm"]["teacher_anchor_weight"] = cfg["teacher_anchor_weight"]
        cfg["algorithm"]["teacher_anchor_yaw_scale"] = cfg["teacher_anchor_yaw_scale"]
        cfg["algorithm"]["teacher_anchor_batch_size"] = cfg["teacher_anchor_batch_size"]
        return UnconditionedAMPPPO.construct_algorithm(obs, env, cfg, device)

    def load_teacher_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if "actor_state_dict" not in checkpoint:
            raise KeyError(f"Teacher checkpoint has no actor_state_dict: {checkpoint_path}")

        teacher = copy.deepcopy(self.actor)
        teacher.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        teacher.eval()
        teacher.requires_grad_(False)
        self.teacher_actor = teacher
        print(f"[INFO]: Loaded frozen teacher actor from {checkpoint_path}")

    def _sample_anchor_observations(self) -> TensorDict:
        observations = self.storage.observations.flatten(0, 1)
        sample_count = min(self.teacher_anchor_batch_size, observations.batch_size[0])
        indices = torch.randperm(observations.batch_size[0], device=self.device)[:sample_count]
        return observations[indices].clone()

    def _teacher_anchor_update(self, observations: TensorDict) -> tuple[float, float, float]:
        if self.teacher_actor is None:
            raise RuntimeError(
                "TeacherAnchoredUnconditionedAMPPPO requires load_teacher_checkpoint() before learning."
            )

        policy_obs = observations["policy"]
        yaw_command = policy_obs[..., -1]
        sample_weights = torch.exp(-0.5 * (yaw_command / self.teacher_anchor_yaw_scale).square())

        teacher_observations = observations.clone()
        teacher_observations["policy"][..., -1] = 0.0
        with torch.no_grad():
            teacher_actions = self.teacher_actor(teacher_observations)

        student_actions = self.actor(observations)
        per_sample_mse = (student_actions - teacher_actions).square().mean(dim=-1)
        anchor_loss = (sample_weights * per_sample_mse).sum() / sample_weights.sum().clamp_min(1.0)

        self.optimizer.zero_grad()
        (self.teacher_anchor_weight * anchor_loss).backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optimizer.step()

        active_fraction = (sample_weights >= 0.5).float().mean()
        return anchor_loss.item(), sample_weights.mean().item(), active_fraction.item()

    def update(self) -> dict[str, float]:
        anchor_observations = self._sample_anchor_observations()
        loss_dict = super().update()
        anchor_loss, mean_weight, active_fraction = self._teacher_anchor_update(anchor_observations)
        loss_dict.update(
            {
                "teacher_anchor": anchor_loss,
                "teacher_anchor_mean_weight": mean_weight,
                "teacher_anchor_active_fraction": active_fraction,
            }
        )
        return loss_dict

    def train_mode(self) -> None:
        super().train_mode()
        if self.teacher_actor is not None:
            self.teacher_actor.eval()


class CleanReplayAnchoredUnconditionedAMPPPO(UnconditionedAMPPPO):
    """PM01 AMP with behavior cloning on frozen clean teacher rollouts."""

    def __init__(
        self,
        *args,
        clean_replay_weight: float = 0.1,
        clean_replay_batch_size: int = 8192,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if clean_replay_weight < 0.0:
            raise ValueError("clean_replay_weight must be non-negative")
        if clean_replay_batch_size <= 0:
            raise ValueError("clean_replay_batch_size must be positive")
        self.clean_replay_weight = clean_replay_weight
        self.clean_replay_batch_size = clean_replay_batch_size
        self.clean_replay_obs = None
        self.clean_replay_actions = None

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str):
        cfg["algorithm"]["clean_replay_weight"] = cfg["clean_replay_weight"]
        cfg["algorithm"]["clean_replay_batch_size"] = cfg["clean_replay_batch_size"]
        return UnconditionedAMPPPO.construct_algorithm(obs, env, cfg, device)

    def load_teacher_replay(self, replay_path: str) -> None:
        replay = torch.load(replay_path, map_location="cpu", weights_only=False)
        if replay.get("schema") != "sprite0615_clean_teacher_replay_v1":
            raise ValueError(f"Unsupported teacher replay schema in {replay_path}")
        observations = replay["policy_obs"]
        actions = replay["teacher_actions"]
        if observations.ndim != 2 or observations.shape[1] != self.actor.obs_dim:
            raise ValueError(
                f"Replay observation shape {tuple(observations.shape)} does not match actor dim {self.actor.obs_dim}."
            )
        if actions.ndim != 2 or actions.shape[1] != self.storage.actions_shape[0]:
            raise ValueError(
                f"Replay action shape {tuple(actions.shape)} does not match action dim {self.storage.actions_shape[0]}."
            )
        if observations.shape[0] != actions.shape[0]:
            raise ValueError("Replay observation/action sample counts differ.")
        self.clean_replay_obs = observations.contiguous()
        self.clean_replay_actions = actions.contiguous()
        print(
            f"[INFO]: Loaded clean teacher replay from {replay_path}: "
            f"{observations.shape[0]} samples."
        )

    def _clean_replay_update(self) -> float:
        if self.clean_replay_obs is None or self.clean_replay_actions is None:
            raise RuntimeError(
                "CleanReplayAnchoredUnconditionedAMPPPO requires load_teacher_replay() before learning."
            )
        sample_count = min(self.clean_replay_batch_size, self.clean_replay_obs.shape[0])
        indices = torch.randint(self.clean_replay_obs.shape[0], (sample_count,))
        observations = self.clean_replay_obs[indices].to(self.device, non_blocking=True)
        targets = self.clean_replay_actions[indices].to(self.device, non_blocking=True)
        obs = TensorDict({"policy": observations}, batch_size=[sample_count], device=self.device)
        actions = self.actor(obs)
        replay_loss = (actions - targets).square().mean()

        self.optimizer.zero_grad()
        (self.clean_replay_weight * replay_loss).backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return replay_loss.item()

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        loss_dict["clean_replay"] = self._clean_replay_update()
        return loss_dict


class TemporalCleanReplayAnchoredUnconditionedAMPPPO(UnconditionedAMPPPO):
    """PM01 AMP with pointwise and 100 Hz temporal teacher replay anchors."""

    def __init__(
        self,
        *args,
        clean_replay_weight: float = 0.03,
        temporal_replay_weight: float = 0.03,
        clean_replay_batch_size: int = 4096,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if clean_replay_weight < 0.0 or temporal_replay_weight < 0.0:
            raise ValueError("Replay weights must be non-negative")
        if clean_replay_batch_size <= 0:
            raise ValueError("clean_replay_batch_size must be positive")
        self.clean_replay_weight = clean_replay_weight
        self.temporal_replay_weight = temporal_replay_weight
        self.clean_replay_batch_size = clean_replay_batch_size
        self.clean_replay_obs = None
        self.clean_replay_actions = None
        self.clean_replay_centers = None
        self.clean_replay_num_envs = None

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str):
        for name in ("clean_replay_weight", "temporal_replay_weight", "clean_replay_batch_size"):
            cfg["algorithm"][name] = cfg[name]
        return UnconditionedAMPPPO.construct_algorithm(obs, env, cfg, device)

    def load_teacher_replay(self, replay_path: str) -> None:
        replay = torch.load(replay_path, map_location="cpu", weights_only=False)
        if replay.get("schema") != "sprite0615_clean_teacher_sequence_replay_v2":
            raise ValueError(f"Unsupported temporal teacher replay schema in {replay_path}")
        observations = replay["policy_obs"].contiguous()
        actions = replay["teacher_actions"].contiguous()
        sequence_ids = replay["sequence_ids"].contiguous()
        sequence_steps = replay["sequence_steps"].contiguous()
        num_envs = int(replay["collection_num_envs"])
        if observations.ndim != 2 or observations.shape[1] != self.actor.obs_dim:
            raise ValueError("Temporal replay observation shape does not match the actor contract.")
        if actions.shape != (observations.shape[0], self.storage.actions_shape[0]):
            raise ValueError("Temporal replay action shape does not match the actor contract.")

        indices = torch.arange(observations.shape[0])
        center_mask = (indices >= num_envs) & (indices + num_envs < observations.shape[0])
        centers = indices[center_mask]
        valid = (
            (sequence_ids[centers - num_envs] == sequence_ids[centers])
            & (sequence_ids[centers + num_envs] == sequence_ids[centers])
            & (sequence_steps[centers - num_envs] + 1 == sequence_steps[centers])
            & (sequence_steps[centers] + 1 == sequence_steps[centers + num_envs])
        )
        centers = centers[valid]
        if centers.numel() == 0:
            raise ValueError("Temporal teacher replay contains no valid three-frame sequences.")

        self.clean_replay_obs = observations
        self.clean_replay_actions = actions
        self.clean_replay_centers = centers
        self.clean_replay_num_envs = num_envs
        print(
            f"[INFO]: Loaded temporal clean replay from {replay_path}: "
            f"{observations.shape[0]} samples, {centers.numel()} valid centers."
        )

    def _temporal_replay_update(self) -> tuple[float, float]:
        if self.clean_replay_centers is None:
            raise RuntimeError("Temporal replay must be loaded before learning.")
        count = min(self.clean_replay_batch_size, self.clean_replay_centers.numel())
        chosen = torch.randint(self.clean_replay_centers.numel(), (count,))
        centers = self.clean_replay_centers[chosen]
        stride = self.clean_replay_num_envs
        triplet_indices = torch.cat((centers - stride, centers, centers + stride))
        observations = self.clean_replay_obs[triplet_indices].to(self.device, non_blocking=True)
        targets = self.clean_replay_actions[triplet_indices].to(self.device, non_blocking=True)
        obs = TensorDict({"policy": observations}, batch_size=[3 * count], device=self.device)
        predicted = self.actor(obs)
        pred_prev, pred_now, pred_next = predicted.split(count)
        target_prev, target_now, target_next = targets.split(count)

        pointwise_loss = (pred_now - target_now).square().mean()
        predicted_second = pred_next - 2.0 * pred_now + pred_prev
        target_second = target_next - 2.0 * target_now + target_prev
        temporal_loss = (predicted_second - target_second).square().mean()
        loss = self.clean_replay_weight * pointwise_loss + self.temporal_replay_weight * temporal_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return pointwise_loss.item(), temporal_loss.item()

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        pointwise, temporal = self._temporal_replay_update()
        loss_dict["clean_replay"] = pointwise
        loss_dict["temporal_replay"] = temporal
        return loss_dict


class IntegratedTemporalCleanReplayAMPPPO(TemporalCleanReplayAnchoredUnconditionedAMPPPO):
    """Temporal replay regularization integrated into every PPO minibatch."""

    def _sample_temporal_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.clean_replay_centers is None:
            raise RuntimeError("Temporal replay must be loaded before learning.")
        count = min(self.clean_replay_batch_size, self.clean_replay_centers.numel())
        chosen = torch.randint(self.clean_replay_centers.numel(), (count,))
        centers = self.clean_replay_centers[chosen]
        stride = self.clean_replay_num_envs
        indices = torch.cat((centers - stride, centers, centers + stride))
        observations = self.clean_replay_obs[indices].to(self.device, non_blocking=True)
        targets = self.clean_replay_actions[indices].to(self.device, non_blocking=True)
        obs = TensorDict({"policy": observations}, batch_size=[3 * count], device=self.device)
        predicted = self.actor(obs)
        pred_prev, pred_now, pred_next = predicted.split(count)
        target_prev, target_now, target_next = targets.split(count)
        pointwise = (pred_now - target_now).square().mean()
        pred_second = pred_next - 2.0 * pred_now + pred_prev
        target_second = target_next - 2.0 * target_now + target_prev
        temporal = (pred_second - target_second).square().mean()
        return pointwise, temporal

    def _update_discriminator(self) -> dict[str, float]:
        if self.ppo_update_counter % 4 != 0:
            return {}
        totals = {"loss": 0.0, "grad": 0.0, "expert": 0.0, "policy": 0.0}
        reference_generator = self.discriminator_data_loader.mini_batch_generator(
            self.num_mini_batches // 2, self.num_learning_epochs
        )
        policy_generator = self.storage.mini_batch_generator(
            self.num_mini_batches // 2, self.num_learning_epochs
        )
        batches = 0
        for batch, reference in zip(policy_generator, reference_generator):
            policy_history = batch.observations["amp"]
            expert_score = self.discriminator(reference)
            policy_score = self.discriminator(policy_history)
            expert_loss = nn.functional.mse_loss(expert_score, torch.ones_like(expert_score))
            policy_loss = nn.functional.mse_loss(policy_score, -torch.ones_like(policy_score))
            grad_penalty = self.discriminator.compute_grad_pen(reference)
            discriminator_loss = 0.5 * (expert_loss + policy_loss) + grad_penalty
            self.disc_optimizer.zero_grad()
            discriminator_loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.disc_optimizer.step()
            with torch.no_grad():
                self.discriminator.update_normalization(reference.detach())
                self.discriminator.update_normalization(policy_history.detach())
            totals["loss"] += discriminator_loss.item()
            totals["grad"] += grad_penalty.item()
            totals["expert"] += expert_score.mean().item()
            totals["policy"] += policy_score.mean().item()
            batches += 1
        return {
            "discriminator_loss": totals["loss"] / batches,
            "amp_grad_penalty": totals["grad"] / batches,
            "amp_expert_score": totals["expert"] / batches,
            "amp_policy_score": totals["policy"] / batches,
        }

    def update(self) -> dict[str, float]:  # noqa: C901
        if self.actor.is_recurrent or self.critic.is_recurrent or self.rnd:
            raise NotImplementedError("Integrated temporal replay currently supports feed-forward PPO without RND.")
        loss_dict = self._update_discriminator()
        totals = {
            "value": 0.0,
            "surrogate": 0.0,
            "entropy": 0.0,
            "symmetry": 0.0,
            "clean_replay": 0.0,
            "temporal_replay": 0.0,
        }
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        updates = 0
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1e-8
                    )

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                augment = self.symmetry["data_augmentation_func"]
                batch.observations, batch.actions = augment(
                    env=self.symmetry["_env"], obs=batch.observations, actions=batch.actions
                )
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            self.actor(batch.observations, stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations)
            distribution_params = tuple(
                value[:original_batch_size] for value in self.actor.output_distribution_params
            )
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-7, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(1e-3, self.learning_rate * 1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss = torch.max(
                    (values - batch.returns).square(),
                    (value_clipped - batch.returns).square(),
                ).mean()
            else:
                value_loss = (batch.returns - values).square().mean()
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            symmetry_loss = torch.zeros((), device=self.device)
            if self.symmetry:
                augment = self.symmetry["data_augmentation_func"]
                if not self.symmetry["use_data_augmentation"]:
                    batch.observations, _ = augment(
                        env=self.symmetry["_env"], obs=batch.observations, actions=None
                    )
                mean_actions = self.actor(batch.observations.detach().clone())
                original_mean = mean_actions[:original_batch_size]
                _, mirrored_mean = augment(
                    env=self.symmetry["_env"], obs=None, actions=original_mean
                )
                symmetry_loss = nn.functional.mse_loss(
                    mean_actions[original_batch_size:],
                    mirrored_mean.detach()[original_batch_size:],
                )
                if self.symmetry["use_mirror_loss"]:
                    loss = loss + self.symmetry["mirror_loss_coeff"] * symmetry_loss

            pointwise_loss, temporal_loss = self._sample_temporal_losses()
            loss = (
                loss
                + self.clean_replay_weight * pointwise_loss
                + self.temporal_replay_weight * temporal_loss
            )
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            totals["value"] += value_loss.item()
            totals["surrogate"] += surrogate_loss.item()
            totals["entropy"] += entropy.mean().item()
            totals["symmetry"] += symmetry_loss.item()
            totals["clean_replay"] += pointwise_loss.item()
            totals["temporal_replay"] += temporal_loss.item()
            updates += 1

        self.storage.clear()
        loss_dict.update({name: value / updates for name, value in totals.items()})
        self.ppo_update_counter += 1
        return loss_dict
