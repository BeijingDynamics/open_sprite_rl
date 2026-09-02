from __future__ import annotations

from tensordict import TensorDict

from engineai_lab.algorithms.amp_ppo import AMPPPO
from engineai_lab.utils.AMP_discriminator import Discriminator
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv

from .velocity_conditioned_amp_data import VelocityConditionedAMPDataLoader


class VelocityConditionedAMPPPO(AMPPPO):
    """EngineAI AMP with a command-conditioned expert distribution."""

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str):
        cfg["algorithm"]["style_reward_weight"] = cfg["style_reward_weight"]
        cfg["algorithm"]["discriminator"] = Discriminator(
            input_dim_per_frame=cfg["frame_dim"],
            input_history_length=cfg["frame_length"],
            hidden_dims=cfg["discriminator_hidden_dims"],
            feature_normalization=cfg["frame_normalization"],
            device=device,
        ).to(device)
        cfg["algorithm"]["data_loader"] = VelocityConditionedAMPDataLoader(
            cfg["dataset_path"],
            history_length=cfg["frame_length"],
            stand_probability=cfg["expert_stand_probability"],
            command_min=cfg["expert_command_min"],
            command_max=cfg["expert_command_max"],
            mirror_probability=cfg.get("expert_mirror_probability", 0.0),
            condition_source=cfg.get("expert_condition_source", "target"),
            device=device,
        )
        return PPO.construct_algorithm(obs, env, cfg, device)

    def save(self) -> dict:
        """Persist AMP state omitted by the upstream EngineAI implementation."""
        saved = super().save()
        saved["discriminator_state_dict"] = self.discriminator.state_dict()
        saved["discriminator_optimizer_state_dict"] = self.disc_optimizer.state_dict()
        saved["amp_ppo_update_counter"] = self.ppo_update_counter
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Restore AMP state when available while accepting legacy actor-only checkpoints."""
        load_discriminator = load_cfg is None or load_cfg.get("discriminator", False)
        load_optimizer = load_cfg is None or load_cfg.get("optimizer", False)
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if not load_discriminator:
            return load_iteration

        discriminator_state = loaded_dict.get("discriminator_state_dict")
        if discriminator_state is None:
            print("Legacy AMP checkpoint has no discriminator state; using a fresh discriminator.")
            return load_iteration

        try:
            self.discriminator.load_state_dict(discriminator_state, strict=strict)
        except RuntimeError as error:
            print(f"AMP discriminator contract changed; using a fresh discriminator: {error}")
            return load_iteration

        if load_optimizer and "discriminator_optimizer_state_dict" in loaded_dict:
            self.disc_optimizer.load_state_dict(loaded_dict["discriminator_optimizer_state_dict"])
        self.ppo_update_counter = int(loaded_dict.get("amp_ppo_update_counter", 0))
        return load_iteration
