from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.actuators import DCMotor, DCMotorCfg
from isaaclab.utils import DelayBuffer, configclass
from isaaclab.utils.types import ArticulationActions


class CoupledAnkleDCMotor(DCMotor):
    """Two-motor differential ankle with pitch/roll coupling per side."""

    def __init__(self, cfg: CoupledAnkleDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._ankle_pairs: list[tuple[int, int]] = []
        for side in ("left", "right"):
            pitch_name = f"{side}_ankle_pitch_joint"
            roll_name = f"{side}_ankle_roll_joint"
            try:
                pitch_index = self._joint_names.index(pitch_name)
                roll_index = self._joint_names.index(roll_name)
            except ValueError as error:
                raise ValueError(
                    f"Coupled ankle actuator requires {pitch_name} and {roll_name}; "
                    f"received {self._joint_names}"
                ) from error
            self._ankle_pairs.append((pitch_index, roll_index))

    def _clip_motor_effort(self, effort: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        peak = torch.as_tensor(self.cfg.saturation_effort, dtype=effort.dtype, device=effort.device)
        no_load_speed = self.velocity_limit[:, 0].unsqueeze(1)
        velocity = torch.clamp(velocity, min=-2.0 * no_load_speed, max=2.0 * no_load_speed)
        maximum = torch.clamp(peak * (1.0 - velocity / no_load_speed), max=peak)
        minimum = torch.clamp(peak * (-1.0 - velocity / no_load_speed), min=-peak)
        return torch.clamp(effort, min=minimum, max=maximum)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        clipped = effort.clone()
        for pitch_index, roll_index in self._ankle_pairs:
            pitch_effort = effort[:, pitch_index]
            roll_effort = effort[:, roll_index]
            pitch_velocity = self._joint_vel[:, pitch_index]
            roll_velocity = self._joint_vel[:, roll_index]

            motor_effort = torch.stack(
                (
                    0.5 * (pitch_effort + roll_effort),
                    0.5 * (pitch_effort - roll_effort),
                ),
                dim=1,
            )
            motor_velocity = torch.stack(
                (
                    pitch_velocity + roll_velocity,
                    pitch_velocity - roll_velocity,
                ),
                dim=1,
            )
            motor_effort = self._clip_motor_effort(motor_effort, motor_velocity)

            clipped[:, pitch_index] = motor_effort[:, 0] + motor_effort[:, 1]
            clipped[:, roll_index] = motor_effort[:, 0] - motor_effort[:, 1]
        return clipped


@configclass
class CoupledAnkleDCMotorCfg(DCMotorCfg):
    class_type: type = CoupledAnkleDCMotor


class DelayedCoupledAnkleDCMotor(CoupledAnkleDCMotor):
    """Differential ankle motor with EngineAI-compatible command delay."""

    cfg: DelayedCoupledAnkleDCMotorCfg

    def __init__(self, cfg: DelayedCoupledAnkleDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        num_envs = self._num_envs if env_ids is None or env_ids == slice(None) else len(env_ids)
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        for buffer in (
            self.positions_delay_buffer,
            self.velocities_delay_buffer,
            self.efforts_delay_buffer,
        ):
            buffer.set_time_lag(time_lags, env_ids)
            buffer.reset(env_ids)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DelayedCoupledAnkleDCMotorCfg(CoupledAnkleDCMotorCfg):
    class_type: type = DelayedCoupledAnkleDCMotor
    min_delay: int = 0
    max_delay: int = 0


class CoupledAnkleContinuousDCMotor(CoupledAnkleDCMotor):
    """Differential ankle that also enforces each motor's continuous torque."""

    def _clip_motor_effort(self, effort: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        peak = torch.as_tensor(self.cfg.saturation_effort, dtype=effort.dtype, device=effort.device)
        continuous = torch.as_tensor(self.cfg.effort_limit, dtype=effort.dtype, device=effort.device)
        no_load_speed = self.velocity_limit[:, 0].unsqueeze(1)
        velocity_at_limit = no_load_speed * (1.0 + continuous / peak)
        velocity = torch.clamp(velocity, min=-velocity_at_limit, max=velocity_at_limit)
        maximum = torch.clamp(peak * (1.0 - velocity / no_load_speed), max=continuous)
        minimum = torch.clamp(peak * (-1.0 - velocity / no_load_speed), min=-continuous)
        return torch.clamp(effort, min=minimum, max=maximum)


@configclass
class CoupledAnkleContinuousDCMotorCfg(DCMotorCfg):
    class_type: type = CoupledAnkleContinuousDCMotor


class DelayedDCMotor(DCMotor):
    """Standard DC motor with EngineAI-compatible command delay."""

    cfg: DelayedDCMotorCfg

    def __init__(self, cfg: DelayedDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        num_envs = self._num_envs if env_ids is None or env_ids == slice(None) else len(env_ids)
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        for buffer in (
            self.positions_delay_buffer,
            self.velocities_delay_buffer,
            self.efforts_delay_buffer,
        ):
            buffer.set_time_lag(time_lags, env_ids)
            buffer.reset(env_ids)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DelayedDCMotorCfg(DCMotorCfg):
    class_type: type = DelayedDCMotor
    min_delay: int = 0
    max_delay: int = 0
