"""Joint position targets with MJLab-style calibration bias and motor latency."""

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


class PhysicsStepDelay:
    """Per-robot command lag in physics ticks, shared by all driven joints."""

    def __init__(self, neutral: torch.Tensor, min_lag: int, max_lag: int, update_period: int):
        if not 0 <= min_lag <= max_lag or update_period < 1:
            raise ValueError("Require 0 <= min_lag <= max_lag and a positive update period")
        self.min_lag, self.max_lag, self.update_period = min_lag, max_lag, update_period
        self.values = neutral.unsqueeze(0).repeat(max_lag + 1, 1, 1)
        self.env_indices = torch.arange(neutral.shape[0], device=neutral.device)
        self.lags = torch.empty(neutral.shape[0], dtype=torch.long, device=neutral.device)
        self.phases = torch.empty_like(self.lags)
        self.cursor = 0
        self.ticks = 0
        self.reset(neutral)

    def reset(self, neutral: torch.Tensor, env_ids=None):
        ids = self.env_indices if env_ids is None else env_ids
        self.values[:, ids] = neutral[ids].unsqueeze(0)
        self.lags[ids] = torch.randint(
            self.min_lag, self.max_lag + 1, (len(ids),), device=self.values.device)
        self.phases[ids] = torch.randint(self.update_period, (len(ids),), device=self.values.device)

    def compute(self, target: torch.Tensor) -> torch.Tensor:
        # Stagger resampling as MJLab does; no per-env CPU synchronization.
        due = (self.ticks + self.phases).remainder(self.update_period) == 0
        draws = torch.randint(self.min_lag, self.max_lag + 1, self.lags.shape, device=target.device)
        self.lags.copy_(torch.where(due, draws, self.lags))
        self.values[self.cursor].copy_(target)
        rows = (self.cursor - self.lags).remainder(self.max_lag + 1)
        delayed = self.values[rows, self.env_indices]
        self.cursor = (self.cursor + 1) % (self.max_lag + 1)
        self.ticks += 1
        return delayed


class BiasedJointPositionAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._delay_buffer = None
        if getattr(cfg, "delay_max_lag", 0) > 0:
            self._delay_buffer = PhysicsStepDelay(
                self._asset.data.default_joint_pos.torch[:, self._joint_ids],
                cfg.delay_min_lag, cfg.delay_max_lag, cfg.delay_update_period,
            )

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if self._delay_buffer is not None:
            self._delay_buffer.reset(self._asset.data.default_joint_pos.torch[:, self._joint_ids], env_ids)

    def apply_actions(self):
        target = self.processed_actions
        if getattr(self, "_delay_buffer", None) is not None:
            target = self._delay_buffer.compute(target)
        if hasattr(self._env, "_velocity_encoder_bias"):
            target = target - self._env._velocity_encoder_bias
        self._asset.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)


@configclass
class BiasedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = BiasedJointPositionAction
    delay_min_lag: int = 0
    delay_max_lag: int = 0
    delay_update_period: int = 50

    def __post_init__(self):
        if not 0 <= self.delay_min_lag <= self.delay_max_lag or self.delay_update_period < 1:
            raise ValueError("Require 0 <= delay_min_lag <= delay_max_lag and a positive update period")
