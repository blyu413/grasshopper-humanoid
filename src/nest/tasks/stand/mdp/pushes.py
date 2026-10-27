"""Finite horizontal force pulses for standing-policy fine tuning."""

from __future__ import annotations

import math

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg


class RandomTorsoPulses(ManagerTermBase):
    """Update the persistent wrench each control step, clearing it after each pulse."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self.body_ids = cfg.params["asset_cfg"].body_ids
        if not isinstance(self.body_ids, list) or len(self.body_ids) != 1:
            raise ValueError("Torso pulses require exactly one resolved body")
        self.wait = torch.zeros(env.num_envs, device=env.device)
        self.remaining = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.force = torch.zeros((env.num_envs, 1, 3), device=env.device)
        self.zero_torque = torch.zeros_like(self.force)
        self.reset()

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.int32)
        low, high = self.cfg.params["interval_s"]
        self.wait[env_ids] = low + torch.rand(len(env_ids), device=self.device) * (high - low)
        self.remaining[env_ids] = 0
        self.force[env_ids] = 0
        self.asset.permanent_wrench_composer.reset(env_ids=env_ids)

    def __call__(self, env, env_ids, asset_cfg: SceneEntityCfg,
                 force_range: tuple[float, float], initial_max_force: float,
                 ramp_steps: int, duration_s: tuple[float, float],
                 interval_s: tuple[float, float], push_probability: float):
        # This term is scheduled every control step (interval_range_s=(0, 0)).
        self.wait -= env.step_dt
        self.remaining.sub_(1).clamp_(min=0)
        due = (self.wait <= 0).nonzero().flatten()
        force_limit = initial_max_force + (force_range[1] - initial_max_force) * min(
            env.common_step_counter / max(1, ramp_steps), 1.0
        )
        if len(due):
            angle = torch.rand(len(due), device=self.device) * (2 * math.pi)
            magnitude = force_range[0] + torch.rand(len(due), device=self.device) * (force_limit - force_range[0])
            magnitude *= torch.rand(len(due), device=self.device) < push_probability
            self.force[due, 0, 0] = magnitude * angle.cos()
            self.force[due, 0, 1] = magnitude * angle.sin()
            self.force[due, 0, 2] = 0
            duration = duration_s[0] + torch.rand(len(due), device=self.device) * (duration_s[1] - duration_s[0])
            self.remaining[due] = (duration / env.step_dt).round().long().clamp(min=1)
            self.wait[due] = interval_s[0] + torch.rand(len(due), device=self.device) * (interval_s[1] - interval_s[0])
        applied_force = self.force * (self.remaining > 0)[:, None, None]
        self.asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=applied_force, torques=self.zero_torque, body_ids=self.body_ids, is_global=True
        )
        env.extras.setdefault("log", {})["Push/force_limit_N"] = force_limit
        env.extras["log"]["Push/active_fraction"] = (self.remaining > 0).float().mean()
        env.extras["log"]["Push/mean_applied_force_N"] = applied_force.norm(dim=-1).mean()
