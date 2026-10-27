"""ADAPT-inspired independent sustained-force and transient-force schedules."""

from __future__ import annotations

import math

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg


class MixedTorsoForces(ManagerTermBase):
    """Sum two independently timed horizontal forces and mask clean episodes.

    Each channel holds its sampled vector for its full duration, then waits a
    separately sampled idle duration. A pulse may overlap a sustained force;
    a channel cannot replace its own active force before it expires.
    """

    CHANNELS = ("sustained", "pulse")

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self.body_ids = cfg.params["asset_cfg"].body_ids
        if not isinstance(self.body_ids, list) or len(self.body_ids) != 1:
            raise ValueError("Mixed torso forces require one resolved body")
        self._validate(cfg.params)
        self.force = torch.zeros((2, self.num_envs, 1, 3), device=self.device)
        self.remaining = torch.zeros((2, self.num_envs), dtype=torch.long, device=self.device)
        self.idle = torch.zeros_like(self.remaining)
        self.clean = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.zero_torque = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.last_applied_force = torch.zeros_like(self.zero_torque)
        self._last_logged_stage = -1
        self.reset()

    @classmethod
    def _validate(cls, params):
        stages = params["stages"]
        if not stages or stages[0]["step"] != 0:
            raise ValueError("The force curriculum must begin at step zero")
        if any(a["step"] >= b["step"] for a, b in zip(stages, stages[1:])):
            raise ValueError("Curriculum stage steps must increase strictly")
        if not 0 <= params["clean_episode_probability"] <= 1:
            raise ValueError("Clean episode probability must be in [0, 1]")
        if params["max_combined_force"] <= 0 or params["settle_s"] < 0 or params["start_step"] < 0:
            raise ValueError("Invalid force cap, settling duration, or start step")
        for stage in stages:
            for channel in cls.CHANNELS:
                for field in ("force_range", "duration_s", "idle_s"):
                    lo, hi = stage[channel][field]
                    if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= hi):
                        raise ValueError(f"Invalid {channel} {field} range")
                    if field == "duration_s" and lo <= 0:
                        raise ValueError("Force duration must be positive")

    def _stage(self):
        step = self._env.common_step_counter + self.cfg.params["start_step"]
        index = 0
        for candidate, stage in enumerate(self.cfg.params["stages"]):
            if step >= stage["step"]:
                index = candidate
        return index, self.cfg.params["stages"][index]

    def _sample_steps(self, bounds, count):
        lo, hi = bounds
        seconds = lo + torch.rand(count, device=self.device) * (hi - lo)
        return (seconds / self._env.step_dt).round().long()

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.int32)
        _, stage = self._stage()
        for channel_index, channel in enumerate(self.CHANNELS):
            self.force[channel_index, env_ids] = 0
            self.remaining[channel_index, env_ids] = 0
            self.idle[channel_index, env_ids] = self._sample_steps(
                stage[channel]["idle_s"], len(env_ids)
            ) + math.ceil(self.cfg.params["settle_s"] / self._env.step_dt)
        self.clean[env_ids] = torch.rand(len(env_ids), device=self.device) < self.cfg.params["clean_episode_probability"]
        self.last_applied_force[env_ids] = 0
        self.asset.permanent_wrench_composer.reset(env_ids=env_ids)

    def __call__(self, env, env_ids, asset_cfg: SceneEntityCfg, stages: list[dict],
                 clean_episode_probability: float, max_combined_force: float,
                 settle_s: float, start_step: int):
        # EventManager calls this globally every control step. The two channels
        # own their force/idle clocks, rather than relying on interval re-sampling.
        stage_index, stage = self._stage()
        for channel_index, channel in enumerate(self.CHANNELS):
            config = stage[channel]
            remaining, idle = self.remaining[channel_index], self.idle[channel_index]
            was_active = remaining > 0
            remaining.sub_(1).clamp_(min=0)
            ended = was_active & (remaining == 0)
            ended_ids = ended.nonzero().flatten()
            if len(ended_ids):
                idle[ended_ids] = self._sample_steps(config["idle_s"], len(ended_ids))
            # Do not consume the newly sampled rest period on the force's last step.
            idle[~was_active] = (idle[~was_active] - 1).clamp(min=0)
            due = (~was_active & (idle == 0)).nonzero().flatten()
            if len(due):
                if not config["enabled"]:
                    self.force[channel_index, due] = 0
                    idle[due] = self._sample_steps(config["idle_s"], len(due)).clamp(min=1)
                    continue
                lo, hi = config["force_range"]
                magnitude = lo + torch.rand(len(due), device=self.device) * (hi - lo)
                angle = torch.rand(len(due), device=self.device) * (2 * math.pi)
                self.force[channel_index, due, 0, 0] = magnitude * angle.cos()
                self.force[channel_index, due, 0, 1] = magnitude * angle.sin()
                self.force[channel_index, due, 0, 2] = 0
                remaining[due] = self._sample_steps(config["duration_s"], len(due)).clamp(min=1)

        channel_forces = self.force * (self.remaining > 0)[:, :, None, None]
        applied = channel_forces.sum(dim=0)
        norms = applied.norm(dim=-1, keepdim=True)
        applied *= (max_combined_force / norms.clamp(min=1e-6)).clamp(max=1)
        applied *= (~self.clean)[:, None, None]
        self.last_applied_force.copy_(applied)
        self.asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=applied, torques=self.zero_torque, body_ids=self.body_ids, is_global=True
        )
        log = env.extras.setdefault("log", {})
        log["Push/stage"] = float(stage_index)
        log["Push/clean_fraction"] = self.clean.float().mean()
        log["Push/mean_applied_force_N"] = applied.norm(dim=-1).mean()
        log["Push/max_applied_force_N"] = applied.norm(dim=-1).max()
        for channel_index, channel in enumerate(self.CHANNELS):
            log[f"Push/{channel}_active_fraction"] = ((self.remaining[channel_index] > 0) & ~self.clean).float().mean()
            log[f"Push/{channel}_force_limit_N"] = stage[channel]["force_range"][1]
            log[f"Push/{channel}_duration_max_s"] = stage[channel]["duration_s"][1]
        if stage_index != self._last_logged_stage:
            print(f"MIXED_FORCE_STAGE index={stage_index} local_step={env.common_step_counter} "
                  f"sustained={stage['sustained']} pulse={stage['pulse']}", flush=True)
            self._last_logged_stage = stage_index
