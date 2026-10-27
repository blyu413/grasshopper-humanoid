"""MJLab velocity sampling with explicit coverage of straight walking."""

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class MJLabVelocityCommand(UniformVelocityCommand):
    """Mix direct yaw-rate, heading, forward-only and standing commands.

    Isaac Lab supplies heading feedback and standing commands. MJLab additionally
    reserves forward-only samples. Enforce those *after* heading feedback on every
    tick so an overlapping heading mask cannot overwrite a straight command.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not 0.0 <= cfg.rel_forward_envs <= 1.0:
            raise ValueError("rel_forward_envs must be in [0, 1]")
        self.is_forward_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self.is_forward_env[env_ids] = (
            torch.rand(len(env_ids), device=self.device) < self.cfg.rel_forward_envs
        )
        forward_ids = env_ids[self.is_forward_env[env_ids]]
        self.vel_command_b[forward_ids, 0] = self.vel_command_b[forward_ids, 0].abs().clamp(
            min=min(0.3, self.cfg.ranges.lin_vel_x[1]), max=self.cfg.ranges.lin_vel_x[1],
        )
        # CommandTerm.reset() samples before the first observation, without
        # running _update_command(). Apply the mixture to that observation too.
        self._update_command()

    def _update_command(self):
        super()._update_command()
        forward_ids = self.is_forward_env & ~self.is_standing_env
        self.vel_command_b[forward_ids, 1:] = 0.0


@configclass
class MJLabVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = MJLabVelocityCommand
    rel_forward_envs: float = 0.0
