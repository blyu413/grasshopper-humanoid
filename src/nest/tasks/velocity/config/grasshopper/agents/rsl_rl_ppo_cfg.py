"""MJLab G1 PPO defaults; the standing architecture is an explicit opt-in."""

from isaaclab.utils import configclass

from nest.tasks.stand.config.grasshopper.agents.rsl_rl_ppo_cfg import GrasshopperStandPPORunnerCfg


@configclass
class GrasshopperWalkPPORunnerCfg(GrasshopperStandPPORunnerCfg):
    max_iterations = 30_000
    save_interval = 50
    experiment_name = "grasshopper_walk_mjlab_g1_1ms"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    def __post_init__(self):
        self.actor.hidden_dims = [512, 256, 128]
        self.critic.hidden_dims = [512, 256, 128]
        self.actor.obs_normalization = True
        self.critic.obs_normalization = True
        self.actor.distribution_cfg.init_std = 1.0
        self.algorithm.learning_rate = 1.0e-3
        self.algorithm.clip_param = 0.2
        self.algorithm.entropy_coef = 0.01
