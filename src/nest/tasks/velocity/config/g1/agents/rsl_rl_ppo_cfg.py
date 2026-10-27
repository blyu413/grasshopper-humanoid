"""G1 runner with a separate log namespace for MJLab-style rewards."""

from isaaclab.utils import configclass
from isaaclab_tasks.core.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg


@configclass
class G1MJLabRewardsPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_flat_mjlab_rewards"
        self.max_iterations = 15_000
        self.actor.hidden_dims = [512, 256, 128]
        self.critic.hidden_dims = [512, 256, 128]
        self.actor.obs_normalization = True
        self.critic.obs_normalization = True
        self.algorithm.entropy_coef = 0.01
