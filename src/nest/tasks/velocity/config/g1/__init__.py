"""G1 foot-height probe and MJLab-style reward task registrations."""

import gymnasium as gym

from . import agents


gym.register(
    id="Nest-Velocity-Flat-G1-Foot-Height",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:G1FootHeightEnvCfg"},
)

gym.register(
    id="Nest-Velocity-Flat-G1-MJLab-Rewards",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mjlab_env_cfg:G1MJLabRewardsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MJLabRewardsPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Nest-Velocity-Flat-G1-MJLab-Rewards-Video",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mjlab_env_cfg:G1MJLabRewardsVideoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MJLabRewardsPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
