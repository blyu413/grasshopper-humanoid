"""Grasshopper standing environment registration."""

import gymnasium as gym

from . import agents


gym.register(
    id="Nest-Stand-Grasshopper",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GrasshopperStandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GrasshopperStandPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Nest-Stand-Grasshopper-Push",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.push_env_cfg:GrasshopperStandPushEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GrasshopperStandPushPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
