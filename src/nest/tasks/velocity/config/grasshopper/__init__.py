"""Grasshopper flat-ground walking task registration."""

import gymnasium as gym


gym.register(
    id="Nest-Walk-Flat-Grasshopper",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GrasshopperWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:GrasshopperWalkPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
