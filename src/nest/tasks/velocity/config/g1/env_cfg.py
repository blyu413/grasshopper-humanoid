"""G1 flat velocity task with isolated MJLab-style foot height sensors.

This extends the native Isaac Lab task without changing its rewards or
observations. Call ``nest.sensors.foot_height.foot_heights(env.scene)`` to
retrieve the two measured clearances.
"""

from isaaclab.utils import configclass
from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.core.velocity.velocity_env_cfg import MySceneCfg

from nest.sensors.foot_height import foot_frame_cfg, foot_ray_caster_cfg


@configclass
class G1FootHeightSceneCfg(MySceneCfg):
    left_foot_frame = foot_frame_cfg("left")
    right_foot_frame = foot_frame_cfg("right")
    left_foot_height = foot_ray_caster_cfg("left")
    right_foot_height = foot_ray_caster_cfg("right")


@configclass
class G1FootHeightEnvCfg(G1FlatEnvCfg):
    scene: G1FootHeightSceneCfg = G1FootHeightSceneCfg(num_envs=4096, env_spacing=2.5)
