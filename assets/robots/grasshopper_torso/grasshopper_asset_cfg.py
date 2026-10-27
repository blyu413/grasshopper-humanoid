"""Compatibility import for the NEST Grasshopper asset configuration."""

from nest.assets.robots.grasshopper import (  # noqa: F401
    GRASSHOPPER_NEWTON_CFG,
    GRASSHOPPER_PHYSX_CFG,
    JOINT_EFFORT_LIMITS,
    JOINT_VELOCITY_LIMITS,
    POLICY_JOINT_NAMES,
    PROJECT_ROOT,
    URDF_PATH,
    USD_PATH,
    make_grasshopper_cfg,
)
