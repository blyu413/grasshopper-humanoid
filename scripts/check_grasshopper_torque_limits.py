#!/usr/bin/env python3
"""Verify that runtime joint torque limits match the Grasshopper URDF."""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--physics", choices=("physx", "newton_mjwarp"), required=True)
parser.add_argument("--num-envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

from isaaclab.app import launch_simulation
from isaaclab.assets import AssetBaseCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils

from nest.assets.robots.grasshopper import (
    JOINT_EFFORT_LIMITS,
    POLICY_JOINT_NAMES,
    make_grasshopper_cfg,
)


@configclass
class TorqueLimitCheckSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot = make_grasshopper_cfg(args.physics)


def main(physics_cfg: PhysicsCfg) -> None:
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args.device, physics=physics_cfg))
    scene = InteractiveScene(TorqueLimitCheckSceneCfg(num_envs=args.num_envs, env_spacing=2.5))
    sim.reset()

    robot = scene["robot"]
    runtime_limits = robot.data.joint_effort_limits.torch[0].cpu().tolist()
    actual = dict(zip(robot.joint_names, runtime_limits, strict=True))
    mismatches = {
        name: (JOINT_EFFORT_LIMITS[name], actual.get(name))
        for name in JOINT_EFFORT_LIMITS
        if name not in actual or not math.isclose(actual[name], JOINT_EFFORT_LIMITS[name], rel_tol=1e-6)
    }
    if mismatches:
        raise AssertionError(f"Runtime torque limits differ from URDF: {mismatches}")

    print(f"backend={args.physics}")
    print(f"runtime_joints={len(actual)} independent_policy_joints={len(POLICY_JOINT_NAMES)}")
    print(f"effort_limit_range_Nm={min(actual.values()):g}..{max(actual.values()):g}")
    print("torque_limits_match_urdf=true")


if __name__ == "__main__":
    if args.physics == "physx" and not args.experience:
        args.experience = "isaaclab.python.kit"
    with launch_simulation(PhysicsCfg(), args) as selected_physics_cfg:
        if selected_physics_cfg is None:
            raise RuntimeError("No physics backend selected")
        main(selected_physics_cfg)
