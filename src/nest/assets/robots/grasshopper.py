"""Configuration of the Grasshopper lower-body robot.

The URDF is the source of joint torque/velocity limits and sole capsules. Isaac Lab passes
those limits to both backends explicitly because Newton does not preserve the
imported USD drive maximum force as its runtime joint effort limit.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


PROJECT_ROOT = Path(__file__).resolve().parents[4]
URDF_PATH = PROJECT_ROOT / "assets/robots/grasshopper_torso/grasshopper_lower_body.urdf"
MESH_FEET_USD_PATH = PROJECT_ROOT / "assets/robots/grasshopper_torso/usd/grasshopper_lower_body/grasshopper_lower_body.usda"
USD_PATH = PROJECT_ROOT / "assets/robots/grasshopper_torso/grasshopper_lower_body_capsule_feet.usda"
FOOT_COLLISION_DESIGN = "mjlab_g1_seven_capsules_v1"


def _read_urdf_joint_limits() -> tuple[dict[str, float], dict[str, float], tuple[str, ...]]:
    """Read actuated joint limits [N·m, rad/s] and independent joint names."""
    root = ElementTree.parse(URDF_PATH).getroot()
    effort_limits: dict[str, float] = {}
    velocity_limits: dict[str, float] = {}
    independent_joints: list[str] = []

    for joint in root.findall("joint"):
        if joint.get("type") not in {"revolute", "continuous", "prismatic"}:
            continue
        name = joint.get("name")
        limit = joint.find("limit")
        if not name or limit is None:
            raise ValueError("Every actuated Grasshopper joint must have a name and a <limit> element")
        try:
            effort = float(limit.attrib["effort"])
            velocity = float(limit.attrib["velocity"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Joint {name!r} must define numeric effort and velocity limits") from exc
        if not math.isfinite(effort) or not math.isfinite(velocity) or effort <= 0.0 or velocity <= 0.0:
            raise ValueError(f"Joint {name!r} has invalid effort or velocity limits")

        effort_limits[name] = effort
        velocity_limits[name] = velocity
        if joint.find("mimic") is None:
            independent_joints.append(name)

    if not effort_limits:
        raise ValueError(f"No actuated joints found in {URDF_PATH}")
    return effort_limits, velocity_limits, tuple(independent_joints)


JOINT_EFFORT_LIMITS, JOINT_VELOCITY_LIMITS, POLICY_JOINT_NAMES = _read_urdf_joint_limits()
"""URDF limits [N·m, rad/s] and the 14 independent policy joints."""

DEFAULT_ROOT_HEIGHT = 0.98
"""Root-body height [m] placing the soles near the ground at zero joint angles."""

STAND_ROOT_HEIGHT = 0.964
"""Root-body height [m] placing the soles near the ground in the bent-knee pose."""

STAND_JOINT_POS = {
    "R_HIP_P_out": -0.2,
    "R_KNEE_P_out": 0.4,
    "R_ANKLE_P_out": -0.2,
    "L_HIP_P_out": 0.2,
    "L_KNEE_P_out": -0.4,
    "L_ANKLE_P_out": 0.2,
}
"""Mirrored hip/knee/ankle pitch angles [rad] for a level-foot, bent-knee stance."""


def make_grasshopper_cfg(physics: Literal["physx", "newton", "newton_mjwarp"]) -> ArticulationCfg:
    """Create the free-base Grasshopper articulation for a physics backend."""
    if physics == "physx":
        physics_variant = "physx"
    elif physics in {"newton", "newton_mjwarp"}:
        physics_variant = "mujoco"
    else:
        raise ValueError(f"Unsupported physics backend: {physics!r}")
    if not USD_PATH.is_file():
        raise FileNotFoundError(f"Converted Grasshopper USD not found: {USD_PATH}")

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(USD_PATH),
            variants={"Physics": physics_variant},
            copy_from_source=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, DEFAULT_ROOT_HEIGHT),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        joint_ordering="robot_schema",
        body_ordering="robot_schema",
        actuators={
            "joints": ImplicitActuatorCfg(
                joint_names_expr=list(JOINT_EFFORT_LIMITS),
                actuator_effort_limit=JOINT_EFFORT_LIMITS.copy(),
                joint_effort_limit=JOINT_EFFORT_LIMITS.copy(),
                actuator_velocity_limit=JOINT_VELOCITY_LIMITS.copy(),
                joint_velocity_limit=JOINT_VELOCITY_LIMITS.copy(),
                stiffness=0.0,
                damping=0.0,
            )
        },
    )


GRASSHOPPER_NEWTON_CFG = make_grasshopper_cfg("newton_mjwarp")
GRASSHOPPER_PHYSX_CFG = make_grasshopper_cfg("physx")


def make_grasshopper_stand_cfg() -> ArticulationCfg:
    """Create the official-style implicit-PD variant for the standing task."""
    cfg = GRASSHOPPER_NEWTON_CFG.copy()
    cfg.init_state.pos = (0.0, 0.0, STAND_ROOT_HEIGHT)
    cfg.init_state.joint_pos = {name: STAND_JOINT_POS.get(name, 0.0) for name in JOINT_EFFORT_LIMITS}
    driven_effort_limits = {name: JOINT_EFFORT_LIMITS[name] for name in POLICY_JOINT_NAMES}
    driven_velocity_limits = {name: JOINT_VELOCITY_LIMITS[name] for name in POLICY_JOINT_NAMES}
    passive_joint_names = [name for name in JOINT_EFFORT_LIMITS if name not in POLICY_JOINT_NAMES]
    passive_effort_limits = {name: JOINT_EFFORT_LIMITS[name] for name in passive_joint_names}
    passive_velocity_limits = {name: JOINT_VELOCITY_LIMITS[name] for name in passive_joint_names}
    cfg.actuators = {
        "driven_joints": ImplicitActuatorCfg(
            joint_names_expr=list(POLICY_JOINT_NAMES),
            actuator_effort_limit=driven_effort_limits,
            joint_effort_limit=driven_effort_limits,
            actuator_velocity_limit=driven_velocity_limits,
            joint_velocity_limit=driven_velocity_limits,
            stiffness={name: 500.0 if name in STAND_JOINT_POS else 100.0 for name in POLICY_JOINT_NAMES},
            damping={name: 15.0 if name in STAND_JOINT_POS else 5.0 for name in POLICY_JOINT_NAMES},
        ),
        "passive_mimic_joint": ImplicitActuatorCfg(
            joint_names_expr=passive_joint_names,
            actuator_effort_limit=passive_effort_limits,
            joint_effort_limit=passive_effort_limits,
            actuator_velocity_limit=passive_velocity_limits,
            joint_velocity_limit=passive_velocity_limits,
            stiffness=0.0,
            damping=0.0,
        ),
    }
    return cfg


GRASSHOPPER_STAND_CFG = make_grasshopper_stand_cfg()
