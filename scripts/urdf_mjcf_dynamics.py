"""Preserve passive URDF joint damping in exported MJCF models."""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET


def read_joint_damping(urdf_path: Path) -> dict[str, float]:
    damping = {}
    for joint in ET.parse(urdf_path).getroot().findall("joint"):
        if joint.get("type") not in {"revolute", "continuous", "prismatic"}:
            continue
        dynamics = joint.find("dynamics")
        value = float(dynamics.get("damping", "0")) if dynamics is not None else 0.0
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid passive damping for {joint.get('name')}")
        damping[joint.attrib["name"]] = value
    return damping


def apply_joint_damping(root: ET.Element, damping: dict[str, float], prefix: str) -> None:
    joints = {joint.get("name"): joint for joint in root.findall("worldbody//joint")}
    missing = [name for name in damping if prefix + name not in joints]
    if missing:
        raise ValueError(f"MJCF is missing URDF joints: {missing}")
    for name, value in damping.items():
        joints[prefix + name].set("damping", str(value))
