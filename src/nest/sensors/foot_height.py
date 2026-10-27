"""MJLab-style terrain clearance at the G1 foot sites.

The ray pattern and reduction match MJLab's TerrainHeightSensorCfg used by
motion-force-adapter. The static terrain is the only ray-cast target, so the
robot's own foot geometry does not register as ground.
"""

from __future__ import annotations

import math
from typing import Literal

import torch

from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import SensorFrameCfg
from isaaclab.utils import configclass


FOOT_SITE_OFFSET = (0.04, 0.0, -0.037)
FOOT_RING_RADIUS = 0.03
FOOT_RING_SAMPLES = 6
FOOT_RAY_MAX_DISTANCE = 1.0


def foot_ring_pattern(cfg: FootRingPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the center ray followed by six vertical rays on a 3 cm ring."""
    angles = torch.arange(cfg.num_samples, device=device, dtype=torch.float32)
    angles *= 2.0 * math.pi / cfg.num_samples
    starts = torch.zeros((cfg.num_samples + 1, 3), device=device)
    starts[1:, 0] = cfg.radius * torch.cos(angles)
    starts[1:, 1] = cfg.radius * torch.sin(angles)
    directions = torch.zeros_like(starts)
    directions[:, 2] = -1.0
    return starts, directions


@configclass
class FootRingPatternCfg(PatternBaseCfg):
    """MJLab ``RingPatternCfg.single_ring(radius=.03, num_samples=6)``."""

    func = foot_ring_pattern
    radius: float = FOOT_RING_RADIUS
    num_samples: int = FOOT_RING_SAMPLES


def foot_frame_cfg(side: Literal["left", "right"]) -> AssetBaseCfg:
    """Author a site Xform at MJLab's offset from the ankle roll link.

    A real child Xform is needed here: RayCaster's yaw-aligned ``offset.pos``
    would not follow ankle pitch/roll like the MJLab site does.
    """
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_ankle_roll_link/foot_height_frame",
        spawn=SensorFrameCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=FOOT_SITE_OFFSET),
    )


def foot_ray_caster_cfg(side: Literal["left", "right"]) -> RayCasterCfg:
    """Cast seven downward rays from one foot site against terrain only."""
    return RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_ankle_roll_link/foot_height_frame",
        spawn=None,
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=FootRingPatternCfg(),
        ray_alignment="yaw",
        max_distance=FOOT_RAY_MAX_DISTANCE,
        global_world_only=True,
    )


def reduce_foot_ray_heights(
    frame_z: torch.Tensor,
    hit_z: torch.Tensor,
    max_distance: float = FOOT_RAY_MAX_DISTANCE,
    normal_z: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute minimum vertical clearance, with MJLab's miss fallback.

    ``frame_z`` has shape ``[num_envs]`` and ``hit_z`` has shape
    ``[num_envs, num_rays]``. Isaac Lab reports misses as infinite hit
    coordinates. Newton exposes hit normals, allowing exact MJLab backface
    handling. Other backends do not expose them, so negative clearance is
    conservatively treated as zero there.
    """
    valid = torch.isfinite(hit_z)
    heights = frame_z.unsqueeze(-1) - hit_z
    if normal_z is not None:
        heights = torch.where(valid & (normal_z < 0.0), 0.0, heights)
    else:
        heights = heights.clamp_min(0.0)
    heights = torch.where(valid, heights, max_distance)
    minimum = heights.min(dim=-1).values
    return torch.where(valid.any(dim=-1), minimum, frame_z.clamp(0.0, max_distance))


def foot_heights(scene) -> torch.Tensor:
    """Read left/right terrain clearances as a ``[num_envs, 2]`` tensor."""
    heights = []
    for side in ("left", "right"):
        sensor = scene[f"{side}_foot_height"]
        data = sensor.data
        normal_z = data.ray_normals_w.torch[:, :, 2] if hasattr(data, "ray_normals_w") else None
        heights.append(
            reduce_foot_ray_heights(
                data.pos_w.torch[:, 2],
                data.ray_hits_w.torch[:, :, 2],
                sensor.cfg.max_distance,
                normal_z,
            )
        )
    return torch.stack(heights, dim=-1)
