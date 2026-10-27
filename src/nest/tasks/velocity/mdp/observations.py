"""Privileged foot observations for the MJLab-style flat-ground critic."""

import torch

from isaaclab.managers import SceneEntityCfg

from .rewards import _feet_contact, _foot_clearances


def foot_height(env, asset_cfg: SceneEntityCfg, foot_offsets) -> torch.Tensor:
    return _foot_clearances(env, asset_cfg, foot_offsets)


def foot_air_time(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[sensor_cfg.name].data.current_air_time.torch[:, sensor_cfg.body_ids]


def foot_contact(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    return _feet_contact(env, sensor_cfg).float()


def foot_contact_forces(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor = env.scene[sensor_cfg.name]
    force = sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids]
    if sensor.data.net_friction_forces_w is not None:
        force = force + sensor.data.net_friction_forces_w.torch[:, sensor_cfg.body_ids]
    return force.flatten(start_dim=1)
