"""MJLab velocity rewards adapted to Isaac Lab's data interfaces.

The formulas here mirror ``mjlab.tasks.velocity.mdp.rewards``. Foot-site
velocity is reconstructed from the foot link's rigid-body twist. G1 uses
ray-caster sites; flat-ground robots can supply local sole offsets instead.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from nest.sensors.foot_height import foot_heights


def _metric(env, name: str, value: torch.Tensor) -> None:
    if hasattr(env, "extras"):
        env.extras.setdefault("log", {})[f"Metrics/{name}"] = value.detach()


def _command_active(env, command_name: str, threshold: float) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return (torch.linalg.vector_norm(command[:, :2], dim=-1) + command[:, 2].abs()) > threshold


def _feet_contact(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor = env.scene[sensor_cfg.name]
    force = sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids]
    return torch.linalg.vector_norm(force, dim=-1) > 0.0


def _foot_site_positions(env, asset_cfg: SceneEntityCfg, foot_offsets=None) -> torch.Tensor:
    """World positions of configured sole sites, or the G1 ray-caster sites."""
    if foot_offsets is None:
        return torch.stack(
            [env.scene[f"{side}_foot_height"].data.pos_w.torch for side in ("left", "right")], dim=1
        )
    data = env.scene[asset_cfg.name].data
    positions = data.body_link_pos_w.torch[:, asset_cfg.body_ids]
    rotations = data.body_link_quat_w.torch[:, asset_cfg.body_ids]
    offsets = positions.new_tensor(foot_offsets).expand_as(positions)
    return positions + quat_apply(rotations, offsets)


def _foot_clearances(env, asset_cfg: SceneEntityCfg, foot_offsets=None, ground_height=0.0) -> torch.Tensor:
    if foot_offsets is None:
        return foot_heights(env.scene)
    # Flat-ground tasks can use the exact plane height without terrain ray casts.
    return (_foot_site_positions(env, asset_cfg, foot_offsets)[:, :, 2] - ground_height).clamp_min(0.0)


def _foot_site_velocity_xy(env, asset_cfg: SceneEntityCfg, foot_offsets=None) -> torch.Tensor:
    """Rigid-body point velocity at the selected foot sites, in world XY."""
    robot = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    site_positions = _foot_site_positions(env, asset_cfg, foot_offsets)
    link_positions = robot.data.body_link_pos_w.torch[:, body_ids]
    link_velocities = robot.data.body_link_lin_vel_w.torch[:, body_ids]
    link_angular_velocities = robot.data.body_link_ang_vel_w.torch[:, body_ids]
    point_velocities = link_velocities + torch.cross(
        link_angular_velocities, site_positions - link_positions, dim=-1
    )
    return point_velocities[:, :, :2]


def track_linear_velocity(env, std: float, command_name: str) -> torch.Tensor:
    """MJLab: exp(-(body-frame XY tracking error plus vertical speed²) / std²)."""
    command = env.command_manager.get_command(command_name)
    velocity = env.scene["robot"].data.root_link_lin_vel_b.torch
    _metric(env, "velocity_error_xy_m_s", (command[:, :2] - velocity[:, :2]).norm(dim=-1).mean())
    error = (command[:, :2] - velocity[:, :2]).square().sum(dim=-1) + velocity[:, 2].square()
    return torch.exp(-error / std**2)


def track_angular_velocity(env, std: float, command_name: str) -> torch.Tensor:
    """MJLab: exp(-(body-frame yaw tracking error plus roll/pitch speed²) / std²)."""
    command = env.command_manager.get_command(command_name)
    velocity = env.scene["robot"].data.root_link_ang_vel_b.torch
    _metric(env, "velocity_error_yaw_rad_s", (command[:, 2] - velocity[:, 2]).abs().mean())
    error = (command[:, 2] - velocity[:, 2]).square() + velocity[:, :2].square().sum(dim=-1)
    return torch.exp(-error / std**2)


def upright(env, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """MJLab torso-upright exponential reward on flat ground."""
    torso_quat = env.scene[asset_cfg.name].data.body_link_quat_w.torch[:, asset_cfg.body_ids[0]]
    gravity = torso_quat.new_tensor((0.0, 0.0, -1.0)).expand(torso_quat.shape[0], -1)
    projected = quat_apply_inverse(torso_quat, gravity)
    return torch.exp(-projected[:, :2].square().sum(dim=-1) / std**2)


class VariablePosture(ManagerTermBase):
    """MJLab speed-dependent posture kernel on the selected robot joints."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        robot = env.scene[asset_cfg.name]
        self._joint_ids = asset_cfg.joint_ids
        joint_names = [robot.joint_names[index] for index in self._joint_ids]
        self._std_standing = self._resolve_std(cfg.params["std_standing"], joint_names, env.device)
        self._std_walking = self._resolve_std(cfg.params["std_walking"], joint_names, env.device)
        self._std_running = self._resolve_std(cfg.params["std_running"], joint_names, env.device)

    @staticmethod
    def _resolve_std(patterns: dict[str, float], names: list[str], device: str) -> torch.Tensor:
        values = []
        for name in names:
            matches = [value for pattern, value in patterns.items() if re.fullmatch(pattern, name)]
            if len(matches) != 1:
                raise ValueError(f"Expected one MJLab posture tolerance for {name!r}, got {len(matches)}")
            values.append(matches[0])
        return torch.tensor(values, device=device, dtype=torch.float32)

    def __call__(
        self,
        env,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        std_standing: dict[str, float],
        std_walking: dict[str, float],
        std_running: dict[str, float],
        walking_threshold: float = 0.05,
        running_threshold: float = 1.5,
    ) -> torch.Tensor:
        del std_standing, std_walking, std_running
        robot = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        speed = torch.linalg.vector_norm(command[:, :2], dim=-1) + command[:, 2].abs()
        std = torch.where(
            (speed < walking_threshold).unsqueeze(-1),
            self._std_standing,
            torch.where((speed < running_threshold).unsqueeze(-1), self._std_walking, self._std_running),
        )
        delta = robot.data.joint_pos.torch[:, self._joint_ids] - robot.data.default_joint_pos.torch[:, self._joint_ids]
        return torch.exp(-(delta.square() / std.square()).mean(dim=-1))


def body_angular_velocity_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    velocity = env.scene[asset_cfg.name].data.body_link_ang_vel_w.torch[:, asset_cfg.body_ids[0], :2]
    return velocity.square().sum(dim=-1)


def angular_momentum_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Whole-body angular momentum about its center of mass, squared.

    Reconstructs MuJoCo's subtree angular-momentum sensor from per-body
    masses, link-frame inertia tensors, COM poses, and world velocities.
    """
    data = env.scene[asset_cfg.name].data
    mass = data.body_mass.torch
    position = data.body_com_pos_w.torch
    velocity = data.body_com_lin_vel_w.torch
    angular_velocity = data.body_com_ang_vel_w.torch
    total_mass = mass.sum(dim=1, keepdim=True).clamp_min(1e-6)
    center = (mass.unsqueeze(-1) * position).sum(dim=1) / total_mass
    center_velocity = (mass.unsqueeze(-1) * velocity).sum(dim=1) / total_mass
    orbital = torch.cross(
        position - center.unsqueeze(1),
        mass.unsqueeze(-1) * (velocity - center_velocity.unsqueeze(1)),
        dim=-1,
    ).sum(dim=1)
    inertia = data.body_inertia.torch.reshape(mass.shape[0], mass.shape[1], 3, 3)
    # Newton's model.body_inertia is fixed in the link frame. Rotate omega
    # into that frame before applying I, then rotate momentum back to world.
    link_quat = data.body_link_quat_w.torch
    local_omega = quat_apply_inverse(link_quat, angular_velocity)
    local_spin = torch.matmul(inertia, local_omega.unsqueeze(-1)).squeeze(-1)
    spin = quat_apply(link_quat, local_spin).sum(dim=1)
    momentum = orbital + spin
    _metric(env, "angular_momentum_mean", momentum.norm(dim=-1).mean())
    return momentum.square().sum(dim=-1)


def feet_clearance(
    env,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
    foot_offsets=None,
    ground_height: float = 0.0,
) -> torch.Tensor:
    height_error = (_foot_clearances(env, asset_cfg, foot_offsets, ground_height) - target_height).abs()
    speed = torch.linalg.vector_norm(_foot_site_velocity_xy(env, asset_cfg, foot_offsets), dim=-1)
    return (height_error * speed).sum(dim=-1) * _command_active(env, command_name, command_threshold)


class FeetSwingHeight(ManagerTermBase):
    """MJLab landing-only squared peak-height error."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self.peak_heights = torch.zeros((env.num_envs, 2), device=env.device)

    def __call__(
        self,
        env,
        sensor_cfg: SceneEntityCfg,
        target_height: float,
        command_name: str,
        command_threshold: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        foot_offsets=None,
        ground_height: float = 0.0,
    ) -> torch.Tensor:
        sensor = env.scene[sensor_cfg.name]
        in_contact = _feet_contact(env, sensor_cfg)
        heights = _foot_clearances(env, asset_cfg, foot_offsets, ground_height)
        # Preserve the normal state buffer when stepping under inference_mode:
        # a later explicit env.reset() may run outside inference_mode.
        self.peak_heights.copy_(torch.where(
            ~in_contact, torch.maximum(self.peak_heights, heights), self.peak_heights
        ))
        first_contact = sensor.compute_first_contact(dt=env.step_dt).torch[:, sensor_cfg.body_ids] > 0.5
        error = self.peak_heights / target_height - 1.0
        cost = (error.square() * first_contact).sum(dim=-1)
        cost *= _command_active(env, command_name, command_threshold)
        _metric(env, "peak_height_mean", (self.peak_heights * first_contact).sum() / first_contact.sum().clamp_min(1))
        self.peak_heights.copy_(torch.where(first_contact, 0.0, self.peak_heights))
        return cost

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self.peak_heights.zero_()
        else:
            self.peak_heights[env_ids] = 0.0


def feet_slip(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float,
    foot_offsets=None,
) -> torch.Tensor:
    speed_sq = _foot_site_velocity_xy(env, asset_cfg, foot_offsets).square().sum(dim=-1)
    contact = _feet_contact(env, sensor_cfg)
    _metric(env, "slip_velocity_mean", (speed_sq.sqrt() * contact).sum() / contact.sum().clamp_min(1))
    _metric(env, "single_support_fraction", contact.sum(dim=-1).eq(1).float().mean())
    _metric(env, "double_support_fraction", contact.all(dim=-1).float().mean())
    sensor_data = env.scene[sensor_cfg.name].data
    if hasattr(sensor_data, "current_air_time") and sensor_data.current_air_time is not None:
        air_time = sensor_data.current_air_time.torch[:, sensor_cfg.body_ids]
        _metric(env, "air_time_mean", air_time.sum() / (air_time > 0).sum().clamp_min(1))
    return (speed_sq * contact).sum(dim=-1) * _command_active(env, command_name, command_threshold)


def soft_landing(env, sensor_cfg: SceneEntityCfg, command_name: str, command_threshold: float) -> torch.Tensor:
    """Landing-force cost using full force when the backend exposes friction."""
    sensor = env.scene[sensor_cfg.name]
    force = sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids]
    friction = sensor.data.net_friction_forces_w
    if friction is not None:
        force = force + friction.torch[:, sensor_cfg.body_ids]
    first_contact = sensor.compute_first_contact(dt=env.step_dt).torch[:, sensor_cfg.body_ids]
    impact = (torch.linalg.vector_norm(force, dim=-1) * first_contact).sum(dim=-1)
    _metric(env, "landing_force_mean", impact.sum() / first_contact.sum().clamp_min(1))
    return impact * _command_active(env, command_name, command_threshold)
