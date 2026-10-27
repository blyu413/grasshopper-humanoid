"""Velocity-command curriculum for the NEST MJLab-style G1 task."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.envs import ManagerBasedRLEnv


class VelocityStage(TypedDict, total=False):
    step: int
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]


def apply_velocity_stage_ranges(ranges, velocity_stages: list[VelocityStage], step: int) -> None:
    """Use the same stage selection before construction and at episode resets."""
    for stage in velocity_stages:
        if step >= stage["step"]:
            for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
                if stage.get(name) is not None:
                    setattr(ranges, name, stage[name])


def velocity_curriculum_start_step(previous: dict | None, start_iteration: int,
                                   steps_per_iteration: int, command_design: str) -> int:
    """Start a new command curriculum on migration; preserve its clock on resume."""
    if previous is None:
        return 0
    if previous.get("command_design") == command_design:
        return int(previous["velocity_curriculum_start_step"])
    return start_iteration * steps_per_iteration


def command_velocity_stages(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    velocity_stages: list[VelocityStage],
    start_step: int = 0,
) -> dict[str, float]:
    """Apply MJLab's step-indexed command ranges when environments reset."""
    del env_ids
    command_cfg = env.command_manager.get_term(command_name).cfg
    apply_velocity_stage_ranges(command_cfg.ranges, velocity_stages, max(0, env.common_step_counter - start_step))
    return {
        "lin_vel_x_min": command_cfg.ranges.lin_vel_x[0],
        "lin_vel_x_max": command_cfg.ranges.lin_vel_x[1],
        "lin_vel_y_min": command_cfg.ranges.lin_vel_y[0],
        "lin_vel_y_max": command_cfg.ranges.lin_vel_y[1],
        "ang_vel_z_min": command_cfg.ranges.ang_vel_z[0],
        "ang_vel_z_max": command_cfg.ranges.ang_vel_z[1],
    }
