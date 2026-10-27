"""Focused checks for the NEST-side MJLab G1 reward adaptation."""

import math
from types import SimpleNamespace

import torch

from nest.tasks.velocity.config.g1.env_cfg import G1FootHeightEnvCfg
from nest.tasks.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1MJLabRewardsPPORunnerCfg
from nest.tasks.velocity.config.g1.mjlab_env_cfg import G1MJLabRewardsEnvCfg, G1MJLabRewardsVideoEnvCfg
from nest.tasks.velocity.mdp.curriculums import command_velocity_stages
from nest.tasks.velocity.mdp.rewards import track_angular_velocity, track_linear_velocity


def test_reward_weights_and_native_control_are_separate():
    adapted = G1MJLabRewardsEnvCfg()
    expected = {
        "track_linear_velocity": 2.0,
        "track_angular_velocity": 2.0,
        "upright": 1.0,
        "pose": 1.0,
        "body_ang_vel": -0.05,
        "angular_momentum": -0.02,
        "dof_pos_limits": -1.0,
        "action_rate_l2": -0.1,
        "foot_clearance": -2.0,
        "foot_swing_height": -0.25,
        "foot_slip": -0.1,
        "soft_landing": -1.0e-5,
    }
    assert {name: term.weight for name, term in vars(adapted.rewards).items()} == expected
    assert adapted.scene.feet_ground_contact.track_friction_forces is True
    assert adapted.scene.contact_forces.track_friction_forces is False
    native = G1FootHeightEnvCfg()
    assert native.rewards.feet_air_time.weight == 0.75
    assert not hasattr(native.rewards, "foot_clearance")


def test_velocity_tracking_includes_body_z_and_roll_pitch():
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_lin_vel_b=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.3]])),
            root_link_ang_vel_b=SimpleNamespace(torch=torch.tensor([[0.4, 0.0, 0.0]])),
        )
    )
    env = SimpleNamespace(
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_command=lambda _: torch.zeros((1, 3))),
    )
    torch.testing.assert_close(
        track_linear_velocity(env, std=0.5, command_name="base_velocity"),
        torch.tensor([math.exp(-0.3**2 / 0.5**2)]),
    )
    torch.testing.assert_close(
        track_angular_velocity(env, std=math.sqrt(0.5), command_name="base_velocity"),
        torch.tensor([math.exp(-0.4**2 / 0.5)]),
    )


def test_mjlab_command_curriculum_and_training_length():
    cfg = G1MJLabRewardsEnvCfg()
    runner_cfg = G1MJLabRewardsPPORunnerCfg()
    assert cfg.scene.num_envs == 4096
    assert runner_cfg.num_steps_per_env == 24
    assert runner_cfg.max_iterations == 15_000
    assert cfg.curriculum.terrain_levels is None
    assert cfg.commands.base_velocity.ranges.lin_vel_y == (-1.0, 1.0)

    command = SimpleNamespace(cfg=cfg.commands.base_velocity)
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _: command),
        common_step_counter=0,
    )
    stages = cfg.curriculum.command_vel.params["velocity_stages"]
    expected = (
        (0, (-1.0, 1.0), (-0.5, 0.5)),
        (59_999, (-1.0, 1.0), (-0.5, 0.5)),
        (60_000, (-1.5, 2.0), (-0.7, 0.7)),
        (119_999, (-1.5, 2.0), (-0.7, 0.7)),
        (120_000, (-2.0, 3.0), (-0.7, 0.7)),
    )
    for step, x_range, yaw_range in expected:
        env.common_step_counter = step
        command_velocity_stages(env, [], "base_velocity", stages)
        assert command.cfg.ranges.lin_vel_x == x_range
        assert command.cfg.ranges.ang_vel_z == yaw_range


def test_video_uses_fixed_command_and_ten_second_recording():
    cfg = G1MJLabRewardsVideoEnvCfg()
    cfg.play_mode()
    assert cfg.curriculum.command_vel is None
    assert cfg.commands.base_velocity.ranges.lin_vel_x == (0.6, 0.6)
    assert cfg.commands.base_velocity.ranges.lin_vel_y == (0.0, 0.0)
    assert cfg.commands.base_velocity.ranges.ang_vel_z == (0.0, 0.0)
    assert cfg.commands.base_velocity.heading_command is False
    assert cfg.sim.visualizer_cfgs[0].origin_track_path == "robot/torso_link"
    assert cfg.video_recorders[0].video_length == 500
    assert cfg.video_recorders[0].fps == 50
