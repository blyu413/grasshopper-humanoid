"""Isaac Lab G1 flat velocity task with MJLab G1 reward formulas and weights.

Only NEST's task configuration is changed. The upstream Isaac Lab task and
the sensor-only NEST task remain available as controls.
"""

import math

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab_tasks.core.velocity.velocity_env_cfg import CurriculumCfg
from isaaclab_visualizers.kit import KitVisualizerCfg

from nest.tasks.velocity import mdp

from .env_cfg import G1FootHeightEnvCfg, G1FootHeightSceneCfg


_FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]
_COMMON_JOINTS = [
    ".*_hip_.*",
    ".*_knee_joint",
    ".*_ankle_.*",
    "torso_joint",
    ".*_shoulder_.*",
    ".*_elbow_.*",
]

_STD_STANDING = {".*": 0.05}
_STD_WALKING = {
    ".*hip_pitch.*": 0.3,
    ".*hip_roll.*": 0.15,
    ".*hip_yaw.*": 0.15,
    ".*knee.*": 0.35,
    ".*ankle_pitch.*": 0.25,
    ".*ankle_roll.*": 0.1,
    # Isaac Lab's torso_joint is the available counterpart of MJLab's waist yaw.
    "torso_joint": 0.2,
    ".*shoulder_pitch.*": 0.15,
    ".*shoulder_roll.*": 0.15,
    ".*shoulder_yaw.*": 0.1,
    ".*elbow.*": 0.15,
}
_STD_RUNNING = {
    ".*hip_pitch.*": 0.5,
    ".*hip_roll.*": 0.2,
    ".*hip_yaw.*": 0.2,
    ".*knee.*": 0.6,
    ".*ankle_pitch.*": 0.35,
    ".*ankle_roll.*": 0.15,
    "torso_joint": 0.3,
    ".*shoulder_pitch.*": 0.5,
    ".*shoulder_roll.*": 0.2,
    ".*shoulder_yaw.*": 0.15,
    ".*elbow.*": 0.35,
}


@configclass
class MJLabG1SceneCfg(G1FootHeightSceneCfg):
    """Add a foot-only contact sensor so friction tracking stays inexpensive."""

    feet_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(left|right)_ankle_roll_link",
        history_length=3,
        track_air_time=True,
        track_friction_forces=True,
    )


@configclass
class MJLabG1RewardsCfg:
    """Active rewards from ``unitree_g1_flat_env_cfg`` in vendored MJLab."""

    track_linear_velocity = RewTerm(
        func=mdp.track_linear_velocity,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_angular_velocity = RewTerm(
        func=mdp.track_angular_velocity,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    upright = RewTerm(
        func=mdp.upright,
        weight=1.0,
        params={"std": math.sqrt(0.2), "asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    pose = RewTerm(
        func=mdp.VariablePosture,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=_COMMON_JOINTS),
            "command_name": "base_velocity",
            "std_standing": _STD_STANDING,
            "std_walking": _STD_WALKING,
            "std_running": _STD_RUNNING,
            "walking_threshold": 0.05,
            "running_threshold": 1.5,
        },
    )
    body_ang_vel = RewTerm(
        func=mdp.body_angular_velocity_penalty,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    angular_momentum = RewTerm(func=mdp.angular_momentum_penalty, weight=-0.02, params={})
    dof_pos_limits = RewTerm(func=isaac_mdp.joint_pos_limits, weight=-1.0, params={})
    action_rate_l2 = RewTerm(func=isaac_mdp.action_rate_l2, weight=-0.1, params={})
    foot_clearance = RewTerm(
        func=mdp.feet_clearance,
        weight=-2.0,
        params={
            "target_height": 0.1,
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", body_names=_FEET, preserve_order=True),
        },
    )
    foot_swing_height = RewTerm(
        func=mdp.FeetSwingHeight,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("feet_ground_contact", body_names=_FEET, preserve_order=True),
            "target_height": 0.1,
            "command_name": "base_velocity",
            "command_threshold": 0.05,
        },
    )
    foot_slip = RewTerm(
        func=mdp.feet_slip,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("feet_ground_contact", body_names=_FEET, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=_FEET, preserve_order=True),
            "command_name": "base_velocity",
            "command_threshold": 0.05,
        },
    )
    soft_landing = RewTerm(
        func=mdp.soft_landing,
        weight=-1.0e-5,
        params={
            "sensor_cfg": SceneEntityCfg("feet_ground_contact", body_names=_FEET, preserve_order=True),
            "command_name": "base_velocity",
            "command_threshold": 0.05,
        },
    )
    # MJLab's air_time has zero weight. Its self-collision term cannot be
    # reproduced while the current Isaac Lab G1 USD disables self-collisions.


@configclass
class MJLabG1CurriculumCfg(CurriculumCfg):
    """MJLab G1 Flat's three command-speed stages (24 steps per PPO iteration)."""

    command_vel = CurrTerm(
        func=mdp.command_velocity_stages,
        params={
            "command_name": "base_velocity",
            "velocity_stages": [
                {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
                {"step": 2_500 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
                {"step": 5_000 * 24, "lin_vel_x": (-2.0, 3.0)},
            ],
        },
    )


@configclass
class G1MJLabRewardsEnvCfg(G1FootHeightEnvCfg):
    """Keep the native G1 scene/MDP, replacing only its reward configuration."""

    scene: MJLabG1SceneCfg = MJLabG1SceneCfg(num_envs=4096, env_spacing=2.5)
    curriculum: MJLabG1CurriculumCfg = MJLabG1CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards = MJLabG1RewardsCfg()
        command_cfg = self.commands.base_velocity
        command_cfg.resampling_time_range = (3.0, 8.0)
        command_cfg.rel_standing_envs = 0.1
        command_cfg.rel_heading_envs = 0.3
        command_cfg.ranges.lin_vel_x = (-1.0, 1.0)
        command_cfg.ranges.lin_vel_y = (-1.0, 1.0)
        command_cfg.ranges.ang_vel_z = (-0.5, 0.5)


@configclass
class G1MJLabRewardsVideoEnvCfg(G1MJLabRewardsEnvCfg):
    """A fixed 0.6 m/s forward command for a comparable 10-second policy clip."""

    def play_mode(self):
        super().play_mode()
        self.curriculum.command_vel = None
        command_cfg = self.commands.base_velocity
        command_cfg.heading_command = False
        command_cfg.rel_heading_envs = 0.0
        command_cfg.rel_standing_envs = 0.0
        command_cfg.resampling_time_range = (100.0, 100.0)
        command_cfg.ranges.lin_vel_x = (0.6, 0.6)
        command_cfg.ranges.lin_vel_y = (0.0, 0.0)
        command_cfg.ranges.ang_vel_z = (0.0, 0.0)
        command_cfg.ranges.heading = None
        self.sim.visualizer_cfgs = [
            KitVisualizerCfg(
                headless=True,
                eye=(2.5, -3.0, 1.3),
                lookat=(0.0, 0.0, -0.35),
                origin_type="asset",
                origin_track_path="robot/torso_link",
            )
        ]
        self.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:kit",
                output_dir="outputs/g1_velocity_tracking",
                output_filename_prefix="g1_mjlab_rewards_final_0p6ms_10s",
                fps=50,
                video_length=500,
            )
        ]
