"""Flat-ground walking and yaw-rate control, with a 1 m/s forward ceiling."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import mdp
from isaaclab.managers import (
    CurriculumTermCfg, EventTermCfg, ObservationGroupCfg, ObservationTermCfg,
    RewardTermCfg, SceneEntityCfg, TerminationTermCfg,
)
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from nest.assets.robots.grasshopper import (
    GRASSHOPPER_STAND_CFG, JOINT_EFFORT_LIMITS, POLICY_JOINT_NAMES, PROJECT_ROOT,
)
from nest.tasks.stand.config.grasshopper.env_cfg import (
    GrasshopperStandEnvCfg,
    GrasshopperStandSceneCfg,
    GrasshopperPhysicsCfg,
    ObservationsCfg as StandObservationsCfg,
)
from nest.tasks.velocity import mdp as velocity_mdp


MAX_FORWARD_SPEED = 1.0
TRAINING_DESIGN = "mjlab_g1_flat_grasshopper_v2"
DOMAIN_RANDOMIZATION_DESIGN = "mjlab_hardware_dr_v1"
COMMAND_DESIGN = "mjlab_g1_yaw_v1"
MAX_YAW_RATE = 0.7
POLICY_OBSERVATION_SIZE = 54
CRITIC_OBSERVATION_SIZE = 66
WALK_ACTION_SCALE = {
    name: 0.25 * JOINT_EFFORT_LIMITS[name] / GRASSHOPPER_STAND_CFG.actuators["driven_joints"].stiffness[name]
    for name in POLICY_JOINT_NAMES
}
STEPS_PER_ITERATION = 24
WALK_VELOCITY_STAGES = [
    # G1's initial yaw range and expansion at 5000 iterations. Retain the
    # Grasshopper forward ceiling and zero lateral command.
    {"step": 0, "lin_vel_x": (0.0, MAX_FORWARD_SPEED), "ang_vel_z": (-0.5, 0.5)},
    {"step": 5000 * STEPS_PER_ITERATION, "ang_vel_z": (-MAX_YAW_RATE, MAX_YAW_RATE)},
]
_WALK_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=0.6, dynamic_friction=0.6, restitution=0.0,
)
_FEET = ["R_FOOT", "L_FOOT"]
_JOINTS = SceneEntityCfg("robot", joint_names=list(POLICY_JOINT_NAMES), preserve_order=True)
# Centers of the sole surfaces in the URDF R_FOOT/L_FOOT mesh frames [m].
# The left mesh is mirrored in X. Link origins are above the soles.
FOOT_SOLE_OFFSETS = ((0.0539, 0.0, -0.0773), (-0.0539, 0.0, -0.0773))
_FOOT_ASSET = SceneEntityCfg("robot", body_names=_FEET, preserve_order=True)
_FOOT_SENSOR = SceneEntityCfg("feet_contact", body_names=_FEET, preserve_order=True)
_TORSO = SceneEntityCfg("robot", body_names="TORSO")
_STD_WALKING = {
    "[RL]_HIP_P_out": 0.3,
    "[RL]_HIP_R_out": 0.15,
    "[RL]_HIP_Y_out": 0.15,
    "[RL]_KNEE_P_out": 0.35,
    "[RL]_ANKLE_P_out": 0.25,
    "[RL]_ANKLE_R_out": 0.1,
    "WAIST_Y_out": 0.2,
    "WAIST_P_out": 0.1,
}
_STD_RUNNING = {
    "[RL]_HIP_P_out": 0.5,
    "[RL]_HIP_R_out": 0.2,
    "[RL]_HIP_Y_out": 0.2,
    "[RL]_KNEE_P_out": 0.6,
    "[RL]_ANKLE_P_out": 0.35,
    "[RL]_ANKLE_R_out": 0.15,
    "WAIST_Y_out": 0.3,
    "WAIST_P_out": 0.2,
}


@configclass
class WalkSceneCfg(GrasshopperStandSceneCfg):
    # A native infinite plane matches MJLab's collision geometry. No downloads.
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(PROJECT_ROOT / "assets/terrains/flat_plane.usda"),
            physics_material=_WALK_MATERIAL,
        ),
    )
    feet_contact = ContactSensorCfg(
        # The converted URDF retains the complete link hierarchy below Geometry.
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/.*/(R_FOOT|L_FOOT)",
        history_length=3,
        track_air_time=True,
        track_friction_forces=True,
        force_threshold=0.0,
        update_period=0.0,
    )


@configclass
class WalkObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": _JOINTS},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, params={"asset_cfg": _JOINTS},
                                       noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObservationTermCfg(func=mdp.last_action)
        velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        foot_height = ObservationTermCfg(func=velocity_mdp.foot_height,
                                         params={"asset_cfg": _FOOT_ASSET, "foot_offsets": FOOT_SOLE_OFFSETS})
        foot_air_time = ObservationTermCfg(func=velocity_mdp.foot_air_time, params={"sensor_cfg": _FOOT_SENSOR})
        foot_contact = ObservationTermCfg(func=velocity_mdp.foot_contact, params={"sensor_cfg": _FOOT_SENSOR})
        foot_contact_forces = ObservationTermCfg(func=velocity_mdp.foot_contact_forces, params={"sensor_cfg": _FOOT_SENSOR})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class LegacyPolicyCfg(StandObservationsCfg.PolicyCfg):
    """Explicit compatibility layout for a 52/55-input historical actor."""

    velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})


@configclass
class WalkActionsCfg:
    joint_pos = velocity_mdp.BiasedJointPositionActionCfg(
        asset_name="robot", joint_names=list(POLICY_JOINT_NAMES), preserve_order=True,
        scale=WALK_ACTION_SCALE, use_default_offset=True,
        # MJLab's LagHW training: dt=5ms, lag 0..4, resample every 50 ticks.
        delay_min_lag=0, delay_max_lag=4, delay_update_period=50,
    )


@configclass
class WalkEventsCfg:
    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5),
                               "z": (0.01, 0.05), "yaw": (-3.14, 3.14)}, "velocity_range": {}},
    )
    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(1.0, 3.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.4, 0.4),
                                   "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)}},
    )
    foot_friction = EventTermCfg(
        func=velocity_mdp.SharedFootFriction, mode="startup",
        params={"asset_cfg": _FOOT_ASSET, "static_friction_range": (0.3, 1.2),
                "dynamic_friction_range": (0.3, 1.2), "restitution_range": (0.0, 0.0),
                "num_buckets": 64},
    )
    encoder_bias = EventTermCfg(
        func=velocity_mdp.encoder_bias, mode="startup",
        params={"asset_cfg": _JOINTS, "bias_range": (-0.015, 0.015)},
    )
    body_density = EventTermCfg(
        func=velocity_mdp.BodyDensity, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "scale_range": (0.9, 1.1)},
    )
    actuator_gains = EventTermCfg(
        func=mdp.randomize_actuator_gains, mode="startup",
        params={"asset_cfg": _JOINTS, "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.9, 1.1), "operation": "scale"},
    )
    joint_friction = EventTermCfg(
        func=velocity_mdp.joint_friction, mode="startup",
        params={"asset_cfg": _JOINTS, "friction_range": (0.0, 0.5)},
    )


@configclass
class WalkTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    root_height = None
    bad_orientation = TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})


@configclass
class WalkPhysicsCfg(GrasshopperPhysicsCfg):
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=300, nconmax=64, integrator="implicitfast",
            # G1 uses 20 line-search iterations. Grasshopper's CAD/mimic model
            # reaches that limit in runtime checks, so retain a 50-step budget.
            iterations=10, ls_iterations=50, ccd_iterations=50, tolerance=1.0e-8,
        ),
        num_substeps=1,
    )
    default: NewtonCfg = newton_mjwarp


@configclass
class WalkCommandsCfg:
    base_velocity = velocity_mdp.MJLabVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.1,
        heading_command=True,
        heading_control_stiffness=0.5,
        rel_heading_envs=0.3,
        rel_forward_envs=0.2,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, MAX_FORWARD_SPEED), lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.5, 0.5), heading=(-math.pi, math.pi),
        ),
        vel_xy_success_threshold=0.2,
        vel_yaw_success_threshold=0.2,
    )


@configclass
class WalkCurriculumCfg:
    command_vel = CurriculumTermCfg(
        func=velocity_mdp.command_velocity_stages,
        params={"command_name": "base_velocity", "velocity_stages": WALK_VELOCITY_STAGES},
    )


@configclass
class WalkRewardsCfg:
    """MJLab G1 Flat rewards with Grasshopper joints and flat-ground sole sites."""

    track_linear_velocity = RewardTermCfg(
        func=velocity_mdp.track_linear_velocity, weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_angular_velocity = RewardTermCfg(
        func=velocity_mdp.track_angular_velocity, weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    upright = RewardTermCfg(
        func=velocity_mdp.upright, weight=1.0,
        params={"std": math.sqrt(0.2), "asset_cfg": _TORSO},
    )
    pose = RewardTermCfg(
        func=velocity_mdp.VariablePosture, weight=1.0,
        params={
            "asset_cfg": _JOINTS, "command_name": "base_velocity",
            "std_standing": {".*": 0.05}, "std_walking": _STD_WALKING, "std_running": _STD_RUNNING,
            "walking_threshold": 0.05, "running_threshold": 1.5,
        },
    )
    body_ang_vel = RewardTermCfg(
        func=velocity_mdp.body_angular_velocity_penalty, weight=-0.05, params={"asset_cfg": _TORSO},
    )
    angular_momentum = RewardTermCfg(func=velocity_mdp.angular_momentum_penalty, weight=-0.02)
    joint_limits = RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": _JOINTS})
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    foot_clearance = RewardTermCfg(
        func=velocity_mdp.feet_clearance, weight=-2.0,
        params={
            "target_height": 0.1, "command_name": "base_velocity", "command_threshold": 0.05,
            "asset_cfg": _FOOT_ASSET, "foot_offsets": FOOT_SOLE_OFFSETS, "ground_height": 0.0,
        },
    )
    foot_swing_height = RewardTermCfg(
        func=velocity_mdp.FeetSwingHeight, weight=-0.25,
        params={
            "sensor_cfg": _FOOT_SENSOR, "asset_cfg": _FOOT_ASSET, "foot_offsets": FOOT_SOLE_OFFSETS,
            "target_height": 0.1, "command_name": "base_velocity", "command_threshold": 0.05,
            "ground_height": 0.0,
        },
    )
    foot_slip = RewardTermCfg(
        func=velocity_mdp.feet_slip, weight=-0.1,
        params={
            "sensor_cfg": _FOOT_SENSOR, "asset_cfg": _FOOT_ASSET, "foot_offsets": FOOT_SOLE_OFFSETS,
            "command_name": "base_velocity", "command_threshold": 0.05,
        },
    )
    soft_landing = RewardTermCfg(
        func=velocity_mdp.soft_landing, weight=-1.0e-5,
        params={"sensor_cfg": _FOOT_SENSOR, "command_name": "base_velocity", "command_threshold": 0.05},
    )
    # MJLab G1's air_time weight is zero. Its self-collision cost is not ported:
    # this task's asset/contact setup does not provide a self-contact sensor.


@configclass
class GrasshopperWalkEnvCfg(GrasshopperStandEnvCfg):
    scene: WalkSceneCfg = WalkSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: WalkObservationsCfg = WalkObservationsCfg()
    actions: WalkActionsCfg = WalkActionsCfg()
    events: WalkEventsCfg = WalkEventsCfg()
    terminations: WalkTerminationsCfg = WalkTerminationsCfg()
    commands: WalkCommandsCfg = WalkCommandsCfg()
    rewards: WalkRewardsCfg = WalkRewardsCfg()
    curriculum: WalkCurriculumCfg = WalkCurriculumCfg()
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.005, render_interval=4, physics=WalkPhysicsCfg(), physics_material=_WALK_MATERIAL,
    )

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.robot.soft_joint_pos_limit_factor = 0.9

    def use_legacy_policy_observations(self):
        self.observations.policy = LegacyPolicyCfg()
        self.actions.joint_pos.scale = 0.25

    def play_mode(self):
        super().play_mode()
        self.curriculum.command_vel = None
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.foot_friction = None
        self.events.encoder_bias = None
        self.events.body_density = None
        self.events.actuator_gains = None
        self.events.joint_friction = None
        self.actions.joint_pos.delay_min_lag = 0
        self.actions.joint_pos.delay_max_lag = 0
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = None
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_forward_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (100.0, 100.0)
