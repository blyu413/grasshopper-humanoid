"""Manager-based Grasshopper standing environment."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import PresetCfg, preset

from nest.assets.robots.grasshopper import GRASSHOPPER_STAND_CFG, POLICY_JOINT_NAMES, STAND_ROOT_HEIGHT


_POLICY_JOINT_CFG = SceneEntityCfg("robot", joint_names=list(POLICY_JOINT_NAMES), preserve_order=True)
_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=1.0,
    dynamic_friction=0.8,
    restitution=0.0,
)


@configclass
class GrasshopperPhysicsCfg(PresetCfg):
    """PhysX and Newton/MJWarp presets for the standing task."""

    isaacsim_physx: PhysxCfg = PhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx)
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=512, nconmax=64, integrator="implicitfast"),
        num_substeps=2,
    )
    default: NewtonCfg = newton_mjwarp


@configclass
class GrasshopperStandSceneCfg(InteractiveSceneCfg):
    """Flat ground and a free-base Grasshopper."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0), physics_material=_PHYSICS_MATERIAL),
    )
    robot: ArticulationCfg = GRASSHOPPER_STAND_CFG.copy()
    robot.spawn.variants = preset(
        default={"Physics": "mujoco"},
        newton_mjwarp={"Physics": "mujoco"},
        isaacsim_physx={"Physics": "physx"},
        physx={"Physics": "physx"},
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class ActionsCfg:
    """Small position offsets around the assembled standing pose."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(POLICY_JOINT_NAMES),
        preserve_order=True,
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Proprioceptive observations for the standing policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _POLICY_JOINT_CFG})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _POLICY_JOINT_CFG})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Small root perturbations and a consistent reference-pose joint reset."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "roll": (-0.03, 0.03),
                "pitch": (-0.03, 0.03),
                "yaw": (-0.1, 0.1),
            },
            "velocity_range": {},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )


@configclass
class RewardsCfg:
    """Reward survival and penalize falling, drift and excessive actuation."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    termination = RewTerm(func=mdp.terminated_penalty, weight=-5.0)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    height = RewTerm(func=mdp.base_height_l2, weight=-2.0, params={"target_height": STAND_ROOT_HEIGHT})
    vertical_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=-0.05, params={"asset_cfg": _POLICY_JOINT_CFG})
    joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-0.001, params={"asset_cfg": _POLICY_JOINT_CFG})
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6, params={"asset_cfg": _POLICY_JOINT_CFG})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    success_rate = RewTerm(func=mdp.survival_success_rate, weight=0.0)


@configclass
class TerminationsCfg:
    """End an episode when the robot tips over or its root falls."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    root_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.7})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class GrasshopperStandEnvCfg(ManagerBasedRLEnvCfg):
    """Free-base standing task with 14 independent joint-position actions."""

    scene: GrasshopperStandSceneCfg = GrasshopperStandSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=2,
        physics=GrasshopperPhysicsCfg(),
        physics_material=_PHYSICS_MATERIAL,
    )

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(3.0, 3.0, 2.0), lookat=(0.0, 0.0, 1.0))
