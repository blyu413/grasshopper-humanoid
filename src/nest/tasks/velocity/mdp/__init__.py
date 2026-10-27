"""NEST velocity-task reward terms."""

from .curriculums import command_velocity_stages
from .commands import MJLabVelocityCommand, MJLabVelocityCommandCfg
from .actions import BiasedJointPositionAction, BiasedJointPositionActionCfg
from .events import BodyDensity, SharedFootFriction, encoder_bias, joint_friction
from .observations import foot_air_time, foot_contact, foot_contact_forces, foot_height
from .rewards import (
    FeetSwingHeight,
    VariablePosture,
    angular_momentum_penalty,
    body_angular_velocity_penalty,
    feet_clearance,
    feet_slip,
    soft_landing,
    track_angular_velocity,
    track_linear_velocity,
    upright,
)
