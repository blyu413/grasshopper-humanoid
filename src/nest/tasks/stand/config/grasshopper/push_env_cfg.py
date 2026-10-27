"""Standing fine tuning with sustained and transient torso-force channels."""

from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from nest.tasks.stand.mdp import MixedTorsoForces

from .env_cfg import EventCfg, GrasshopperStandEnvCfg


# ADAPT reference: tasks/momo_task/env_cfg.py and mdp/wrench_events.py.
# Local continuation steps, independent of the iteration restored from a checkpoint.
MIXED_FORCE_STAGES = [
    {"step": 0,
     "sustained": {"enabled": True, "force_range": (5.0, 20.0), "duration_s": (1.0, 2.0), "idle_s": (2.0, 4.0)},
     "pulse": {"enabled": False, "force_range": (40.0, 80.0), "duration_s": (0.1, 0.25), "idle_s": (4.0, 6.0)}},
    {"step": 250 * 24,
     "sustained": {"enabled": True, "force_range": (10.0, 50.0), "duration_s": (2.0, 4.0), "idle_s": (2.0, 4.0)},
     "pulse": {"enabled": True, "force_range": (40.0, 120.0), "duration_s": (0.1, 0.3), "idle_s": (4.0, 6.0)}},
    {"step": 750 * 24,
     "sustained": {"enabled": True, "force_range": (10.0, 80.0), "duration_s": (4.0, 6.0), "idle_s": (2.0, 4.0)},
     "pulse": {"enabled": True, "force_range": (40.0, 180.0), "duration_s": (0.1, 0.35), "idle_s": (3.0, 5.0)}},
    {"step": 1250 * 24,
     "sustained": {"enabled": True, "force_range": (10.0, 100.0), "duration_s": (5.0, 7.0), "idle_s": (2.0, 4.0)},
     "pulse": {"enabled": True, "force_range": (40.0, 200.0), "duration_s": (0.1, 0.4), "idle_s": (2.5, 4.5)}},
]


@configclass
class PushEventCfg(EventCfg):
    torso_forces = EventTermCfg(
        func=MixedTorsoForces,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["TORSO"]),
            "stages": MIXED_FORCE_STAGES,
            "clean_episode_probability": 0.2,
            "max_combined_force": 220.0,
            "settle_s": 1.0,
            "start_step": 0,
        },
    )


@configclass
class GrasshopperStandPushEnvCfg(GrasshopperStandEnvCfg):
    events: PushEventCfg = PushEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
