# Directory and script guide

[Project overview](../README.md) · [Setup](setup.md)

## Source packages

| Path | Responsibility |
| --- | --- |
| `src/nest/assets/robots/grasshopper.py` | Robot asset paths, URDF joint limits and actuator configuration |
| `src/nest/mbrl/supermimic/` | Model-based RL: world model, replay buffer, policy and BPTT training |
| `src/nest/mbrl/supermimic/reference/` | Pinned upstream implementation, source provenance and license |
| `src/nest/tasks/stand/` | Standing and push-recovery task configuration and force schedules |
| `src/nest/tasks/velocity/config/grasshopper/` | Grasshopper walking task and PPO configuration |
| `src/nest/tasks/velocity/config/g1/` | G1 reference tasks and PPO configuration |
| `src/nest/tasks/velocity/mdp/` | Actions, commands, curricula, events, observations and rewards |
| `src/nest/tasks/velocity/warm_start.py` | Compatible policy initialization for standing/walking/yaw |
| `src/nest/sensors/foot_height.py` | Foot-height ray pattern and reduction |

`mbrl` names the algorithm family (model-based reinforcement learning).
[SuperMimic](https://arxiv.org/abs/2610.28480) learns a differentiable world model
and uses backpropagation through time (BPTT) for policy updates through imagined
rollouts. BPTT describes that optimization step; the package also contains model
fitting, replay and environment integration. Its Python namespace is
`nest.mbrl.supermimic`.

## Script entry points

All paths below are relative to `scripts/`. Their locations are retained so existing
commands and local experiment tooling continue to work.

| Group | Scripts |
| --- | --- |
| Training | `train_grasshopper_walk.py`, `train_grasshopper_walk_wm.py`, `train_grasshopper_stand.py`, `train_grasshopper_push.py`, `train_grasshopper_stand_push_capture.py` |
| Evaluation and comparison | `test_grasshopper_push.py`, `evaluate_grasshopper_push_run.py`, `evaluate_mixed_grasshopper_run.py`, `compare_grasshopper_push.py`, `plot_grasshopper_push.py`, `benchmark_grasshopper_walk.py` |
| Model construction and export | `build_grasshopper_capsule_feet.py`, `export_grasshopper_mjcf.py`, `export_standing_checkpoint.py`, `urdf_mjcf_dynamics.py` |
| Model/physics checks | `check_grasshopper_torque_limits.py`, `check_grasshopper_foot_collisions.py`, `check_g1_foot_height.py`, `compare_grasshopper_models.py` |
| Deployment and replay | `deploy_grasshopper_mujoco.py`, `deploy_grasshopper_walk_mujoco.py`, `replay_grasshopper_walk.py` |
| Rendering | `render_grasshopper_stand_pose.py`, `render_grasshopper_walk_snapshot.py`, `render_grasshopper_walk_replay.py`, `render_grasshopper_walk_training.py`, `render_grasshopper_model_angles.py`, `render_grasshopper_presentation_views.py`, `render_grasshopper_engine.py` |
| Simulator launcher | `isaacsim-python.sh` |

`train_grasshopper_walk.py` supports training from scratch. Standing from scratch
uses the registered task through `isaaclab train`; the standing continuation scripts
default to local historical checkpoints. `export_standing_checkpoint.py` uses an
existing exported reference policy. These helpers are retained as experiment tools,
not fresh-clone setup steps. For G1 recording, use the explicit checkpoint command
in the [G1 task guide](../src/nest/tasks/velocity/config/g1/README.md).

## Assets and generated data

The [asset inventory](../assets/README.md) distinguishes the current lower-body
training model from the imported full-body/pedestal descriptions. Their model
versions and dynamics are intentionally kept separate.

| Directory | Version-control policy |
| --- | --- |
| `assets/`, `src/`, `scripts/`, `tests/`, `docs/` | Source and reusable runtime inputs |
| `logs/` | Ignored: checkpoints, TensorBoard events, training logs and policy exports |
| `outputs/` | Ignored: generated MJCF, compiled models, evaluations, videos and experiment snapshots |
| `.venv/`, cache/build directories | Ignored: machine-local installation and caches |

Training results and completed evaluations stay in place. Obsolete experiment-only
helpers and failed attempts without weights are backed up under
`outputs/_cleanup_archive/`. No checkpoint or video is required to load the
current Grasshopper training asset. No pretrained policy is selected for repository
distribution as part of this cleanup.

## Tests

`tests/` contains both `unittest.TestCase` classes and plain test functions. The
task guides retain their existing test commands. `unittest discover` executes the
class-based checks but does not execute the plain functions in `test_foot_height.py`
and `test_mjlab_g1_rewards.py`. A unified test runner and CI configuration are deferred.
