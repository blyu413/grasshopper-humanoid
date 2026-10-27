# Grasshopper walking and yaw control with MJLab G1 settings

Task: `Nest-Walk-Flat-Grasshopper`. Training design:
`mjlab_g1_flat_grasshopper_v2`. This task follows the locally vendored
`motion-force-adapter/_external/mjlab/src/mjlab/tasks/velocity/` G1 Flat
configuration. It retains the 0–1 m/s forward command ceiling; measured
velocity can overshoot. The original standing task configurations are unchanged.
The hardware randomization profile is `mjlab_hardware_dr_v1`; observation and
network layouts remain compatible with the completed v2 5000-iteration run.

| Setting | Current default |
| --- | --- |
| Physics/control | 0.005 s physics step, decimation 4, 50 Hz control |
| Solver | implicitfast, 10 Newton iterations, CCD 50, tolerance 1e-8 |
| Line search | 50 iterations; G1's 20 produced limit warnings on Grasshopper |
| Terrain/contact | Native plane; 7 capsules per foot, radius 0.01 m |
| Foot contact parameters | Priority 1, condim 3, margin/gap 0, solref (0.02, 1) |
| Actor | 54 values: linear/angular velocity, gravity, 14 joint positions/velocities/actions, command |
| Actor noise | Uniform ±0.5 m/s, ±0.2 rad/s, ±0.05 gravity, ±0.01 rad joint position, ±1.5 rad/s joint velocity |
| Critic | 66 clean values: actor state plus foot height, air time, contact and 3D force |
| Network | Actor and critic [512, 256, 128], ELU, observation normalization |
| PPO | 24 steps, std 1.0, learning rate 1e-3 adaptive, entropy 0.01, 5 epochs / 4 minibatches |
| Actions | 14 independent joints; per-joint scale = 0.25 × URDF effort limit / PD stiffness |
| Limits/episodes | Soft joint factor 0.9; 20 s; terminate at 70° root tilt |

Grasshopper retains its own bent-knee reference, root height, PD gains and URDF
torque/velocity limits. The mimic joint remains passive. Root velocities replace
G1's pelvis IMU site measurements because this model has no matching IMU site.
The critic uses the exact flat-plane height instead of terrain ray casts.

## Commands and randomization

The command profile is `mjlab_g1_yaw_v1`. It follows the vendored MJLab G1 command
mixture and angular curriculum:

| Command setting | Training behavior |
| --- | --- |
| Forward / lateral velocity | 0–1 m/s / zero |
| Direct body yaw rate | Uniform −0.5 to +0.5 rad/s initially |
| Angular expansion | −0.7 to +0.7 rad/s after 5000 **yaw-training** iterations |
| Heading samples | 30% mask; target heading uniform ±π, yaw = clip(0.5 × wrapped heading error) |
| Straight samples | 20% mask; forward speed at least 0.3 m/s, lateral/yaw zero |
| Standing samples | 10% mask; all three commands zero |
| Resampling | Every 3–8 s |

Masks are sampled independently, as in MJLab, rather than partitioning samples
into percentages that sum to 100. Standing takes precedence; straight commands
override heading feedback on every tick. The remaining samples include direct
left/right yaw commands while walking. Lateral and reverse commands remain
disabled. The policy still receives `(vx, vy, yaw_rate)`; heading feedback is
converted to yaw rate, so neither the actor nor critic changes dimensions.
The existing angular tracking reward keeps MJLab's weight 2 and std √0.5.
Posture and foot rewards already recognize yaw-only activity.

Startup draws one friction coefficient in [0.3, 1.2] per robot, shared by all 14
foot capsules. Foot priority 1 makes this coefficient win over the plane during
MuJoCo friction mixing. Encoder bias in ±0.015 rad is subtracted from position
targets, matching MJLab; joint observations remain unbiased.
Resets randomize X/Y by ±0.5 m, yaw by ±3.14 rad and height by +0.01–0.05 m.
Every 1–3 s, velocity pushes add ±0.5 m/s in X/Y, ±0.4 m/s in Z,
±0.52 rad/s in roll/pitch and ±0.78 rad/s in yaw.

Additional hardware variation follows the previous MJLab G1 LagHW training in
`motion-force-adapter/tasks/momo_lumped/env_cfg.py`
(`unitree_g1_momo_lumped_flat_hand_load_lag_hw_env_cfg`):

| Randomization | Grasshopper range | Sampling |
| --- | --- | --- |
| Body density | Mass and inertia both ×0.9–1.1, per positive-mass body | Startup, fixed per robot |
| PD gains | Kp and Kd independently ×0.9–1.1, per driven joint | Startup |
| Dry joint friction | 0–0.5 N·m, per driven joint | Startup |
| Command latency | 0–4 physics ticks = 0–20 ms, shared by the robot's joints | Resample every 50 ticks = 250 ms, staggered by robot |

Mass and inertia use the same density factor so the inertial properties remain
consistent. Zero-mass helper bodies are excluded, and COM positions remain
fixed. This is a conservative adaptation of MJLab's pseudo-inertia density
variation; its log-scale parameter is not copied numerically. PD and delay
ranges match the prior LagHW profile; dry friction uses a smaller range than
that training's final leg-friction range of 0.05–1.5 N·m.
Armature scaling is omitted because Grasshopper does not have a validated
nonzero reflected-armature configuration; scaling zero would have no effect.
Delayed targets are buffered at the 200 Hz physics rate, and episode resets
replace that robot's entire history with its reference target.

Two G1 features remain disabled: self-collision cost requires a validated
Grasshopper self-contact configuration, and torso COM randomization is excluded
because the current Isaac Lab Newton implementation documents incomplete mass
matrix recomputation after runtime COM changes. Play mode disables observation
noise, pushes and startup randomization.
Play mode also removes command latency, preserving the nominal playback used
to review the original 5k policy.

## MJLab walking rewards

The reward configuration follows the locally vendored MJLab
`mjlab/tasks/velocity/velocity_env_cfg.py` and `config/g1/env_cfgs.py`
(`unitree_g1_flat_env_cfg`) in `motion-force-adapter/_external/mjlab/src/`.
The task uses the same 12 active terms and weights as NEST's G1 MJLab adaptation,
with Grasshopper's body/joint names and sole geometry:

| Term | Weight | Behavior |
| --- | ---: | --- |
| Linear velocity | 2.0 | Body-frame XY tracking and zero vertical speed; std 0.5 |
| Angular velocity | 2.0 | Body-frame yaw tracking and zero roll/pitch speed; std √0.5 |
| Upright | 1.0 | Torso tilt exponential kernel; std √0.2 |
| Pose | 1.0 | Speed-dependent per-joint exponential kernel around the standing pose |
| Torso angular velocity | -0.05 | Squared world-frame roll/pitch angular velocity |
| Angular momentum | -0.02 | Squared whole-body angular momentum about the COM |
| Joint limits | -1.0 | Soft-limit violations on the 14 independent joints |
| Action rate | -0.1 | Squared action differences |
| Foot clearance | -2.0 | Absolute clearance error, weighted by foot-site XY speed |
| Swing height | -0.25 | Squared normalized peak-height error, charged at landing |
| Foot slip | -0.1 | Squared foot-site XY speed while in contact |
| Soft landing | -0.00001 | Contact-force magnitude on the first contact step |

The posture tolerance is 0.05 rad below a command magnitude of 0.05. Walking
tolerances follow MJLab's corresponding joints: hip pitch 0.3, hip roll/yaw 0.15,
knee 0.35, ankle pitch 0.25, ankle roll 0.1, waist yaw 0.2, and waist pitch 0.1 rad.
The running regime at a combined linear-command norm plus absolute yaw rate of
1.5 is retained for formula compatibility; it can activate during faster turns.

This replaces the inherited standing reward stack, including alive/termination,
fixed-height, joint-deviation, torque and joint-velocity terms. It also removes
the earlier positive air-time reward, whose weight is zero in MJLab G1. The
episode fall criteria still apply. MJLab's self-collision cost is omitted because
this scene has no self-contact sensor. These weights are an initial adaptation,
not a demonstrated improvement in trained gait quality.

The foot sensor matches the actual nested `Geometry/.../R_FOOT` and
`Geometry/.../L_FOOT` rigid bodies. Sole sites use local offsets
`(0.0539, 0, -0.0773)` and `(-0.0539, 0, -0.0773)` m, the centers of the URDF
mesh sole surfaces. The shared robot now uses seven longitudinal sole capsules per
foot, following MJLab G1, with radius 0.01 m and a 258 × 86 mm footprint adapted to
Grasshopper. Their bottoms remain at local Z=-0.0773 m, so the site offsets remain
valid. The original foot mesh colliders are disabled in both USD physics variants.
Site velocity includes the body's angular contribution.
Since the flat ground surface is at world Z=0, clearance uses transformed sole-site Z
directly; no terrain rays are needed. This approximation must be replaced for
uneven terrain. All four foot terms are gated off at command magnitude ≤0.05.
Landing force includes normal plus friction force when available, with normal
force as the fallback. Contact is inferred from nonzero normal force.

## Initialization and preparation

Training starts from scratch by default, as in MJLab G1. The old standing actor
can be used explicitly with `--init-checkpoint PATH`; that compatibility mode
uses the historical 55-input, [256, 128, 128] actor without normalization and
0.25 rad action scale. Its actor mean is preserved for every command, while the
critic uses the new 66-input layout and the optimizer starts fresh. The new
exploration std remains 1.0. This is an explicit departure from the default G1
network profile, not an automatic initialization.

Prepare without a GPU or simulation:

```bash
.venv/bin/python scripts/train_grasshopper_walk.py --prepare-only --visualizer none
```

Every invocation creates a new directory under `logs/rsl_rl/grasshopper_walk_mjlab_g1_1ms/`, or an explicit
new `--log-dir`. It contains environment/agent YAML, a source snapshot, and a `run.json` with the
checkpoint hash, observation/action dimensions, curriculum and versions. Preparation also writes
`initial_actor.pt` (a state dict, not a trained or deployable walking policy).
The manifest records `training_design: mjlab_g1_flat_grasshopper_v2`,
`reward_design: mjlab_g1_flat_adapted_v2` and
`foot_collision_design: mjlab_g1_seven_capsules_v1`, plus the robot USD hash.
It additionally records `domain_randomization_design: mjlab_hardware_dr_v1`,
all added ranges and the source run's DR profile when resuming. The same v2
actor, critic and optimizer can resume from the existing 5k checkpoint under
the stronger DR distribution; this does not rewrite that source checkpoint.
The manifest also records `command_design`, the source command profile, actual
initial command sample counts, and `velocity_curriculum_start_step`.
When transferring a forward-only run, the yaw curriculum starts at zero instead
of jumping to the stage implied by its previous forward-walking iteration count.
Later resumes of a yaw run retain that clock. The previously unseen yaw input
column in actor and critic is initialized to zero, its normalization std is
seeded to 0.5/√3, and only that column's optimizer moments are cleared. This
preserves zero-yaw network outputs and prevents the old zero-variance command
normalizer from amplifying new commands by 100×. All other weights, normalization
statistics and optimizer state are restored.
The source snapshot includes the shared reward implementation, URDF, capsule USD
override and its build script. Earlier runs, including the 3000-iteration MJLab
reward run, used mesh foot contacts. Historical walking runs also use the old
reward stack, so their return values are not directly comparable.
Full PPO resume requires the same v2 training design; old checkpoints have
different observations and network shapes. Playback supports both 54- and
55-input exported actors, using the current robot contacts and physics.

## Smoke checks and training

Tests for transfer equivalence/trainability, the actual USD foot paths, observation order,
unchanged standing configuration, and all speed-stage boundaries:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_grasshopper_walk.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_grasshopper_mjlab_rewards.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_grasshopper_foot_collisions.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_grasshopper_domain_randomization.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_grasshopper_yaw_control.py' -v
```

CPU smoke runs at most two PPO iterations in one native MuJoCo environment. It **omits** the foot
sensor and the three contact-dependent terms (swing height, foot slip, soft landing)
because Newton cannot export native CPU contacts. The critic's three
contact-dependent terms are also omitted; flat-ground foot height and clearance
still run.
Its logs use a separate `_cpu_smoke` experiment and status `cpu_smoke_complete`.

```bash
.venv/bin/python scripts/train_grasshopper_walk.py --cpu-smoke --iterations 2 --visualizer none
```

Full-task GPU smoke, including foot-contact rewards:

```bash
.venv/bin/python scripts/train_grasshopper_walk.py \
  --device cuda:0 --num-envs 4 --iterations 2 --visualizer none
```

The G1 runner default is 30,000 iterations. Choose an explicit shorter budget:

```bash
.venv/bin/python -u scripts/train_grasshopper_walk.py \
  --device cuda:0 --num-envs 4096 --iterations 5000 --visualizer none
```

Checkpoints save every 50 iterations, with TorchScript and ONNX exports every 1,000 iterations
and at the end. Checkpoint indices are zero-based (`model_4999.pt` after 5,000 iterations).
TensorBoard metrics include tracking errors, per-term rewards, episode length,
falls, air time, single/double support, sliding speed and peak swing height.

Verify effective solver parameters and randomized contact properties:

```bash
uv run --locked python scripts/check_grasshopper_foot_collisions.py \
  --training-settings --device cuda:0 --visualizer none --num-envs 16 --steps 180
```

Resume a walking checkpoint with full actor, critic and optimizer state. `--iterations` means
additional iterations. The resumed simulation counter and initial command range continue the
walking curriculum at `(saved iteration + 1) × 24`, relative to the saved command
curriculum origin (a forward-only transfer starts a new yaw curriculum):

```bash
.venv/bin/python -u scripts/train_grasshopper_walk.py \
  --resume logs/rsl_rl/grasshopper_walk_mjlab_g1_1ms/RUN/model_2499.pt \
  --device cuda:0 --num-envs 4096 --iterations 2500 --visualizer none
```

For the completed forward-only 5k policy, add 5000 iterations of yaw training
using the existing uv environment (no dependency synchronization):

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run --locked --no-sync python -u scripts/train_grasshopper_walk.py \
  --resume logs/rsl_rl/grasshopper_walk_mjlab_g1_1ms/2026-10-19_23-59-37_4096env_5000iter/model_4999.pt \
  --device cuda:0 --num-envs 4096 --iterations 5000 --visualizer none
```

This first continuation trains the ±0.5 rad/s stage; further training expands
to ±0.7 rad/s, following G1's 5000-iteration boundary.

## Acceptance after training

Use fixed forward commands of 0, 0.25, 0.5, 0.75 and 1.0 m/s with paired reset seeds, at least
20 samples per speed, 20 s per trial, and a 2 s tracking warm-up. Latch the first fall instead of
counting automatic resets as survival. Initial acceptance targets are at least 95% survival and
mean XY tracking error at most 0.2 m/s at **each** speed. Inspect lateral drift, yaw rate, peak actual
speed, joint/torque saturation, foot sliding and alternating support in video. These are evaluation
targets, not outcomes established by a short smoke run. A learned walking policy must be exported
and replayed with their 54-value observation layout (55 in legacy mode);
the existing 52-input standing deployment script is incompatible.

Evaluate yaw commands −0.5, 0 and +0.5 rad/s at forward speeds 0, 0.5 and 1 m/s.
Check each direction separately for survival, signed actual yaw rate, mean
absolute yaw error and linear tracking error after the 2 s warm-up. Include
near-zero forward speed to assess whether in-place turning generalizes; exact
in-place turns are not a separate training mode. A useful initial target is
≥95% survival and ≤0.2 rad/s yaw error per case, with alternating support and no
large foot slide. A short PPO smoke does not establish turning performance.
For nominal fixed-command replay (heading feedback and DR disabled):

```bash
uv run --locked --no-sync python scripts/replay_grasshopper_walk.py \
  --policy PATH/TO/exports/iteration_10000/policy.pt \
  --output-dir outputs/grasshopper_yaw_review \
  --speeds 0 0.5 1 --yaw-rates -0.5 0 0.5 \
  --num-envs 20 --seconds 20 --capture-states --device cuda:0 --visualizer none
```

Replay reports body-frame forward speed and signed yaw rate as well as their
tracking errors. Captured states include yaw commands/rates for RTX annotations;
older state files remain supported.

For a comparable front-facing start, add `--initial-yaw 0`. New traces also save
the complete root-link linear velocity in world/body frames and a case index.
To render commanded and measured velocities as arrows:

```bash
scripts/isaacsim-python.sh scripts/render_grasshopper_walk_replay.py \
  --states outputs/grasshopper_yaw_review/walking_states.npz \
  --output-dir outputs/grasshopper_yaw_review/render \
  --video-name command_vs_actual.mp4 --velocity-arrows
```

Cyan shows the command and orange shows the measured velocity. Straight arrows
use world-XY vectors (body-frame forward commands are rotated by the recorded
root quaternion), at the same scale of 1.25 m per m/s. Ground arcs show signed
body yaw rates with a shared 2.5 s angle scale; different radii separate the two
arcs. The thin gray trail is the measured root path and clears at each case.
Measurements are instantaneous frame samples without smoothing. The overlays
are visualization geometry only; the replay physics and recorded robot poses
are unchanged. Arrow rendering requires a new trace with complete velocities.

## Standalone native MuJoCo test

`scripts/deploy_grasshopper_walk_mujoco.py` uses only native `mujoco.mj_step`
and CPU TorchScript for its rollouts. It supports the current 54-input actor;
the older `deploy_grasshopper_mujoco.py` remains a standing-policy runner.
First export the nominal training physics model once:

```bash
uv run --locked --no-sync python scripts/replay_grasshopper_walk.py \
  --policy PATH/TO/policy.pt --output-dir outputs/native_walk/export \
  --cpu-mujoco --num-envs 1 --speeds 0.5 --seconds 3 --initial-yaw 0 \
  --export-mjcf --visualizer none
```

This creates both a geometry XML and a binary snapshot of the live compiled
parameters. The standalone runner restores those parameters exactly, including
mass/inertia, separate PD servos, contact properties and joint limits. Massless
CAD display meshes are added for rendering. Physics dt is explicitly restored
from the saved training YAML because Newton sets it immediately before stepping.
The actor's observation uses COM velocity in root-link coordinates; tracking
metrics use link-origin velocity. TorchScript includes its trained normalizer.

```bash
uv run --locked --no-sync python scripts/deploy_grasshopper_walk_mujoco.py \
  --policy PATH/TO/policy.pt --training-config PATH/TO/RUN/params/env.yaml \
  --model-export outputs/native_walk/export --output-dir outputs/native_walk/test \
  --speeds 0.5 --yaw-rates 0 0.5 -0.5 --trials 20 --seconds 20
```

The test has no Isaac Lab/Newton runtime imports, body supports, automatic
resets, observation noise, pushes or hardware randomization. It samples initial
heights within the training range and keeps heading zero for comparable views.
It reports both the training tilt limit and a stricter height/tilt screen,
tracking errors, torque-limit utilization and MuJoCo warnings. The native video
records the first 12 seconds of trial 0 for each command; cyan is commanded
velocity/yaw rate and orange is measured. `grasshopper_walk.mjb` preserves the
exact tested dynamics; the companion XML is the editable geometry description.

An analytical rotating-COM fixture checks the observation convention:

```bash
uv run --locked --no-sync python -m unittest discover -s tests \
  -p 'test_grasshopper_mujoco_observations.py' -v
```

## SuperMimic V3 world-model training

The same Grasshopper task also supports the Motion Force Adapter V3 WM+BPTT
backend, with the source policy/network/optimization design and robot-specific
state, contact and motor-delay adapters. See
[the WM training guide](../../../../mbrl/supermimic/README.md)
for preflight, training, resume and history-aware replay commands.

```bash
.venv/bin/python scripts/train_grasshopper_walk_wm.py \
  --num-envs 4096 --iterations 5000 --device cuda:0 --visualizer none
```
