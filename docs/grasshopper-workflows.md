# Grasshopper workflows and experiment notes

[Project overview](../README.md) · [Environment setup](setup.md) · [Asset inventory](../assets/README.md)

Run all commands from the repository root. Sections naming dated runs or checkpoints
describe historical experiments; those weights are not included in a fresh clone.
For a fresh standing run, use the `isaaclab train` commands below. Walking training
starts from scratch unless a checkpoint is explicitly supplied.

## Grasshopper torque limits

The Grasshopper URDF is the single source of truth for output-joint torque limits. Its 15 revolute joints currently
use limits from 135 to 393.06 N·m. Newton/MJWarp does not preserve the imported USD drive `maxForce` values at
runtime, so loading the USD without an actuator configuration can incorrectly produce a 1,000,000 N·m limit.

Use `make_grasshopper_cfg()` from
`src/nest/assets/robots/grasshopper.py` when creating a torque-controlled scene. The old
`assets/robots/grasshopper_torso/grasshopper_asset_cfg.py` path remains as a compatibility import. The asset
configuration parses the URDF and explicitly supplies the same limits to the actuator model and physics solver for
both backends. The
`POLICY_JOINT_NAMES` tuple contains the 14 independent policy actions; `WAIST_P_RIGHT_out` is a mimic joint and must
not be controlled independently.

Verify the runtime values with:

```bash
uv run --locked python scripts/check_grasshopper_torque_limits.py \
  --physics newton_mjwarp --headless

./scripts/isaacsim-python.sh scripts/check_grasshopper_torque_limits.py \
  --physics physx --headless
```

## Grasshopper foot collisions

The shared Grasshopper model uses the MJLab G1 layout: seven longitudinal capsules per
foot, each with a 10 mm radius. Their lengths and positions fit Grasshopper's 258 × 86 mm
sole footprint and mirrored foot frames. The collision bottom remains at local Z = -0.0773 m.
The capsule asset preserves visual meshes, rigid-body mass/inertia and joints.
Its default friction is 0.6; MuJoCo foot priority 1, condim 3, zero margin/gap and
solref (0.02, 1) follow G1. Walking settings are described separately below.

The URDF defines the capsules using Newton's `<capsule radius="..." height="..."/>`
extension; height is the straight section between the hemispherical caps. The default
`assets/robots/grasshopper_torso/grasshopper_lower_body_capsule_feet.usda` references the
existing USD and disables its foot mesh colliders while adding native USD capsules.
Both the MuJoCo and PhysX variants use the new contacts. The old mesh asset remains
available as `MESH_FEET_USD_PATH` for historical comparisons.

Rebuild after editing the URDF foot geometry, then verify compiled MJWarp shapes and sensors:

```bash
uv run --locked python scripts/build_grasshopper_capsule_feet.py
uv run --locked python scripts/check_grasshopper_foot_collisions.py \
  --device cuda:0 --visualizer none --num-envs 16 --steps 180
```

Native MuJoCo exports are regenerated with `scripts/export_grasshopper_mjcf.py`.
Policies trained with the old mesh contacts retain their weights; this geometry change
alone has not established improved walking behavior.

## Grasshopper standing task

`Nest-Stand-Grasshopper` is a free-base, manager-based single-agent task following Isaac Lab's external-project and
`hold_pose` example structure. It uses the existing Isaac Lab MDP functions directly; there is no custom MDP package.

| Path | Role |
| --- | --- |
| `src/nest/assets/robots/grasshopper.py` | Robot configuration, URDF-derived joint limits, standing PD variant |
| `src/nest/tasks/stand/config/grasshopper/__init__.py` | Gym task registration |
| `src/nest/tasks/stand/config/grasshopper/env_cfg.py` | Physics presets, scene, MDP and environment settings |
| `src/nest/tasks/stand/config/grasshopper/agents/rsl_rl_ppo_cfg.py` | RSL-RL PPO configuration |

The policy observes 52 proprioceptive values and commands 14 independent joint-position offsets (0.25 rad per unit
action) around the bent-knee standing pose. The driven joints use implicit PD; the mimic joint is passive. The knee
pitch reference is approximately 0.4 rad (23 degrees), with mirrored 0.2 rad hip and ankle offsets to keep the feet
level. The root starts at 0.964 m, placing the soles near the ground. Episodes last at most 20 s and end early if the root falls below
0.7 m or tilts beyond 0.8 rad. Rewards and terminations use the built-in Isaac Lab MDP terms.

After `uv sync --locked`, check either backend without training:

```bash
uv run --locked isaaclab zero_agent --task Nest-Stand-Grasshopper \
  --num_envs 4 --max_steps 120 physics=newton_mjwarp

./scripts/isaacsim-python.sh .venv/bin/isaaclab zero_agent \
  --task Nest-Stand-Grasshopper --num_envs 4 --max_steps 120 physics=isaacsim_physx
```

Start PPO training with either backend:

```bash
uv run --locked isaaclab train --rl_library rsl_rl \
  --task Nest-Stand-Grasshopper --num_envs 4096 --max_iterations 5000 physics=newton_mjwarp

./scripts/isaacsim-python.sh .venv/bin/isaaclab train --rl_library rsl_rl \
  --task Nest-Stand-Grasshopper --num_envs 4096 --max_iterations 5000 physics=isaacsim_physx
```

Training logs are written under `logs/rsl_rl/grasshopper_stand/`. The local run
`2026-09-24_18-54-51` completed 5000 iterations with 4096 environments on Newton/MJWarp.
That historical run used 8 s episodes and recorded approximately 19.1 minutes between the first and last
training iterations. The current standing configuration uses 20 s episodes; with the same 24 steps per
environment per iteration, 4096 environments and 5000 iterations, the total sample count remains 491,520,000.
It contains `model_4999.pt` and the exported `exported/policy.pt`, `policy.onnx`, and
`policy.onnx.data`. The exported TorchScript weights match the final checkpoint.
These files are local artifacts in the Git-ignored `logs/` directory.
The asset path is resolved relative to the NEST checkout, so
the editable installation expects the repository's `assets/` directory to remain in place.
The base USD and all payloads are versioned under `assets/robots/grasshopper_torso/usd/`;
loading the training robot does not require `outputs/`. Historical checkpoints and
MuJoCo exports referenced below are local artifacts and must be trained, exported,
or obtained separately.

Continue the original standing checkpoint with 20 s episodes on MuJoCo Warp:

```bash
.venv/bin/python -u scripts/train_grasshopper_stand.py \
  --device cuda:0 --visualizer none --num-envs 4096 --iterations 5000
```

The script restores the actor, critic, optimizer and final learning rate from `model_4999.pt`,
then runs 5000 additional iterations. It keeps the standing task's existing rewards and reset ranges.
New checkpoints, TensorBoard logs, configuration snapshots and exported policies are saved separately
under `logs/rsl_rl/grasshopper_stand_20s/`. A `run.json` records the source checkpoint and progress.

## Grasshopper walking and yaw control (1 m/s)

`Nest-Walk-Flat-Grasshopper` uses MJLab G1 Flat rewards, 50 Hz control, noisy
54-value actor observations, a clean 66-value critic, normalized [512, 256, 128]
networks, std 1.0 exploration, velocity pushes and shared randomized foot friction.
The hardware DR profile additionally varies coherent body mass/inertia (±10%),
PD gains (±10%), dry joint friction (0–0.5 N·m) and motor command delay (0–20 ms).
Forward commands span 0–1 m/s from the start. Yaw commands start at ±0.5 rad/s,
expanding to ±0.7 after 5000 yaw-training iterations. MJLab's command mixture
uses 30% heading, 20% straight-walking and 10% standing masks, with standing and
straight commands taking precedence. Lateral commands remain zero.
The robot retains its nominal PD
settings and URDF limits; joint action scales follow G1's effort/stiffness rule.
Nominal playback disables these randomizations. Existing v2 5k checkpoints can
resume with the new DR and yaw profiles while keeping the actor/critic layouts.
Forward-only policies start a new yaw curriculum; the previously unused yaw
input is initialized without changing the zero-yaw network output.
Training starts from scratch; `--init-checkpoint` explicitly selects the historical
standing actor's compatible observation/action profile. See
[`src/nest/tasks/velocity/config/grasshopper/README.md`](../src/nest/tasks/velocity/config/grasshopper/README.md)
for preparation, CPU/GPU smoke checks, training, resume and evaluation criteria.

```bash
.venv/bin/python scripts/train_grasshopper_walk.py --prepare-only --visualizer none
```

## Grasshopper external-force evaluation

The standing training task randomizes the initial root pose but does not apply pushes during an episode.
`scripts/test_grasshopper_push.py` tests the existing policy with actual force pulses at the `TORSO` center of mass.
The default pulse lasts 0.2 s, starts at 2 s, and acts along each world horizontal axis. Each trial lasts 8 s;
the original randomized resets are retained, with eight paired reset samples per case. Automatic resets are
disabled during evaluation and failures are latched using the task's original 0.7 m height and 0.8 rad tilt limits.

Run with the original Newton/MJWarp solver on a CUDA machine:

```bash
.venv/bin/python scripts/test_grasshopper_push.py --device cuda:0 --visualizer none \
  --forces 0 50 100 200 400 600 800 --output-dir outputs/grasshopper_push_test/gpu
```

For CPU evaluation with Newton's native MuJoCo solver:

```bash
.venv/bin/python scripts/test_grasshopper_push.py --mujoco-cpu --visualizer none \
  --forces 0 50 100 200 400 600 800 --output-dir outputs/grasshopper_push_test/sweep
.venv/bin/python scripts/plot_grasshopper_push.py outputs/grasshopper_push_test/sweep/results.json
```

CPU samples run serially. The test's CPU manager skips Newton contact-report export, which the native MuJoCo
solver does not implement; MuJoCo still solves ground/body contacts. This task has no contact sensors.
CPU results should be treated as an evaluation in a different solver from the original GPU training run.

Results include per-trial survival, maximum tilt, peak speed, displacement, and recovery time in JSON/CSV.
Recovery requires tilt below 5 degrees, horizontal speed below 0.1 m/s, angular speed below 0.2 rad/s, and
height within 3 cm of the pre-push height continuously for 0.5 s. `recovered_at_end` additionally requires
the trial to finish in that state without any earlier fall. Add `--video --num-envs 1 --mujoco-cpu` to record
each case with a force arrow and state labels using MuJoCo's rendering of the physics geometry.

## Mixed-force continuation training

`Nest-Stand-Grasshopper-Push` uses two independent torso-force channels inspired by
`../ADAPT/tasks/momo_task/env_cfg.py` and `mdp/wrench_events.py`: a sustained force and a transient
force can overlap, and their vectors are summed before application. Each channel holds its sampled
world horizontal direction and magnitude for the full active duration, then samples a separate rest
period. The combined magnitude is capped at 220 N. Episode reset clears both channels and selects
20% of episodes to remain force-free for the entire episode.

The observations, actions, robot settings, rewards, and fall limits retain the standing task's settings.
Training episodes last 20 s so 5–7 s loads can finish and leave recovery time. A 1 s settling window
is added to the initial channel rest periods. Sustained-force rest periods are always 2–4 s.

| Additional PPO iterations | Sustained force | Sustained duration | Transient force | Transient duration | Transient rest |
| --- | --- | --- | --- | --- | --- |
| 0–249 | 5–20 N | 1–2 s | Disabled | — | — |
| 250–749 | 10–50 N | 2–4 s | 40–120 N | 0.1–0.3 s | 4–6 s |
| 750–1249 | 10–80 N | 4–6 s | 40–180 N | 0.1–0.35 s | 3–5 s |
| 1250 onward | 10–100 N | 5–7 s | 40–200 N | 0.1–0.4 s | 2.5–4.5 s |

Stage boundaries use local continuation control steps (24 per PPO iteration). A stage transition
changes parameters for future force samples and leaves already-active vectors and durations intact.
The previous short-pulse event remains available as `RandomTorsoPulses` for historical experiments.

Continue from the existing trained actor, critic, and optimizer on Newton/MJWarp:

```bash
.venv/bin/python scripts/train_grasshopper_push.py --device cuda:0 --visualizer none \
  --num-envs 4096 --iterations 2000
```

The script starts from `outputs/grasshopper_stand_push_policies/after_2000/model_6999.pt`,
verifies that the restored actor exactly matches that checkpoint,
then uses a 0.0001 learning rate, PPO clipping of 0.1, and an initial action-noise standard-deviation
floor of 0.05. It saves in the separate `logs/rsl_rl/grasshopper_stand_push_mixed/` experiment. Each run
includes its source checkpoint hash, configuration/source snapshots, training status in `run.json`,
checkpoints every 250 additional iterations, and TorchScript/ONNX exports every 500 iterations.
`--checkpoint`, `--iterations`, and `--export-every` can select another starting checkpoint or schedule.
`--curriculum-start-iteration` selects a local curriculum offset for phase continuation or GPU smoke checks.

For the mixed-force run, wait until the entire training run completes before evaluating the source policy
and all four continuation stages:

```bash
.venv/bin/python scripts/evaluate_mixed_grasshopper_run.py \
  logs/rsl_rl/grasshopper_stand_push_mixed/RUN
```

This evaluates three profiles using 16 paired reset samples per case:

- Short forces: three 0.2 s pushes at 2, 4, and 6 s, using 100/150/200 N.
- Sustained forces: one 5 s push at 2 s, using 40/80/100/150/200 N, then 5 s of recovery.
- Combined forces: a 5 s force of 40/80/100 N plus an additional 120 N, 0.2 s pulse at 4 s in the same direction.

The source and continuation policies use the same solver, force schedules, and reset seeds.
Results, weights, checkpoints, comparison plots, and a recommendation are packaged under
`outputs/grasshopper_stand_mixed_policies/`. Model selection first preserves clean standing, then
prioritizes recovery under sustained and combined loads up to 100 N, before short-force and stronger
sustained stress results. The earlier short-force-only run and its artifacts remain in their original folders.

Evaluate an export on the original task to measure learned robustness without training-time random pushes:

```bash
.venv/bin/python scripts/test_grasshopper_push.py --device cuda:0 --visualizer none \
  --policy logs/rsl_rl/grasshopper_stand_push_mixed/RUN/exports/after_0500/policy.pt \
  --num-envs 16 --forces 0 50 100 150 200 --output-dir outputs/grasshopper_push_test/gpu_after
```

Add `--repeat-pushes 3` to apply the same test push at 2, 4, and 6 seconds. Recovery time is then
measured after the last pulse, and any fall during the full trial remains a failure. Before/after
comparisons should use the same solver, pulse schedule, sample count, and reset seed.

To wait until the entire training run has finished, then evaluate and package all four exports:

```bash
.venv/bin/python scripts/evaluate_grasshopper_push_run.py \
  logs/rsl_rl/grasshopper_stand_push/RUN --after-training
```

The evaluator reuses any completed evaluation with the same policy hash and protocol. It writes the four
TorchScript/ONNX policies, continuation checkpoints, evaluation data, comparison plot, and recommendation
to `outputs/grasshopper_stand_push_policies/`. No new stage evaluations run before training is complete
when `--after-training` is selected.

Render the final policy's actual robot meshes with Isaac Sim RTX while retaining Newton/MJWarp physics:

```bash
./scripts/isaacsim-python.sh -u scripts/test_grasshopper_push.py --device cuda:0 \
  --num-envs 16 --policy outputs/grasshopper_stand_push_policies/after_2000/policy.pt \
  --forces 0 200 --directions +X +Y --repeat-pushes 3 --video --video-renderer isaac \
  --video-sample 1 --video-label 'Push tuned 2000 | Isaac Sim RTX' \
  --output-dir outputs/grasshopper_stand_push_policies/isaac_render
```

This records 1280×720, 30 fps clips through Isaac Sim's Kit camera, with a red force arrow and state labels.
The camera shows one reset sample; the other environments remain simulated but are hidden. Renderer warm-up
does not advance physics, and Kit's independent physics stepping is disabled during frame capture.
Headless frame capture explicitly calls `sim.render()` to synchronize current Newton body poses into Fabric
before reading the RTX image.

For a sustained-force test, use one 5-second push at 2 s and observe recovery until 12 s:

```bash
./scripts/isaacsim-python.sh -u scripts/test_grasshopper_push.py --device cuda:0 \
  --num-envs 16 --policy outputs/grasshopper_stand_push_policies/after_2000/policy.pt \
  --forces 0 200 --directions +X +Y --push-duration 5 --repeat-pushes 1 --seconds 12 \
  --video --video-renderer isaac --video-sample 1 --camera-wide \
  --video-label 'Final policy | 5-second push' \
  --output-dir outputs/grasshopper_stand_push_policies/isaac_render_5s
```
