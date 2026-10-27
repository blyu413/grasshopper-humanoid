# Grasshopper locomotion with SuperMimic V3

This ports the validated `motion-force-adapter/supermimic/v3_*` training design
to `Nest-Walk-Flat-Grasshopper`. It is a separate WM+BPTT runner; the existing
PPO runner and task defaults are preserved. No sibling-repository imports,
MoMo observer, motion dataset, or frozen motion controller are required.

## Preserved reference design

- Policy: velocity goal 3 → 2048/1024/512/256 → 32-dimensional tanh latent;
  concatenate 10 frames of proprioception, then 2048/1024/512/256 → actions.
- Policy output weights ×0.01, zero bias; separate observation/goal RunningNorm.
- Physical WM: three 2048-wide SiLU + LayerNorm layers, zero output initialization,
  local position/velocity/joint increments, SO(3) exp/log rotation integration.
- Separate 512/256/128 ELU sensor auxiliary predictor and optimizer. Its loss
  cannot train physical WM parameters; its sensor state cannot drive physics.
- Critic: 256/128 ELU with RunningNorm; privileged physics context and heading goal.
- k=8 real model-fitting windows; H=32 imagined steps; 10×2048 model minibatches;
  4096 imagination starts; gamma=.99, lambda=.95; 5 critic epochs/4 minibatches.
- Adam 2e-4, actor betas=(.9,.95), norm clipping 1 and actor value clipping .1;
  collection noise .1, clamp actions to ±20; 64 initial real steps, 500 WM-only
  iterations, 1000-step GPU FIFO, UTD=.4 (alternating 2/3 real steps per iteration).
- bf16 MLPs, float32 physical state/reward computations, torch.compile by default.
- Model and critic parameters are frozen during policy optimization, while
  gradients with respect to their inputs remain enabled.

Pinned upstream networks/rotation utilities and MIT license are under
`reference/`; see `reference/ORIGIN.md`. Unit tests compare the parameterized
policy and physical model with the pinned G1 implementation at identical shapes,
including forward values and action Jacobians.

## Robot and task adaptations

The actor receives 51 values per frame: root COM linear velocity in link axes,
angular velocity, projected gravity, 14 controlled joint positions/velocities,
and previous normalized action. Current `(vx, vy, yaw_rate)` goes only to the
velocity encoder. Both observation and action joint ordering follow the native
task. Exported policies own no history: the caller repeats the current frame on
reset and then appends one frame per 50 Hz control step.

The physical state retains all **15** joint positions/velocities, including the
passive waist mimic joint. The action space remains **14**. FK is built from the
actual compiled Newton/MuJoCo body/joint tree, with fixed links, hinge anchors,
reference joint angles and model-specific joint ordering. No G1 link/site names
or 29-joint indexing survive in the task adapter. NEST xyzw quaternions are
converted to the reference's wxyz convention at the boundary.

The auxiliary state contains two foot contacts, contact ages, force magnitudes
and swing peaks, plus two supervised landing logits. The MoMo/filter and
self-contact channels are absent because this task does not have those sensors.
The task's twelve reward terms, weights, soft limits, sole offsets and 70-degree
termination threshold are retained. Contact-dependent terms use predicted
sensors in imagination and real native sensor labels during model fitting.
The differentiable soft termination surrogate is the source V3 sigmoid in
cosine space; native evaluation keeps the task's hard threshold.

The actuator queue is explicitly part of state: five physical target frames,
current lag and resampling phase. During model fitting the four actually applied
substep lags are recorded; imagination samples the same exogenous lag process.
The applied target sequence is additional physical context, with the exact
native default/scale and encoder-bias ordering. Mass/inertia density, Kp, Kd,
dry friction, shared foot friction and encoder bias are privileged model/critic
context only. The actor does not receive ground-truth DR parameters.

Transitions and reward labels are captured after physics and before autoreset.
Model windows include the terminal transition and mask later transitions.
Post-reward interval velocity pushes break multi-step windows; they are not
mislabelled as action-driven dynamics. History reconstruction never crosses an
episode or wraps into overwritten FIFO rows. Heading updates respect NEST's
standing and straight-command overrides.

The curriculum uses the source V3 PPO-equivalent clock: model warmup holds the
initial stage, and policy updates advance toward `iterations × 24` control steps.
NEST's existing velocity ranges and 5000-iteration yaw expansion boundary apply.
The first run fixes and saves that virtual-clock rate; extending a resumed run
does not rewind the command curriculum. As in the source, `--iterations` is the
total endpoint, not additional iterations.

## Commands

Prepare manifests, environment YAML, source snapshots and the initial actor:

```bash
.venv/bin/python scripts/train_grasshopper_walk_wm.py --prepare-only --visualizer none
```

CPU integration preflight uses the **same full-sized networks and H=32**, but
one native MuJoCo environment and smaller fitting/imagination batches. As with
NEST's existing CPU smoke, the backend cannot export contact sensors, so the
three contact rewards are omitted. This does not validate full-task contacts:

```bash
.venv/bin/python scripts/train_grasshopper_walk_wm.py --cpu-smoke --visualizer none
```

Production-size GPU preflight includes all twelve rewards, bf16 and compilation.
It collects 64 steps and performs one WM fit (10 minibatches), one policy/critic
update, checkpoint roundtrip and TorchScript export check. Preflight checkpoints
are not trained walking policies:

```bash
.venv/bin/python scripts/train_grasshopper_walk_wm.py \
  --validate-only --num-envs 4096 --device cuda:0 --visualizer none
```

Train from scratch using the source defaults and current Grasshopper task:

```bash
.venv/bin/python -u scripts/train_grasshopper_walk_wm.py \
  --num-envs 4096 --iterations 5000 --device cuda:0 --visualizer none
```

`--no-compile` and `--amp none` are explicit diagnostic options. Every invocation
requires a fresh output directory; default output is `logs/supermimic/grasshopper_walk/`.
Run metadata records actual model/actor dimensions, robot asset hashes, context
schema and copied source hashes. `metrics.jsonl` separates collection, model and
policy/critic times and reports real transition counts.

Resume in a new output directory:

```bash
.venv/bin/python scripts/train_grasshopper_walk_wm.py \
  --resume PATH/TO/model_005000.pt --iterations 10000 \
  --device cuda:0 --visualizer none
```

All networks, normalizers, optimizers and progress are restored. The simulator,
startup DR, actor history and FIFO are freshly initialized, followed by real-data
warmup, matching the source's optimizer-resume semantics. This is not a bitwise
simulation continuation. PPO and G1/MoMo checkpoints are deliberately rejected.

The final actor is `exports/policy.pt`, accepting `(B,10,51)` history and `(B,3)`
goal; normalization is embedded. Keep `exports/policy.json` alongside it. The
existing replay supports this schema automatically, including state capture for
RTX video rendering:

```bash
.venv/bin/python scripts/replay_grasshopper_walk.py \
  --policy PATH/TO/exports/policy.pt --output-dir outputs/wm_walk_review \
  --speeds 0 0.25 0.5 0.75 1 --yaw-rates -0.5 0 0.5 \
  --num-envs 20 --seconds 20 --capture-states --device cuda:0 --visualizer none
```

Portable CPU replay uses `--cpu-mujoco --num-envs 1`. Native standalone deployment
via `deploy_grasshopper_walk_mujoco.py` still expects the legacy one-input PPO
export; use the history-aware replay above for WM exports.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_supermimic.py' -v
```

Tests cover pinned-network/Jacobian equivalence, isolated auxiliary gradients,
frozen-model input gradients, delay buffer behavior through resampling/resets,
FK versus native MuJoCo (including nonzero hinge anchors and a passive mimic
joint), FIFO terminal/push boundaries, history reset and lambda-return semantics.
Preflight additionally checks real NEST observations and individual rewards,
executed delay states, updates in both actor branches, frozen WM parameters,
checkpoint restoration and portable export values.

## Checked against the previous walking policy (2026-10-27)

Baseline: the completed `2026-10-22_03-30-18_4096env_yaw_resume_4950iter`
run, ending at 10,000 cumulative PPO iterations. The entire saved environment
YAML was compared with the 4096-environment WM preflight YAML. The only serialized
difference is `curriculum.command_vel.params.start_step`: the previous resumed
yaw run used 120,000 to exclude its first 5,000 forward-only iterations; the new
WM run starts at the default zero. Commands, rewards, reset ranges, observation
noise, DR, pushes, PD/scales/limits, contacts, timestep, solver, episode length,
number of environments and seed otherwise match.

The URDF and the native task/reward/action source files are byte-identical to
that run's snapshots. The capsule-USD hash differs only because its referenced
USD moved from `outputs/` into `assets/`; all nine referenced USD-package files
are byte-identical to the old package.

Algorithm differences are intentional: the old actor is a normalized 54-input
512/256/128 MLP; V3 uses ten 51-value frames plus goal3 and the reference latent
encoder/decoder. The critic input/network changes to reference V3 physical
features, context and motor memory. Old PPO uses 24 real steps/iteration,
learned Gaussian exploration initially std=1, adaptive LR=1e-3 and entropy=.01.
V3 uses the model/BPTT defaults above, fixed collection noise=.1 and explicit
±20 action clipping. Gamma=.99 and lambda=.95 are shared. Existing PPO randomly
initializes episode counters; V3 starts native reset counters at zero, as in
Motion Force Adapter. The new WM run starts from scratch rather than loading the
previous PPO checkpoint. These changes mean identical iteration counts do not
represent identical sample or compute budgets.

Validation completed: eight focused unit tests, 22 existing walking/yaw/reward
regressions, native CPU smoke, GPU preflight at 64 and 4096 environments (the
latter uses default bf16/compile/batches/H32), portable history-aware replay,
and a short training-loop/optimizer-resume smoke. The 4096-environment preflight
checks all twelve rewards: maximum clean-observation error 9.54e-7, maximum
weighted per-step reward error 2.79e-4, exact motor-delay state agreement,
zero checkpoint roundtrip error and export error 1.89e-6. Both policy branches
update while WM parameters remain unchanged. The original solver emits
iteration-limit diagnostics for some untrained trajectories; solver settings
were retained from the walking task.

Local evidence: `outputs/supermimic_v3_preflight_20261027/validation.json` and
`walking_config_comparison.json` in that directory. The validation and short
smoke checkpoints do not establish locomotion quality; no full 5,000-iteration
training run was launched as part of this migration.

## Small Actor profile (2026-10-27)

`--policy-profile small_mlp` selects the earlier ADAPT V2 actor widths
512/256/128 (ELU), with five frames and no command encoder/latent.
NEST input packing is five complete 51-value proprioception/action frames plus
the current 3-value velocity goal (258 inputs, 14 outputs); it deliberately
retains the current NEST observation semantics rather than V2's term-wise
packing with historical goals and only the newest action. The critic stays
228 -> 256 -> 128 -> 1, matching V2 critic hidden widths. Normalizers and
privileged critic inputs remain those of this NEST implementation.
The small actor uses standard MLP initialization (no V3 output scaling).

Physical WM 2048/2048/2048, sensor auxiliary 512/256/128, H32, model-fitting
batches, 500 model-only iterations, rewards, DR and robot are unchanged.
This is a small-policy profile of the NEST V3 dynamics/BPTT implementation,
not a rollback to V2's mixed physical/sensor world model.

Checkpoints use `nest_grasshopper_small_mlp_bptt`, and reference V3 checkpoints
remain separate and cannot be silently resumed into the small architecture.
Exports declare `[5, 51]` history; `replay_grasshopper_walk.py` supports both.

```bash
.venv/bin/python -u scripts/train_grasshopper_walk_wm.py \
  --policy-profile small_mlp --num-envs 4096 --iterations 5000 \
  --device cuda:0 --visualizer none
```

The small profile passes the full 4096-environment preflight. Independent
FK/reward and CPU/GPU export comparisons disable TF32 only during comparison:
chained 3x3 products otherwise accumulate TF32 rounding (observed angular
momentum reward difference 3.27e-4 versus 9.18e-6 with strict FP32). Actual
training updates retain their configured high-matmul/bf16/compile settings.
Evidence: `outputs/supermimic_small_preflight_fp32_20261027/validation.json`.
