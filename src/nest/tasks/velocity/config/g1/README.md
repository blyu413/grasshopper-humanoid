# G1 velocity rewards in NEST

`Nest-Velocity-Flat-G1-MJLab-Rewards` adapts the rewards from the vendored
MJLab `unitree_g1_flat_env_cfg` to NEST's Isaac Lab G1 flat-velocity scene.
It does not modify Isaac Lab or Motion Force Adapter. The earlier
`Nest-Velocity-Flat-G1-Foot-Height` task retains the native Isaac Lab rewards
as a control.

The 12 active MJLab terms implemented here use MJLab's weights and, where
Isaac Lab exposes equivalent state, the same equations. The height-dependent
terms use the foot-site ray casters in `nest.sensors.foot_height`. Reward
integration is per environment step in both frameworks, so the weights are
not rescaled.

Remaining differences:

- MJLab's G1 has 29 joints; the current Isaac Lab G1 has 37. The posture
  reward uses corresponding leg/arm joints and maps `torso_joint` to MJLab's
  waist-yaw tolerance. Isaac Lab's finger joints are not included. Joint
  limits still cover all Isaac Lab joints.
- MJLab's self-collision cost is omitted because the current Isaac Lab G1
  asset has self-collisions disabled. MJLab's air-time reward has weight zero
  and is also omitted.
- On Newton, `soft_landing` uses normal plus friction force. On backends that
  do not expose aggregate friction force it falls back to normal force.
  Foot contact is inferred from nonzero normal force rather than MJLab's
  geometric `found` field.
- Robot dynamics, observations, reset/termination rules, and most training
  algorithm settings remain those of the native Isaac Lab G1 task. The
  command resampling interval, standing/heading fractions, velocity ranges,
  PPO network sizes, normalization, entropy coefficient, and iteration count
  follow MJLab G1 Flat. Isaac Lab does not offer MJLab's separate
  `rel_forward_envs` command fraction.
  Matching rewards alone does not make the training runs identical.

The MJLab command curriculum is compressed to half its original duration,
using 24 environment steps per PPO iteration:

| PPO iterations | x velocity (m/s) | yaw rate (rad/s) |
| --- | --- | --- |
| 0–2,499 | -1.0 to 1.0 | -0.5 to 0.5 |
| 2,500–4,999 | -1.5 to 2.0 | -0.7 to 0.7 |
| 5,000–14,999 | -2.0 to 3.0 | -0.7 to 0.7 |

Lateral velocity stays at -1.0 to 1.0 m/s. Training is configured for
15,000 iterations with 4,096 environments and 24 steps per iteration.

CPU smoke check (two environments, 100 steps):

```bash
.venv/bin/python scripts/check_g1_foot_height.py \
  --task Nest-Velocity-Flat-G1-MJLab-Rewards --num-envs 2 \
  --steps 100 --report-every 25 --device cpu
```

Training task ID: `Nest-Velocity-Flat-G1-MJLab-Rewards`. Its RSL-RL logs use
the separate `g1_flat_mjlab_rewards` experiment name.

```bash
.venv/bin/isaaclab train --rl_library rsl_rl \
  --task Nest-Velocity-Flat-G1-MJLab-Rewards --num_envs 4096
```

The completed 15,000-iteration run is recorded with the separate
`Nest-Velocity-Flat-G1-MJLab-Rewards-Video` task. Its play-mode configuration
uses a fixed 0.6 m/s forward command, a camera following the torso, and a
500-frame, 50-fps clip. After training completes, supply its final checkpoint
explicitly to record to `outputs/g1_velocity_tracking/`:

```bash
./scripts/isaacsim-python.sh .venv/bin/isaaclab play \
  --rl_library rsl_rl \
  --task Nest-Velocity-Flat-G1-MJLab-Rewards-Video \
  --checkpoint /path/to/model_14999.pt \
  --num_envs 1 --video --video_length 500 \
  physics=newton_mjwarp
```
