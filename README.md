# NEST simulation environment

NEST provides Isaac Lab reinforcement-learning tasks for the Grasshopper robot:
standing, push recovery, flat-ground walking and yaw control. It also includes G1
reference tasks, robot assets, evaluation/rendering tools and native MuJoCo deployment scripts.

## Start here

1. Follow [Environment setup](docs/setup.md) to install the pinned external dependencies.
2. Run the asset check below, then prepare a walking experiment or start standing training.
3. Use the task guides for training, evaluation and checkpoint requirements.

Run commands from the repository root:

```bash
uv sync --locked
.venv/bin/python -m unittest discover -s tests -p 'test_grasshopper_foot_collisions.py' -v
.venv/bin/python scripts/train_grasshopper_walk.py --prepare-only --visualizer none
```

`uv sync` requires the sibling Isaac Lab checkout described in the setup guide.
The repository contains the robot assets needed to construct the training tasks;
trained policies, logs and generated videos are separate local artifacts.

## Repository contents

| Path | Contents |
| --- | --- |
| `src/nest/` | Robot configuration, Gym tasks, sensors and MDP terms; `mbrl/supermimic/` contains model-based RL |
| `scripts/` | Training, evaluation, model conversion, rendering and deployment entry points |
| `tests/` | Existing reward, observation, collision, command and push checks |
| `assets/` | Versioned robot descriptions, meshes, complete runtime USD layers and terrain |
| `docs/` | Environment setup, workflow instructions and directory/script guide |
| `pyproject.toml`, `uv.lock`, `.python-version` | Python dependencies and environment constraints |
| `logs/`, `outputs/` | Local training results and generated artifacts; ignored by Git |
| `.venv/` | Local Python environment; ignored by Git |

## Guides

- [Environment setup and external dependency versions](docs/setup.md)
- [Directory and script guide](docs/project-layout.md)
- [Robot assets and source notes](assets/README.md)
- [Grasshopper standing, push recovery and historical experiment notes](docs/grasshopper-workflows.md)
- [Grasshopper walking, domain randomization, yaw and native MuJoCo](src/nest/tasks/velocity/config/grasshopper/README.md)
- [G1 velocity tracking and MJLab-style rewards](src/nest/tasks/velocity/config/g1/README.md)

## Registered tasks

| Task | Purpose |
| --- | --- |
| `Nest-Stand-Grasshopper` | Standing from a bent-knee pose |
| `Nest-Stand-Grasshopper-Push` | Standing with sustained and transient torso forces |
| `Nest-Walk-Flat-Grasshopper` | Flat-ground walking and yaw control |
| `Nest-Velocity-Flat-G1-Foot-Height` | G1 foot-height sensor probe |
| `Nest-Velocity-Flat-G1-MJLab-Rewards` | G1 velocity tracking with adapted MJLab rewards |
| `Nest-Velocity-Flat-G1-MJLab-Rewards-Video` | G1 playback and recording |

## Code and experiment artifacts

Keep reusable source, configuration and runtime assets in the repository. Training
checkpoints, exported policies, evaluation data and recordings belong under `logs/`
or `outputs/`. Historical scripts may default to a particular local checkpoint;
consult their arguments and the workflow guide before running them on a fresh clone.

The imported robot descriptions include third-party meshes. Their upstream revision
and redistribution permissions have not been established in this checkout; see the
[asset source notes](assets/README.md#source-and-license-notes) before public distribution.
