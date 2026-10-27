# Environment setup

[Project overview](../README.md)

## Supported project layout

The current dependency configuration targets Linux x86-64 and Python 3.12. It uses
the following external installations:

| Component | Version / location |
| --- | --- |
| Python | 3.12, in `NEST/.venv` |
| Isaac Lab | `release/3.0.0`, commit `adc6c81c840acc03cff0340e518595431ffb19a1` |
| Isaac Lab source | `../IsaacLab-3.0.0` relative to NEST |
| Isaac Sim | 6.1.0 GA; default location `../isaac-sim-6.1.0` |
| Python packages | Exact resolution in `uv.lock`; build and index settings in `pyproject.toml` |

```text
workspace/
├── IsaacLab-3.0.0/
├── isaac-sim-6.1.0/       # needed for PhysX / Isaac Sim rendering
└── NEST/
    ├── assets/
    ├── src/
    └── .venv/            # created locally by uv
```

Isaac Lab is built from its local `tools/wheel_builder` directory. `uv.lock` pins
the Python resolution but does not freeze the contents of that external checkout.
Keep the pinned Isaac Lab commit and document any local source modifications.
Do not add the external installations or `.venv` to the NEST repository.

## Install the dependencies

Install Git, Git LFS and uv, and provision the GPU/driver environment required by
the chosen simulator. From the NEST repository root, create the sibling Isaac Lab
checkout if it is not already present:

```bash
git clone --branch release/3.0.0 https://github.com/isaac-sim/IsaacLab.git ../IsaacLab-3.0.0
git -C ../IsaacLab-3.0.0 checkout adc6c81c840acc03cff0340e518595431ffb19a1
git -C ../IsaacLab-3.0.0 lfs pull
git -C ../IsaacLab-3.0.0 rev-parse HEAD
uv sync --locked
```

For an existing Isaac Lab checkout, verify its revision and preserve any local
changes before switching commits. If it is installed elsewhere, a sibling symlink
named `IsaacLab-3.0.0` can satisfy the configured path. Changing the path in
`pyproject.toml` also requires updating `uv.lock`; moving NEST alone does not.

The NEST package is installed editable. Keep `src/` and `assets/` together in the
checkout: a standalone Python wheel does not currently bundle the robot assets.

## Newton / MuJoCo Warp

Newton workflows do not launch Isaac Sim. Check task and asset loading:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_grasshopper_foot_collisions.py' -v
.venv/bin/python scripts/train_grasshopper_walk.py --prepare-only --visualizer none
```

On a machine with a compatible CUDA GPU, run a short simulator check:

```bash
uv run --locked isaaclab zero_agent --task Nest-Stand-Grasshopper \
  --num_envs 4 --max_steps 120 physics=newton_mjwarp
```

For a fresh standing training run, use the `isaaclab train` examples in the
[workflow guide](grasshopper-workflows.md#grasshopper-standing-task). The standalone
`train_grasshopper_stand.py` script resumes an existing checkpoint by default.

## Isaac Sim / PhysX / RTX

Install Isaac Sim 6.1.0 GA separately. The launcher uses the sibling installation
by default and accepts another location through `NEST_ISAACSIM_DIR`:

```bash
export NEST_ISAACSIM_DIR=/path/to/isaac-sim-6.1.0
./scripts/isaacsim-python.sh .venv/bin/isaaclab zero_agent \
  --task Nest-Stand-Grasshopper --num_envs 4 --max_steps 120 physics=isaacsim_physx
```

Omit the environment variable when using the default sibling layout. The launcher
uses NEST's Python packages with Isaac Sim's `python.sh`. The existing project
notes record a successful one-iteration Cartpole run on an RTX 5090 with each
backend. That is historical validation, not a fresh installation test. This combined
standalone-Sim/uv setup is a project-specific integration; other Kit, RTX and ROS
workflows need their own validation.

## Local outputs and optional models

A fresh clone does not contain historical policies or training results. The training
robot's base USD, payloads and capsule override are all versioned under `assets/`.
The old generated USD copy in an existing `outputs/` directory can remain for
historical experiments, but current task configuration does not depend on it.

For continuation training and evaluation, supply the appropriate checkpoint or
policy using each script's `--checkpoint` / `--policy` option. Some historical
helpers also need a reference policy or a specific experiment directory; see the
[script guide](project-layout.md). Obtain weights separately or train a new policy.
Do not treat the historical default paths as files included in the repository.

Native MuJoCo visualization workflows may require a generated MJCF under
`outputs/grasshopper_torso/mjcf/`. Generate it when needed:

```bash
.venv/bin/python scripts/export_grasshopper_mjcf.py
```

That command also renders previews, so it needs an operational rendering backend.
See the walking guide for exporting and testing a trained walking policy in native MuJoCo.
