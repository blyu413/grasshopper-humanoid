#!/usr/bin/env python3
"""Continue the trained Grasshopper standing policy with 20-second episodes."""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nest-stand20-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/nest-stand20-cache")

import warp as wp

wp.config.enable_backward = False
wp.config.kernel_cache_dir = "/tmp/nest-stand20-warp-cache"

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "logs/rsl_rl/grasshopper_stand/2026-09-24_18-54-51/model_4999.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=SOURCE)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--episode-length", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.iterations < 1 or args.num_envs < 1 or args.episode_length <= 0:
        parser.error("iterations, num-envs and episode-length must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.app import launch_simulation
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import resolve_task_config
    import nest.tasks  # noqa: F401

    device = args.device or "cuda:0"
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("This training run requires a CUDA GPU for the original MuJoCo Warp backend")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    source = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    source_iteration = source["iter"]
    learning_rate = source["optimizer_state_dict"]["param_groups"][0]["lr"]
    cfg, agent = resolve_task_config("Nest-Stand-Grasshopper", "rsl_rl_cfg_entry_point",
                                     overrides=["physics=newton_mjwarp"])
    cfg.scene.num_envs, cfg.seed = args.num_envs, args.seed
    cfg.episode_length_s = args.episode_length
    cfg.sim.device, cfg.sim.visualizer_cfgs = device, []
    cfg.sim.physics.solver_cfg.use_mujoco_cpu = False
    cached_ground = Path("/tmp/https") / cfg.scene.ground.spawn.usd_path.removeprefix("https://")
    if cached_ground.is_file():
        cfg.scene.ground.spawn.usd_path = str(cached_ground)
    agent.device, agent.seed, agent.max_iterations = device, args.seed, args.iterations
    agent.experiment_name = "grasshopper_stand_20s"
    agent.resume, agent.load_run, agent.load_checkpoint = True, str(args.checkpoint.parent.resolve()), args.checkpoint.name
    agent.algorithm.learning_rate = learning_rate
    agent = handle_deprecated_rsl_rl_cfg(agent, version("rsl-rl-lib"))
    cfg.validate()
    log_dir = args.log_dir or ROOT / "logs/rsl_rl" / agent.experiment_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=False)
    dump_yaml(str(log_dir / "params/env.yaml"), cfg)
    dump_yaml(str(log_dir / "params/agent.yaml"), agent)
    for relative in ("scripts/train_grasshopper_stand.py", "src/nest/assets/robots/grasshopper.py",
                     "src/nest/tasks/stand/config/grasshopper/env_cfg.py",
                     "src/nest/tasks/stand/config/grasshopper/agents/rsl_rl_ppo_cfg.py"):
        destination = log_dir / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    manifest = {
        "task": "Nest-Stand-Grasshopper", "status": "initializing",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "source_iteration": source_iteration, "additional_iterations": args.iterations,
        "completed_iterations": 0, "num_envs": args.num_envs,
        "episode_length_s": cfg.episode_length_s, "num_steps_per_env": agent.num_steps_per_env,
        "seed": args.seed, "device": device, "gpu": torch.cuda.get_device_name(device),
        "backend": "Newton/MuJoCo Warp GPU", "learning_rate_after_resume": learning_rate,
        "versions": {name: version(name) for name in ("newton", "mujoco", "mujoco-warp", "torch", "rsl-rl-lib")},
        "started_at": datetime.now().astimezone().isoformat(),
    }

    def write_manifest():
        (log_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")

    write_manifest()

    class StandingResumeRunner(OnPolicyRunner):
        def save(self, path, infos=None):
            super().save(path, infos)
            manifest["completed_iterations"] = self.current_learning_iteration - source_iteration
            manifest["latest_checkpoint"] = str(Path(path).resolve())
            manifest["updated_at"] = datetime.now().astimezone().isoformat()
            write_manifest()

    started = time.perf_counter()
    try:
        with launch_simulation(cfg, args):
            with contextlib.closing(gym.make("Nest-Stand-Grasshopper", cfg=cfg)) as env:
                wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
                runner = StandingResumeRunner(wrapped, agent.to_dict(), str(log_dir), device)
                runner.load(str(args.checkpoint), strict=True, map_location=device)
                for name, model in (("actor", runner.alg.get_policy()), ("critic", runner.alg.critic)):
                    restored = model.state_dict()
                    if not all(torch.equal(value.cpu(), source[name + "_state_dict"][key])
                               for key, value in restored.items()):
                        raise AssertionError(f"Restored {name} differs from the source checkpoint")
                    manifest[name + "_restored_exactly"] = True
                manifest["optimizer_restored"] = bool(runner.alg.optimizer.state)
                runner.current_learning_iteration = source_iteration + 1
                runner.alg.learning_rate = learning_rate
                manifest["status"] = "training"
                write_manifest()
                print(f"STANDING_TRAINING_STARTED log_dir={log_dir}", flush=True)
                runner.learn(args.iterations, init_at_random_ep_len=agent.init_at_random_ep_len)
                runner.export_policy_to_jit(str(log_dir / "exported"))
                runner.export_policy_to_onnx(str(log_dir / "exported"))
                manifest["policy"] = str((log_dir / "exported/policy.pt").resolve())
                manifest["status"] = "complete"
                manifest["elapsed_s"] = time.perf_counter() - started
                write_manifest()
                print(f"STANDING_TRAINING_COMPLETE log_dir={log_dir}", flush=True)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        write_manifest()
        raise


if __name__ == "__main__":
    main()
