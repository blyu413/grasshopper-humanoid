#!/usr/bin/env python3
"""Continue Grasshopper standing PPO with mixed sustained and transient forces."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nest-push-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/nest-push-cache")

import warp as wp

wp.config.enable_backward = False
wp.config.kernel_cache_dir = "/tmp/nest-push-warp-cache"

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/grasshopper_stand_push_policies/after_2000/model_6999.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=BASE)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--export-every", type=int, default=500)
    parser.add_argument("--curriculum-start-iteration", type=int, default=0,
                        help="Local curriculum offset; useful for resuming a phase or checking the final force schedule.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.iterations < 1 or args.export_every < 1 or args.num_envs < 4 or args.curriculum_start_iteration < 0:
        parser.error("iterations and export-every must be positive; num-envs must be at least 4")
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.app import launch_simulation
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import resolve_task_config
    import nest.tasks  # noqa: F401

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg, agent = resolve_task_config("Nest-Stand-Grasshopper-Push", "rsl_rl_cfg_entry_point", overrides=[])
    cfg.scene.num_envs, cfg.seed = args.num_envs, args.seed
    cfg.events.torso_forces.params["start_step"] = args.curriculum_start_iteration * agent.num_steps_per_env
    cfg.sim.device = args.device or "cuda:0"
    cfg.sim.visualizer_cfgs = []
    cached_ground = Path("/tmp/https") / cfg.scene.ground.spawn.usd_path.removeprefix("https://")
    if cached_ground.is_file():
        cfg.scene.ground.spawn.usd_path = str(cached_ground)
    agent.device, agent.seed, agent.max_iterations = cfg.sim.device, args.seed, args.iterations
    agent = handle_deprecated_rsl_rl_cfg(agent, version("rsl-rl-lib"))
    # Ask the runner to call save each iteration; the subclass gates actual disk writes.
    agent.save_interval = 1
    cfg.validate()
    log_dir = args.log_dir or ROOT / "logs/rsl_rl" / agent.experiment_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=False)
    dump_yaml(str(log_dir / "params/env.yaml"), cfg)
    dump_yaml(str(log_dir / "params/agent.yaml"), agent)
    for relative in ("scripts/train_grasshopper_push.py", "src/nest/tasks/stand/mdp/pushes.py",
                     "src/nest/tasks/stand/mdp/mixed_pushes.py",
                     "src/nest/tasks/stand/config/grasshopper/push_env_cfg.py",
                     "src/nest/tasks/stand/config/grasshopper/env_cfg.py",
                     "src/nest/tasks/stand/config/grasshopper/agents/rsl_rl_ppo_cfg.py"):
        destination = log_dir / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    source = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    source_iteration = source["iter"]
    manifest = {
        "task": "Nest-Stand-Grasshopper-Push", "status": "initializing",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "source_iteration": source_iteration, "additional_iterations": args.iterations,
        "num_envs": args.num_envs, "seed": args.seed, "device": cfg.sim.device,
        "disturbance_type": "independent sustained and transient horizontal forces",
        "disturbance_curriculum": {
            key: value for key, value in cfg.events.torso_forces.params.items() if key != "asset_cfg"
        },
        "episode_length_s": cfg.episode_length_s,
        "num_steps_per_env": agent.num_steps_per_env,
        "curriculum_start_iteration": args.curriculum_start_iteration,
        "curriculum_reference": "ADAPT/tasks/momo_task/env_cfg.py and mdp/wrench_events.py",
        "force_frame": "world", "force_point": "TORSO center of mass",
        "learning_rate_after_resume": 1e-4, "initial_exploration_std_floor": 0.05,
        "exports": [],
    }

    def write_manifest():
        (log_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")

    write_manifest()

    class PushFineTuneRunner(OnPolicyRunner):
        def save(self, path, infos=None):
            completed = self.current_learning_iteration - source_iteration
            if completed % 250 and completed != args.iterations:
                return
            super().save(path, infos)
            manifest["completed_iterations"] = completed
            manifest["latest_checkpoint"] = str(Path(path).resolve())
            if completed % args.export_every == 0 or completed == args.iterations:
                export_dir = log_dir / f"exports/after_{completed:04d}"
                if not any(e["additional_iterations"] == completed for e in manifest["exports"]):
                    self.export_policy_to_jit(str(export_dir))
                    self.export_policy_to_onnx(str(export_dir))
                    manifest["exports"].append({"additional_iterations": completed,
                                                "checkpoint": str(Path(path).resolve()),
                                                "policy": str((export_dir / "policy.pt").resolve())})
                    print(f"PUSH_POLICY_EXPORTED iterations={completed} path={export_dir}", flush=True)
            write_manifest()

    try:
        with launch_simulation(cfg, args):
            with contextlib.closing(gym.make("Nest-Stand-Grasshopper-Push", cfg=cfg)) as env:
                wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
                runner = PushFineTuneRunner(wrapped, agent.to_dict(), str(log_dir), agent.device)
                runner.load(str(args.checkpoint), strict=True, map_location=agent.device)
                restored = runner.alg.get_policy().state_dict()
                if not all(torch.equal(v.cpu(), source["actor_state_dict"][k]) for k, v in restored.items()):
                    raise AssertionError("Restored actor differs from source checkpoint")
                manifest["actor_restored_exactly"] = True
                manifest["optimizer_restored"] = bool(runner.alg.optimizer.state)
                # Continue at the iteration after the saved one. Keep PPO moments while
                # reducing the learning rate and giving narrow action distributions room to explore.
                runner.current_learning_iteration = source_iteration + 1
                runner.alg.learning_rate = agent.algorithm.learning_rate
                for group in runner.alg.optimizer.param_groups:
                    group["lr"] = agent.algorithm.learning_rate
                with torch.no_grad():
                    runner.alg.get_policy().distribution.std_param.clamp_(min=0.05)
                manifest["status"] = "training"
                write_manifest()
                print(f"PUSH_TRAINING_STARTED log_dir={log_dir}", flush=True)
                runner.learn(args.iterations, init_at_random_ep_len=True)
                manifest["status"] = "complete"
                write_manifest()
                print(f"PUSH_TRAINING_COMPLETE log_dir={log_dir}", flush=True)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        write_manifest()
        raise


if __name__ == "__main__":
    main()
