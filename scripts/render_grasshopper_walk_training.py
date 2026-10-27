#!/usr/bin/env python3
"""Run brief PhysX PPO training and record states for the independent RTX renderer."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import time

# Load the project's Warp 1.17 before Kit can select its bundled Warp 1.16.
# Keep the training environment and Kit on the same Warp installation.
import warp as wp

wp.config.enable_backward = False
wp.config.kernel_cache_dir = "/tmp/nest-walk-isaac-warp-cache"

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Replay an existing 55-input walking checkpoint")
    args = parser.parse_args()
    if args.num_envs < 1 or args.iterations < 1:
        parser.error("num-envs and iterations must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "status": "starting", "task": "Nest-Walk-Flat-Grasshopper",
        "started_at": datetime.now().astimezone().isoformat(),
        "physics": "Isaac Sim PhysX", "renderer": "Isaac Sim RTX RayTracedLighting",
        "num_envs": args.num_envs, "requested_iterations": args.iterations,
        "completed_iterations": 0, "seed": 42, "states": [],
        "note": "Brief training visualization; not evidence of a converged walking gait.",
    }

    def save_report():
        (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")

    save_report()
    from isaaclab.app import AppLauncher

    # AppLauncher disables Isaac Sim's default physics callbacks so Isaac Lab
    # owns the GPU physics lifecycle and tensor-view device consistently.
    launcher = AppLauncher({"headless": True, "enable_cameras": True,
                            "device": "cuda:0", "visualizer": "none"})
    app = launcher.app
    env = None
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from isaaclab_tasks.utils import resolve_task_config
        from isaaclab.utils.io import dump_yaml
        from isaaclab.utils.warp import ProxyArray
        import nest.tasks  # noqa: F401
        from nest.tasks.velocity.warm_start import initialize_from_standing

        report["warp_version"] = wp.__version__
        original_launch = wp.launch

        def launch_proxy_arrays(*launch_args, **launch_kwargs):
            """Adapt this checkout's PhysX ProxyArray arguments to native Warp."""
            packed = list(launch_args)
            for index, key in ((2, "inputs"), (3, "outputs")):
                if key in launch_kwargs and launch_kwargs[key] is not None:
                    launch_kwargs[key] = [v.warp if isinstance(v, ProxyArray) else v
                                          for v in launch_kwargs[key]]
                elif len(packed) > index and packed[index] is not None:
                    packed[index] = [v.warp if isinstance(v, ProxyArray) else v for v in packed[index]]
            return original_launch(*packed, **launch_kwargs)

        wp.launch = launch_proxy_arrays
        report["runtime_compatibility"] = "Explicit ProxyArray.warp views at the Warp launch boundary"
        torch.set_num_threads(4)
        torch.manual_seed(42)
        cfg, agent = resolve_task_config(report["task"], "rsl_rl_cfg_entry_point",
                                        overrides=["physics=isaacsim_physx"])
        cfg.seed = 42
        cfg.scene.num_envs = args.num_envs
        cfg.scene.env_spacing = 2.2
        cfg.sim.device = "cuda:0"
        cfg.sim.use_fabric = False
        cfg.sim.render_interval = 1000000
        cfg.sim.visualizer_cfgs = []
        # PhysX merges the fixed foot links into their ankle parent bodies. The
        # Newton sensor paths therefore cannot be reused for this brief demo.
        cfg.scene.feet_contact = None
        cfg.scene.robot.spawn.activate_contact_sensors = False
        cfg.rewards.foot_clearance = None
        cfg.rewards.foot_swing_height = None
        cfg.rewards.foot_slip = None
        cfg.rewards.soft_landing = None
        report["omitted_rewards"] = ["foot_clearance", "foot_swing_height", "foot_slip", "soft_landing"]
        cfg.scene.ground.spawn.visual_material.diffuse_color = (0.42, 0.46, 0.52)
        cfg.scene.dome_light.spawn.intensity = 600.0
        cfg.scene.dome_light.spawn.color = (0.93, 0.96, 1.0)
        agent.device, agent.seed, agent.max_iterations = "cuda:0", 42, args.iterations
        agent.save_interval = max(1, args.iterations)
        agent = handle_deprecated_rsl_rl_cfg(agent, version("rsl-rl-lib"))
        dump_yaml(str(output / "params/env.yaml"), cfg)
        dump_yaml(str(output / "params/agent.yaml"), agent)
        shutil.copy2(__file__, output / Path(__file__).name)

        print("[NEST] Creating real Isaac Sim PhysX environments", flush=True)
        env = gym.make(report["task"], cfg=cfg)
        raw = env.unwrapped
        raw.render_enabled = False
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = OnPolicyRunner(wrapped, agent.to_dict(), str(output / "training"), "cuda:0")
        if args.checkpoint:
            runner.load(str(args.checkpoint), map_location="cuda:0")
            report["replay_checkpoint"] = str(args.checkpoint.resolve())
            training_report = args.checkpoint.resolve().parent.parent / "run.json"
            if training_report.is_file():
                previous = json.loads(training_report.read_text())
                for key in ("source_checkpoint", "source_sha256", "num_steps_per_env",
                            "completed_iterations", "training_samples", "actor_max_parameter_change"):
                    if key in previous:
                        report[key] = previous[key]
                report["training_report"] = str(training_report)
        else:
            source = ROOT / "logs/rsl_rl/grasshopper_stand/2026-09-24_18-54-51/model_4999.pt"
            saved = torch.load(source, weights_only=True, map_location="cpu")
            initialize_from_standing(runner.alg.get_policy(), saved["actor_state_dict"])
            initial = {k: v.detach().cpu().clone() for k, v in runner.alg.get_policy().state_dict().items()}
            report.update(status="training", source_checkpoint=str(source),
                          source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          num_steps_per_env=agent.num_steps_per_env)
            save_report()
            print(f"[NEST] Training {args.iterations} PPO iterations", flush=True)
            runner.learn(args.iterations, init_at_random_ep_len=True)
            final = runner.alg.get_policy().state_dict()
            report["actor_max_parameter_change"] = max(
                float((v.detach().cpu() - initial[k]).abs().max()) for k, v in final.items())
            report["completed_iterations"] = args.iterations
            report["training_samples"] = args.num_envs * agent.num_steps_per_env * args.iterations
            runner.export_policy_to_jit(str(output / "exported"))
        report["status"] = "recording"
        report["setup_and_training_elapsed_seconds"] = round(time.monotonic() - started, 2)
        save_report()
        policy = runner.get_inference_policy(device="cuda:0")
        obs, _ = wrapped.reset()
        resets = 0
        # Real exploration actions produce different learning states in each environment.
        report["capture_actions"] = "trained PPO actor with its learned exploration distribution"
        for step in range(1, 73):
            with torch.inference_mode():
                actions = policy(obs, stochastic_output=True)
                obs, rewards, dones, _ = wrapped.step(actions)
            resets += int(dones.sum())
            if step not in (24, 48, 72):
                continue
            if not torch.isfinite(raw.scene["robot"].data.root_pos_w.torch).all():
                raise RuntimeError("Nonfinite robot state")
            state_path = output / f"state_{step:03d}.npz"
            robot = raw.scene["robot"]
            np.savez_compressed(state_path, root_pos_w=robot.data.root_pos_w.torch.cpu().numpy(),
                                root_quat_w=robot.data.root_quat_w.torch.cpu().numpy(),
                                joint_pos=robot.data.joint_pos.torch.cpu().numpy(),
                                joint_names=robot.joint_names)
            report["states"].append({"path": str(state_path), "control_step": step,
                                     "quaternion_order": "xyzw"})
            save_report()
            print(f"[NEST] PHYSX_STATE_SAVED {state_path}", flush=True)
        report.update(status="complete", replay_resets=resets,
                      elapsed_seconds=round(time.monotonic() - started, 2))
        save_report()
        print(f"[NEST] COMPLETE {output}", flush=True)
    except BaseException as error:
        import traceback
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        save_report()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
