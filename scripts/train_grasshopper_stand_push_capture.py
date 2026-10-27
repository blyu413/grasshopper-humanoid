#!/usr/bin/env python3
"""Train asynchronous standing disturbances and render an exact PPO milestone."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nest-stand-capture-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/nest-stand-capture-cache")
import warp as wp

wp.config.enable_backward = False
wp.config.kernel_cache_dir = "/tmp/nest-stand-capture-warp"

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
TASK = "Nest-Stand-Grasshopper-Push"


def force_stages(rollout_steps):
    """Independent persistent loads and short impulses, ramped over this run."""
    stages = []
    for iteration, sustained, pulse in (
        (0, (20.0, 80.0), (60.0, 150.0)),
        (300, (25.0, 100.0), (70.0, 180.0)),
        (700, (30.0, 120.0), (80.0, 200.0)),
    ):
        stages.append({
            "step": iteration * rollout_steps,
            "sustained": {"enabled": True, "force_range": sustained,
                          "duration_s": (1.2, 4.0), "idle_s": (0.4, 2.0)},
            "pulse": {"enabled": True, "force_range": pulse,
                      "duration_s": (0.12, 0.35), "idle_s": (1.0, 3.0)},
        })
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--capture-iteration", type=int, default=200)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "logs/rsl_rl/grasshopper_stand/2026-09-24_18-54-51/model_4999.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-render", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 1 <= args.capture_iteration <= args.iterations or args.num_envs < 4:
        parser.error("capture iteration must fall within training; use at least four environments")

    import gymnasium as gym
    import numpy as np
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab.app import launch_simulation
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import resolve_task_config
    import nest.tasks  # noqa: F401

    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg, agent = resolve_task_config(TASK, "rsl_rl_cfg_entry_point", overrides=["physics=newton_mjwarp"])
    cfg.seed = 42
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 2.5
    cfg.sim.device = args.device or "cuda:0"
    cfg.sim.visualizer_cfgs = []
    cfg.scene.ground = AssetBaseCfg(
        prim_path="/World/ground",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
        spawn=sim_utils.CuboidCfg(size=(200.0, 200.0, 0.1),
                                 collision_props=sim_utils.CollisionPropertiesCfg(),
                                 physics_material=cfg.sim.physics_material,
                                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.38, 0.42))),
    )
    disturbance = cfg.events.torso_forces.params
    disturbance["stages"] = force_stages(agent.num_steps_per_env)
    disturbance["clean_episode_probability"] = 0.1
    disturbance["settle_s"] = 0.5
    disturbance["max_combined_force"] = 220.0
    disturbance["start_step"] = 0
    # Independent reset perturbations supplement, rather than replace, the forces.
    cfg.events.reset_base.params["pose_range"].update(
        roll=(-0.05, 0.05), pitch=(-0.05, 0.05), yaw=(-0.25, 0.25))
    agent.device, agent.seed, agent.max_iterations = cfg.sim.device, 42, args.iterations
    agent.save_interval = 1
    agent.algorithm.learning_rate = 1e-4
    agent = handle_deprecated_rsl_rl_cfg(agent, version("rsl-rl-lib"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dump_yaml(str(output / "params/env.yaml"), cfg)
    dump_yaml(str(output / "params/agent.yaml"), agent)
    for relative in ("scripts/train_grasshopper_stand_push_capture.py",
                     "scripts/render_grasshopper_walk_snapshot.py",
                     "src/nest/tasks/stand/mdp/mixed_pushes.py",
                     "src/nest/tasks/stand/config/grasshopper/push_env_cfg.py",
                     "src/nest/tasks/stand/config/grasshopper/env_cfg.py"):
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)

    source = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    start_iteration = int(source["iter"]) + 1
    manifest = {
        "status": "initializing", "task": TASK, "seed": 42,
        "started_at": datetime.now().astimezone().isoformat(),
        "training_physics": "Isaac Lab Newton/MJWarp", "renderer": "Isaac Sim RTX",
        "num_envs": args.num_envs, "additional_iterations": args.iterations,
        "capture_iteration": args.capture_iteration, "completed_iterations": 0,
        "num_steps_per_env": agent.num_steps_per_env, "env_spacing": cfg.scene.env_spacing,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "source_iteration": int(source["iter"]), "start_iteration": start_iteration,
        "disturbance": {k: v for k, v in disturbance.items() if k != "asset_cfg"},
        "force_point": "TORSO center of mass", "force_frame": "world",
        "initial_exploration_std_floor": 0.12,
        "snapshot_source": "Exact end-of-rollout state at the requested PPO iteration; no replay or pose edits",
        "versions": {name: version(name) for name in ("newton", "mujoco-warp", "torch", "rsl-rl-lib")},
    }
    started = time.monotonic()
    render_process = None
    render_log = None
    raw = None

    def write_manifest():
        temporary = output / "run.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(output / "run.json")

    def snapshot(completed):
        nonlocal render_process, render_log
        robot = raw.scene["robot"]
        event = raw.event_manager.get_term_cfg("torso_forces").func
        positions = robot.data.root_pos_w.torch.detach().cpu().numpy()
        quats = robot.data.root_quat_w.torch.detach().cpu().numpy()
        joints = robot.data.joint_pos.torch.detach().cpu().numpy()
        forces = event.last_applied_force.detach().cpu().numpy()[:, 0]
        assert positions.shape == (args.num_envs, 3) and np.isfinite(joints).all()
        torso_ids, _ = robot.find_bodies("TORSO")
        torso_positions = robot.data.body_link_pos_w.torch[:, torso_ids[0]].detach().cpu().numpy()
        path = output / f"state_after_{completed:04d}.npz"
        np.savez_compressed(path, root_pos_w=positions, root_quat_w=quats,
                            joint_pos=joints, joint_names=robot.joint_names,
                            torso_force_w=forces, torso_pos_w=torso_positions,
                            env_origins=raw.scene.env_origins.detach().cpu().numpy(),
                            episode_step=raw.episode_length_buf.detach().cpu().numpy(),
                            last_actions=raw.action_manager.action.detach().cpu().numpy(),
                            training_iteration=completed, env_spacing=cfg.scene.env_spacing,
                            task=TASK, physics_backend="Newton/MJWarp")
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (quats[:, 0]**2 + quats[:, 1]**2), -1, 1)))
        manifest["capture"] = {
            "state": str(path), "iteration": completed,
            "global_control_step": int(raw.common_step_counter),
            "training_samples": completed * agent.num_steps_per_env * args.num_envs,
            "active_force_fraction": float((np.linalg.norm(forces, axis=1) > 1e-3).mean()),
            "max_force_n": float(np.linalg.norm(forces, axis=1).max()),
            "mean_joint_std_rad": float(joints.std(axis=0).mean()),
            "tilt_degrees_p50_p90_max": [float(x) for x in np.percentile(tilt, [50, 90, 100])],
        }
        write_manifest()
        print(f"CAPTURE_STATE_SAVED iteration={completed} path={path} metrics={manifest['capture']}", flush=True)
        if not args.skip_render:
            render_dir = output / f"render_after_{completed:04d}"
            command = [str(ROOT / "scripts/isaacsim-python.sh"), "-u",
                       str(ROOT / "scripts/render_grasshopper_walk_snapshot.py"),
                       "--state", str(path), "--output-dir", str(render_dir),
                       "--prefix", f"nest_stand_push_after_{completed:04d}",
                       "--width", "3840", "--height", "2160", "--grid-spacing", "2.5"]
            render_log = (output / "isaac_render.log").open("w")
            render_process = subprocess.Popen(command, cwd=ROOT, stdout=render_log, stderr=subprocess.STDOUT)
            manifest["render"] = {"status": "running", "pid": render_process.pid,
                                  "output_dir": str(render_dir), "command": command}
            write_manifest()

    class CaptureRunner(OnPolicyRunner):
        def save(self, path, infos=None):
            completed = self.current_learning_iteration - start_iteration + 1
            manifest["completed_iterations"] = completed
            manifest["training_samples"] = completed * agent.num_steps_per_env * args.num_envs
            manifest["elapsed_seconds"] = round(time.monotonic() - started, 2)
            if completed == args.capture_iteration or completed % 250 == 0 or completed == args.iterations:
                target = output / "checkpoints" / f"model_after_{completed:04d}.pt"
                target.parent.mkdir(exist_ok=True)
                super().save(str(target), infos)
                manifest["latest_checkpoint"] = str(target)
            if completed == args.capture_iteration and "capture" not in manifest:
                self.export_policy_to_jit(str(output / f"policy_after_{completed:04d}"))
                snapshot(completed)
            if render_process is not None and render_process.poll() is not None:
                manifest["render"]["return_code"] = render_process.returncode
                manifest["render"]["status"] = "complete" if render_process.returncode == 0 else "failed"
            if completed % 25 == 0 or completed == args.iterations:
                write_manifest()
            if completed == args.iterations:
                self.export_policy_to_jit(str(output / "policy_final"))

    write_manifest()
    try:
        with launch_simulation(cfg, args):
            env = gym.make(TASK, cfg=cfg)
            try:
                raw = env.unwrapped
                wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
                runner = CaptureRunner(wrapped, agent.to_dict(), str(output / "training"), agent.device)
                runner.load(str(args.checkpoint), strict=True, map_location=agent.device)
                runner.current_learning_iteration = start_iteration
                runner.alg.learning_rate = 1e-4
                for group in runner.alg.optimizer.param_groups:
                    group["lr"] = 1e-4
                with torch.no_grad():
                    runner.alg.get_policy().distribution.std_param.clamp_(min=0.12)
                manifest["status"] = "training"
                write_manifest()
                print(f"STAND_PUSH_TRAINING_STARTED output={output}", flush=True)
                runner.learn(args.iterations, init_at_random_ep_len=True)
                manifest["training_status"] = "complete"
                manifest["status"] = "waiting_for_render" if render_process and render_process.poll() is None else "complete"
                write_manifest()
            finally:
                env.close()
        if render_process is not None:
            result = render_process.wait()
            manifest["render"].update(status="complete" if result == 0 else "failed", return_code=result)
            if result:
                raise RuntimeError(f"Isaac renderer failed with exit code {result}; inspect isaac_render.log")
        manifest.update(status="complete", elapsed_seconds=round(time.monotonic() - started, 2))
        write_manifest()
        print(f"STAND_PUSH_COMPLETE output={output}", flush=True)
    except BaseException as error:
        import traceback
        manifest.update(status="failed", error=str(error), traceback=traceback.format_exc())
        write_manifest()
        raise
    finally:
        if render_log is not None:
            render_log.close()


if __name__ == "__main__":
    main()
