#!/usr/bin/env python3
"""Prepare or train Grasshopper walking and yaw-rate tracking (forward <= 1 m/s)."""

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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nest-walk-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/nest-walk-cache")

import warp as wp

wp.config.enable_backward = False
wp.config.kernel_cache_dir = "/tmp/nest-walk-warp-cache"

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
TASK = "Nest-Walk-Flat-Grasshopper"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--init-checkpoint", type=Path, help="Opt into the legacy 55-input/256-wide standing actor; critic/optimizer start fresh")
    source.add_argument("--resume", type=Path, help="Resume a walking checkpoint with the same v2 training design")
    source.add_argument("--from-scratch", action="store_true", help="Default: fresh MJLab-style actor and critic")
    parser.add_argument("--prepare-only", action="store_true", help="Write configuration and initial actor without simulation")
    parser.add_argument("--cpu-smoke", action="store_true", help="One env, at most two PPO iterations; contact-dependent rewards omitted")
    parser.add_argument("--iterations", type=int, default=30000, help="New PPO iterations in this invocation")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.iterations < 1 or args.num_envs < 1:
        parser.error("iterations and num-envs must be positive")
    if args.cpu_smoke and (args.prepare_only or args.resume):
        parser.error("cpu-smoke cannot be combined with prepare-only or resume")

    import gymnasium as gym
    import torch
    from rsl_rl.models import MLPModel
    from rsl_rl.runners import OnPolicyRunner
    from tensordict import TensorDict

    from isaaclab.app import launch_simulation
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import resolve_task_config
    import nest.tasks  # noqa: F401
    from nest.assets.robots.grasshopper import FOOT_COLLISION_DESIGN, POLICY_JOINT_NAMES, URDF_PATH, USD_PATH
    from nest.tasks.velocity.config.grasshopper.env_cfg import (
        COMMAND_DESIGN, CRITIC_OBSERVATION_SIZE, DOMAIN_RANDOMIZATION_DESIGN, MAX_FORWARD_SPEED,
        MAX_YAW_RATE,
        POLICY_OBSERVATION_SIZE, TRAINING_DESIGN,
    )
    from nest.tasks.velocity.mdp.curriculums import apply_velocity_stage_ranges, velocity_curriculum_start_step
    from nest.tasks.velocity.warm_start import initialize_from_standing, initialize_yaw_command

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg, agent = resolve_task_config(TASK, "rsl_rl_cfg_entry_point", overrides=["physics=newton_mjwarp"])
    checkpoint = args.resume or args.init_checkpoint
    legacy_actor = args.init_checkpoint is not None
    source_dr_design = None
    previous = None
    saved = None
    if checkpoint is not None:
        if not checkpoint.is_file():
            parser.error(f"Checkpoint does not exist: {checkpoint}")
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        if args.resume:
            source_manifest = checkpoint.parent / "run.json"
            if not source_manifest.is_file():
                parser.error("Walking resume requires the source run.json beside the checkpoint")
            previous = json.loads(source_manifest.read_text())
            source_dr_design = previous.get("domain_randomization_design", "mjlab_g1_base_v1")
            if previous["task"] != TASK or previous["mode"] != "full_walking":
                parser.error("Resume requires a full walking checkpoint, not a CPU smoke checkpoint")
            if previous.get("training_design") != TRAINING_DESIGN:
                parser.error("This checkpoint predates the v2 observation/critic/physics design; start a fresh run")
            if previous["num_steps_per_env"] != agent.num_steps_per_env:
                parser.error("Cannot resume with a different number of rollout steps per iteration")
            legacy_actor = previous.get("legacy_standing_actor", False)
        elif saved["actor_state_dict"]["mlp.0.weight"].shape != (256, 52):
            parser.error("Standing initialization requires a 52-input, 256-wide actor")
    if legacy_actor:
        cfg.use_legacy_policy_observations()
        agent.actor.hidden_dims = [256, 128, 128]
        agent.actor.obs_normalization = False
    policy_size = 55 if legacy_actor else POLICY_OBSERVATION_SIZE
    start_iteration = int(saved["iter"]) + 1 if args.resume else 0
    source_command_design = previous.get("command_design", "forward_only_v1") if previous else None
    migrate_yaw_input = bool(args.resume and source_command_design != COMMAND_DESIGN)
    iterations = min(args.iterations, 2) if args.cpu_smoke else args.iterations
    cfg.scene.num_envs = 1 if args.cpu_smoke else args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = "cpu" if args.cpu_smoke else args.device or "cuda:0"
    cfg.sim.visualizer_cfgs = []
    if not args.prepare_only and not args.cpu_smoke:
        if not cfg.sim.device.startswith("cuda") or not torch.cuda.is_available():
            parser.error("GPU training requires CUDA. Use --prepare-only or --cpu-smoke in a CPU-only session")
    if args.cpu_smoke:
        from isaaclab_newton.physics import NewtonManager, NewtonMJWarpManager

        class NewtonCpuSmokeManager(NewtonMJWarpManager):
            @classmethod
            def _initialize_contacts(cls):
                if cls._report_contacts:
                    raise RuntimeError("Native MuJoCo CPU smoke does not support Newton contact reporting")
                NewtonManager._contacts = None

        cfg.scene.feet_contact = None
        cfg.scene.robot.spawn.activate_contact_sensors = False
        cfg.rewards.foot_swing_height = None
        cfg.rewards.foot_slip = None
        cfg.rewards.soft_landing = None
        cfg.observations.critic.foot_air_time = None
        cfg.observations.critic.foot_contact = None
        cfg.observations.critic.foot_contact_forces = None
        cfg.sim.physics.solver_cfg.use_mujoco_cpu = True
        cfg.sim.physics.solver_cfg.class_type = NewtonCpuSmokeManager
        cfg.sim.physics.class_type = NewtonCpuSmokeManager
        agent.experiment_name += "_cpu_smoke"
    # First transfer from a forward-only policy starts the yaw curriculum at zero.
    # Subsequent resumes retain that curriculum's clock and all command ranges.
    curriculum_start = velocity_curriculum_start_step(
        previous, start_iteration, agent.num_steps_per_env, COMMAND_DESIGN,
    )
    cfg.curriculum.command_vel.params["start_step"] = curriculum_start
    apply_velocity_stage_ranges(
        cfg.commands.base_velocity.ranges, cfg.curriculum.command_vel.params["velocity_stages"],
        start_iteration * agent.num_steps_per_env - curriculum_start,
    )
    agent.seed, agent.device, agent.max_iterations = args.seed, cfg.sim.device, iterations
    agent = handle_deprecated_rsl_rl_cfg(agent, version("rsl-rl-lib"))
    agent.save_interval = 1  # The runner below selects actual checkpoint/export intervals.
    cfg.validate()
    log_dir = args.log_dir or ROOT / "logs/rsl_rl" / agent.experiment_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "task": TASK, "status": "preparing", "seed": args.seed,
        "device": cfg.sim.device, "cuda_available": torch.cuda.is_available(),
        "mode": "cpu_smoke_without_contact_rewards" if args.cpu_smoke else "full_walking",
        "source_checkpoint": str(checkpoint.resolve()) if checkpoint else None,
        "source_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None,
        "initialization": "resume_full_ppo" if args.resume else "standing_actor_only" if legacy_actor else "scratch",
        "training_design": TRAINING_DESIGN, "legacy_standing_actor": legacy_actor,
        "command_design": COMMAND_DESIGN, "source_command_design": source_command_design,
        "velocity_curriculum_start_step": curriculum_start,
        "yaw_input_initialized": migrate_yaw_input,
        "commands": {
            "lin_vel_x": cfg.commands.base_velocity.ranges.lin_vel_x,
            "lin_vel_y": cfg.commands.base_velocity.ranges.lin_vel_y,
            "ang_vel_z": cfg.commands.base_velocity.ranges.ang_vel_z,
            "heading": cfg.commands.base_velocity.ranges.heading,
            "heading_command": cfg.commands.base_velocity.heading_command,
            "heading_control_stiffness": cfg.commands.base_velocity.heading_control_stiffness,
            "rel_heading_envs": cfg.commands.base_velocity.rel_heading_envs,
            "rel_forward_envs": cfg.commands.base_velocity.rel_forward_envs,
            "rel_standing_envs": cfg.commands.base_velocity.rel_standing_envs,
            "resampling_time_range": cfg.commands.base_velocity.resampling_time_range,
        },
        "domain_randomization_design": DOMAIN_RANDOMIZATION_DESIGN,
        "source_domain_randomization_design": source_dr_design,
        "domain_randomization": {
            "body_density_scale_range": cfg.events.body_density.params["scale_range"],
            "kp_scale_range": cfg.events.actuator_gains.params["stiffness_distribution_params"],
            "kd_scale_range": cfg.events.actuator_gains.params["damping_distribution_params"],
            "joint_dry_friction_nm": cfg.events.joint_friction.params["friction_range"],
            "command_delay_physics_steps": [cfg.actions.joint_pos.delay_min_lag, cfg.actions.joint_pos.delay_max_lag],
            "command_delay_update_period_steps": cfg.actions.joint_pos.delay_update_period,
            "physics_dt_s": cfg.sim.dt,
        },
        "source_iteration": int(saved["iter"]) if saved else None,
        "start_iteration": start_iteration, "requested_iterations": iterations, "completed_iterations": 0,
        "num_envs": cfg.scene.num_envs, "num_steps_per_env": agent.num_steps_per_env,
        "episode_length_s": cfg.episode_length_s, "max_forward_command_m_s": MAX_FORWARD_SPEED,
        "max_yaw_command_rad_s": MAX_YAW_RATE,
        "velocity_stages": cfg.curriculum.command_vel.params["velocity_stages"],
        "policy_observation_size": policy_size,
        "critic_observation_size": CRITIC_OBSERVATION_SIZE - (10 if args.cpu_smoke else 0),
        "actor_hidden_dims": agent.actor.hidden_dims, "actor_obs_normalization": agent.actor.obs_normalization,
        "control_dt_s": cfg.sim.dt * cfg.decimation,
        "reward_design": "mjlab_g1_flat_adapted_v2",
        "foot_collision_design": FOOT_COLLISION_DESIGN,
        "robot_usd_path": str(USD_PATH),
        "robot_usd_sha256": hashlib.sha256(USD_PATH.read_bytes()).hexdigest(),
        "omitted_rewards": ["foot_swing_height", "foot_slip", "soft_landing"] if args.cpu_smoke else [],
        "action_joints": list(POLICY_JOINT_NAMES), "action_scale_rad": cfg.actions.joint_pos.scale,
        "started_at": datetime.now().astimezone().isoformat(),
        "versions": {name: version(name) for name in ("newton", "mujoco", "mujoco-warp", "torch", "rsl-rl-lib")},
    }

    def write_manifest():
        (log_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")

    write_manifest()

    class WalkingRunner(OnPolicyRunner):
        def save(self, path, infos=None):
            completed = self.current_learning_iteration + 1
            if completed % 50 and completed != start_iteration + iterations:
                return
            super().save(path, infos)
            manifest["completed_iterations"] = completed - start_iteration
            manifest["total_walking_iterations"] = completed
            manifest["latest_checkpoint"] = str(Path(path).resolve())
            if completed % 1000 == 0 or completed == start_iteration + iterations:
                export_dir = log_dir / f"exports/iteration_{completed:05d}"
                self.export_policy_to_jit(str(export_dir))
                self.export_policy_to_onnx(str(export_dir))
                manifest["latest_policy"] = str(export_dir / "policy.pt")
            write_manifest()

    try:
        dump_yaml(str(log_dir / "params/env.yaml"), cfg)
        dump_yaml(str(log_dir / "params/agent.yaml"), agent)
        sources = [
            ROOT / "scripts/train_grasshopper_walk.py",
            ROOT / "scripts/build_grasshopper_capsule_feet.py",
            URDF_PATH,
            USD_PATH,
            ROOT / "src/nest/assets/robots/grasshopper.py",
            ROOT / "src/nest/tasks/stand/config/grasshopper/env_cfg.py",
            ROOT / "src/nest/tasks/stand/config/grasshopper/agents/rsl_rl_ppo_cfg.py",
            ROOT / "src/nest/tasks/velocity/warm_start.py",
            ROOT / "src/nest/tasks/velocity/mdp/curriculums.py",
            ROOT / "src/nest/tasks/velocity/mdp/commands.py",
            ROOT / "src/nest/tasks/velocity/mdp/rewards.py",
            ROOT / "src/nest/tasks/velocity/mdp/__init__.py",
            ROOT / "src/nest/tasks/velocity/mdp/observations.py",
            ROOT / "src/nest/tasks/velocity/mdp/events.py",
            ROOT / "src/nest/tasks/velocity/mdp/actions.py",
            ROOT / "assets/terrains/flat_plane.usda",
            ROOT / "src/nest/sensors/foot_height.py",
            ROOT / "src/nest/tasks/velocity/config/grasshopper/README.md",
            *sorted((ROOT / "src/nest/tasks/velocity/config/grasshopper").rglob("*.py")),
        ]
        for path in sources:
            destination = log_dir / "source" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        if args.prepare_only:
            actor_cfg = agent.actor.to_dict()
            actor_cfg.pop("class_name")
            actor = MLPModel(TensorDict({"policy": torch.zeros(1, policy_size)}, batch_size=[1]),
                             agent.obs_groups, "actor", 14, **actor_cfg)
            if args.resume:
                actor.load_state_dict(saved["actor_state_dict"], strict=True)
                if migrate_yaw_input:
                    initialize_yaw_command(actor, policy_size - 1)
            elif saved is not None:
                initialize_from_standing(actor, saved["actor_state_dict"])
            torch.save(actor.state_dict(), log_dir / "initial_actor.pt")
            manifest["status"] = "prepared"
            manifest["simulation_validated"] = False
            write_manifest()
            print(f"WALK_PREPARED log_dir={log_dir}", flush=True)
            return
        with launch_simulation(cfg, args):
            with contextlib.closing(gym.make(TASK, cfg=cfg)) as env:
                env.unwrapped.common_step_counter = start_iteration * agent.num_steps_per_env
                wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
                if wrapped.get_observations()["policy"].shape[-1] != policy_size or wrapped.num_actions != 14:
                    raise AssertionError("Walking observation/action layout changed")
                command = env.unwrapped.command_manager.get_term("base_velocity")
                manifest["initial_command_samples"] = {
                    "positive_yaw": int((command.command[:, 2] > 0.05).sum()),
                    "negative_yaw": int((command.command[:, 2] < -0.05).sum()),
                    "standing": int(command.is_standing_env.sum()),
                    "forward_mask": int(command.is_forward_env.sum()),
                    "heading_mask": int(command.is_heading_env.sum()),
                    "max_abs_yaw_rad_s": float(command.command[:, 2].abs().max()),
                }
                if not args.cpu_smoke:
                    sensor = env.unwrapped.scene["feet_contact"]
                    if sensor.data.net_normal_forces_w.torch.shape != (cfg.scene.num_envs, 2, 3):
                        raise AssertionError("Expected exactly two foot contact sensors per environment")
                    manifest["foot_sensor_names"] = sensor.sensor_names
                runner = WalkingRunner(wrapped, agent.to_dict(), str(log_dir), agent.device)
                if args.resume:
                    runner.load(str(checkpoint), strict=True, map_location=agent.device)
                    runner.current_learning_iteration = start_iteration
                    if migrate_yaw_input:
                        initialize_yaw_command(runner.alg.get_policy(), policy_size - 1, runner.alg.optimizer)
                        initialize_yaw_command(runner.alg.critic, POLICY_OBSERVATION_SIZE - 1, runner.alg.optimizer)
                elif saved is not None:
                    initialize_from_standing(runner.alg.get_policy(), saved["actor_state_dict"])
                manifest["status"] = "training"
                write_manifest()
                print(f"WALK_TRAINING_STARTED log_dir={log_dir}", flush=True)
                runner.learn(iterations, init_at_random_ep_len=True)
                if not torch.isfinite(env.unwrapped.reward_manager._step_reward).all():
                    raise AssertionError("Non-finite per-term walking rewards")
                manifest["final_control_step"] = env.unwrapped.common_step_counter
                manifest["simulation_validated"] = True
                if not args.cpu_smoke:
                    forces = env.unwrapped.scene["feet_contact"].data.net_normal_forces_w.torch
                    if not torch.isfinite(forces).all():
                        raise AssertionError("Non-finite foot contact forces")
                    manifest["final_max_foot_normal_force_n"] = float(forces.norm(dim=-1).max())
                manifest["status"] = "cpu_smoke_complete" if args.cpu_smoke else "complete"
                write_manifest()
                print(f"WALK_{manifest['status'].upper()} log_dir={log_dir}", flush=True)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        write_manifest()
        raise


if __name__ == "__main__":
    main()
