#!/usr/bin/env python3
"""Controlled short PPO benchmarks for Grasshopper walking overhead.

Run each case in a separate process. These are diagnostics, not policy training
artifacts. This historical benchmark explicitly retains the standing actor's
256-wide network and legacy observation/action layout for transfer comparisons.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nest-walk-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/nest-walk-cache")

import warp as wp

wp.config.enable_backward = False
wp.config.kernel_cache_dir = "/tmp/nest-walk-warp-cache"

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=("stand", "walk", "walk_no_sensor", "walk_plane", "walk_no_sensor_plane", "walk_stand_rewards"))
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--profile-steps", type=int, default=0, help="Extra deterministic steps with synchronized component timers")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--output-dir", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.iterations < 5 or args.num_envs < 4:
        parser.error("Use at least 5 iterations and 4 environments")

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from isaaclab.app import launch_simulation
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import resolve_task_config
    from isaaclab_newton.physics import NewtonManager
    import nest.tasks  # noqa: F401
    from nest.tasks.velocity.warm_start import initialize_from_standing
    from importlib.metadata import version

    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    task = "Nest-Stand-Grasshopper" if args.case == "stand" else "Nest-Walk-Flat-Grasshopper"
    cfg, _ = resolve_task_config(task, None, overrides=["physics=newton_mjwarp"])
    _, agent = resolve_task_config("Nest-Walk-Flat-Grasshopper", "rsl_rl_cfg_entry_point", overrides=["physics=newton_mjwarp"])
    agent.actor.hidden_dims = [256, 128, 128]
    agent.actor.obs_normalization = False
    if args.case == "stand":
        agent.obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    else:
        cfg.use_legacy_policy_observations()
    if "plane" in args.case:
        stand, _ = resolve_task_config("Nest-Stand-Grasshopper", None, overrides=["physics=newton_mjwarp"])
        cfg.scene.ground = stand.scene.ground.copy()
    if args.case == "stand" or "plane" in args.case:
        cached = Path("/tmp/https") / cfg.scene.ground.spawn.usd_path.removeprefix("https://")
        if not cached.is_file():
            parser.error(f"Ground asset is not cached: {cached}")
        cfg.scene.ground.spawn.usd_path = str(cached)
    if "no_sensor" in args.case or args.case == "walk_stand_rewards":
        cfg.scene.feet_contact = None
        cfg.scene.robot.spawn.activate_contact_sensors = False
        cfg.rewards.foot_swing_height = None
        cfg.rewards.foot_slip = None
        cfg.rewards.soft_landing = None
        cfg.observations.critic.foot_air_time = None
        cfg.observations.critic.foot_contact = None
        cfg.observations.critic.foot_contact_forces = None
    if args.case == "walk_stand_rewards":
        stand, _ = resolve_task_config("Nest-Stand-Grasshopper", None, overrides=["physics=newton_mjwarp"])
        cfg.rewards = stand.rewards.copy()
    cfg.scene.num_envs, cfg.seed = args.num_envs, 42
    cfg.sim.device, cfg.sim.visualizer_cfgs = args.device or "cuda:0", []
    agent.device, agent.seed, agent.max_iterations = cfg.sim.device, 42, args.iterations
    agent.save_interval = 10**9
    agent = handle_deprecated_rsl_rl_cfg(agent, version("rsl-rl-lib"))
    cfg.validate()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    dump_yaml(str(out / "params/env.yaml"), cfg)
    dump_yaml(str(out / "params/agent.yaml"), agent)
    saved = torch.load(ROOT / "logs/rsl_rl/grasshopper_stand/2026-09-24_18-54-51/model_4999.pt", weights_only=True, map_location="cpu")
    started = time.perf_counter()
    with launch_simulation(cfg, args):
        with contextlib.closing(gym.make(task, cfg=cfg)) as env:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)

            class BenchmarkRunner(OnPolicyRunner):
                def save(self, path, infos=None):
                    pass  # No checkpoint/export I/O during timing.

            runner = BenchmarkRunner(wrapped, agent.to_dict(), str(out), agent.device)
            if args.case == "stand":
                source = {k: v for k, v in saved["actor_state_dict"].items() if k != "distribution.std_param"}
                runner.alg.get_policy().load_state_dict(source, strict=False)
            else:
                initialize_from_standing(runner.alg.get_policy(), saved["actor_state_dict"])
            runner.learn(args.iterations, init_at_random_ep_len=True)
            component_times = {}
            if args.profile_steps:
                base = env.unwrapped
                targets = [
                    (base.sim, "step", "physics"),
                    (base.scene, "write_data_to_sim", "write_state_and_actuators"),
                    (base.scene, "update", "scene_update"),
                    (base.reward_manager, "compute", "rewards"),
                    (base.command_manager, "compute", "commands"),
                    (base.observation_manager, "compute", "observations"),
                    (base.termination_manager, "compute", "terminations"),
                    (base.action_manager, "apply_action", "apply_action"),
                    (base.action_manager, "process_action", "process_action"),
                    (base, "_reset_idx", "resets"),
                ]

                def timer(fn, name):
                    def wrapped_call(*a, **kw):
                        torch.cuda.synchronize()
                        before = time.perf_counter()
                        result = fn(*a, **kw)
                        torch.cuda.synchronize()
                        component_times.setdefault(name, []).append(time.perf_counter() - before)
                        return result
                    return wrapped_call

                for obj, method, name in targets:
                    setattr(obj, method, timer(getattr(obj, method), name))
                policy = runner.get_inference_policy()
                obs = wrapped.get_observations()
                component_times.clear()
                before = time.perf_counter()
                with torch.inference_mode():
                    for _ in range(args.profile_steps):
                        obs, _, _, _ = wrapped.step(policy(obs))
                torch.cuda.synchronize()
                component_times = {
                    "method": "wall timers with CUDA synchronization around each component; synchronization changes scheduling",
                    "steps": args.profile_steps,
                    "elapsed_s": time.perf_counter() - before,
                    "milliseconds_per_control_step": {
                        name: sum(times) * 1000 / args.profile_steps for name, times in component_times.items()
                    },
                }
            solver = NewtonManager._solver
            state = {
                "cuda_graph_active": NewtonManager._graph is not None,
                "all_actuators_graphable": NewtonManager._is_all_graphable(),
                "report_contacts": NewtonManager._report_contacts,
                "mujoco_nv": solver.mj_model.nv,
                "mujoco_ngeom": solver.mj_model.ngeom,
                "mujoco_nconmax": solver.mjw_data.naconmax,
                "mujoco_nworld": solver.mjw_data.nworld,
            }
    events = EventAccumulator(str(out), size_guidance={"scalars": 0})
    events.Reload()
    collection = [x.value for x in events.Scalars("Perf/collection_time")]
    learning = [x.value for x in events.Scalars("Perf/learning_time")]
    total = [x + y for x, y in zip(collection, learning)]
    result = {
        "case": args.case, "num_envs": args.num_envs, "iterations": args.iterations,
        "discarded_warmup_iterations": 3,
        "median_collection_s": statistics.median(collection[3:]),
        "median_learning_s": statistics.median(learning[3:]),
        "median_iteration_s": statistics.median(total[3:]),
        "estimated_5000_iterations_hours": statistics.median(total[3:]) * 5000 / 3600,
        "process_elapsed_s": time.perf_counter() - started,
        "runtime": state,
        "synchronized_component_profile": component_times,
    }
    (out / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print("BENCHMARK_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
