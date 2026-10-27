#!/usr/bin/env python3
"""Smoke-test MJLab-style G1 foot height rays in the native velocity task."""

from __future__ import annotations

import argparse
import contextlib

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    choices=("Nest-Velocity-Flat-G1-Foot-Height", "Nest-Velocity-Flat-G1-MJLab-Rewards"),
    default="Nest-Velocity-Flat-G1-Foot-Height",
)
parser.add_argument("--num-envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--report-every", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

import gymnasium as gym
import torch

from isaaclab.app import launch_simulation

import nest.tasks  # noqa: F401  Register the task.
from isaaclab_tasks.utils import resolve_task_config
from nest.sensors.foot_height import foot_heights


def main() -> None:
    cfg, _ = resolve_task_config(args.task, None, overrides=[])
    cfg.scene.num_envs = args.num_envs
    if args.device is not None:
        cfg.sim.device = args.device
    cfg.validate()
    with launch_simulation(cfg, args):
        with contextlib.closing(gym.make(args.task, cfg=cfg)) as env:
            env.reset()
            actions = torch.zeros_like(env.unwrapped.action_manager.action)
            for step in range(args.steps + 1):
                scene = env.unwrapped.scene
                values = foot_heights(scene)
                if values.shape != (args.num_envs, 2) or not torch.isfinite(values).all():
                    raise AssertionError(f"Invalid foot heights: {values}")
                hit_counts = torch.stack(
                    [
                        torch.isfinite(scene[f"{side}_foot_height"].data.ray_hits_w.torch[:, :, 2]).sum(dim=-1)
                        for side in ("left", "right")
                    ],
                    dim=-1,
                )
                if step == 0 and not (hit_counts > 0).all():
                    raise AssertionError(f"Flat-ground rays missed at reset: {hit_counts}")
                report = step % args.report_every == 0 or step == args.steps
                if report:
                    print(
                        f"step={step} foot_heights_m={values.detach().cpu().tolist()}"
                        f" ray_hits={hit_counts.detach().cpu().tolist()}"
                    )
                if step < args.steps:
                    with torch.inference_mode():
                        _, reward, _, _, _ = env.step(actions)
                    if not torch.isfinite(reward).all():
                        raise AssertionError(f"Non-finite reward: {reward}")
                    term_rewards = env.unwrapped.reward_manager._step_reward
                    if not torch.isfinite(term_rewards).all():
                        raise AssertionError(f"Non-finite reward term: {term_rewards}")
                    if report:
                        print(f"reward_step={step + 1} value={reward.detach().cpu().tolist()}")
            print(f"reward_terms_env0={dict(env.unwrapped.reward_manager.get_active_iterable_terms(0))}")


if __name__ == "__main__":
    main()
