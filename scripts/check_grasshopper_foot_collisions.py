#!/usr/bin/env python3
"""Verify compiled MJWarp foot capsules and contact sensors in the walking task."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/nest-capsule-check-cache")
import warp as wp

wp.config.enable_backward = False
wp.config.kernel_cache_dir = "/tmp/nest-capsule-check-warp-cache"

from isaaclab.app import AppLauncher, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=180)
parser.add_argument("--training-settings", action="store_true", help="Also check actor/critic noise, friction and encoder bias")
parser.add_argument("--output", type=Path, default=Path("outputs/grasshopper_capsule_feet_check/report.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.steps < 1:
    parser.error("--num-envs and --steps must be positive")

import gymnasium as gym
import mujoco
import numpy as np
import torch
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_newton.physics import NewtonManager

import nest.tasks

cfg, _ = resolve_task_config("Nest-Walk-Flat-Grasshopper", None, overrides=["physics=newton_mjwarp"])
if not args.training_settings:
    cfg.play_mode()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.sim.visualizer_cfgs = []
cfg.seed = 42
cfg.validate()

with launch_simulation(cfg, args):
    wrapped = gym.make("Nest-Walk-Flat-Grasshopper", cfg=cfg)
    env = wrapped.unwrapped
    obs, _ = env.reset(seed=42)
    robot = env.scene["robot"]
    model = NewtonManager._solver.mj_model
    plane_ids = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    assert len(plane_ids) == 1, "Expected a native MuJoCo ground plane"
    feet = {}
    foot_geom_ids = []
    for name in ("R_FOOT", "L_FOOT"):
        bodies = [i for i in range(model.nbody)
                  if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").endswith(name)]
        assert bodies, f"No compiled {name} body"
        rows = []
        for body in bodies:
            geoms = np.flatnonzero((model.geom_bodyid == body)
                                  & ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
            assert len(geoms) == 7, (name, body, geoms)
            assert np.all(model.geom_type[geoms] == mujoco.mjtGeom.mjGEOM_CAPSULE)
            np.testing.assert_allclose(model.geom_size[geoms, 0], 0.01, rtol=1e-6)
            np.testing.assert_array_equal(model.geom_priority[geoms], 1)
            np.testing.assert_array_equal(model.geom_condim[geoms], 3)
            np.testing.assert_allclose(model.geom_margin[geoms], 0.0)
            np.testing.assert_allclose(model.geom_gap[geoms], 0.0)
            np.testing.assert_allclose(model.geom_solref[geoms], np.tile([0.02, 1.0], (7, 1)), rtol=1e-6)
            foot_geom_ids.extend(geoms.tolist())
            rows.append({"body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body),
                         "collision_types": model.geom_type[geoms].tolist(),
                         "positions_m": model.geom_pos[geoms].tolist(),
                         "contact_priority": model.geom_priority[geoms].tolist(),
                         "contact_solref": model.geom_solref[geoms].tolist(),
                         "radius_and_half_length_m": model.geom_size[geoms, :2].tolist()})
        feet[name] = rows
    sensor = env.scene["feet_contact"]
    peak_force = 0.0
    contact_steps = 0
    resets = 0
    settings = {}
    if args.training_settings:
        assert tuple(obs["policy"].shape) == (args.num_envs, 54)
        assert tuple(obs["critic"].shape) == (args.num_envs, 66)
        first = env.observation_manager.compute()
        second = env.observation_manager.compute()
        assert not torch.equal(first["policy"], second["policy"]), "Actor noise is missing"
        torch.testing.assert_close(first["critic"], second["critic"], rtol=0, atol=0)
        bias = env._velocity_encoder_bias
        assert tuple(bias.shape) == (args.num_envs, 14)
        assert bias.abs().max() <= 0.015
        assert bias.std() > 0
        settings["encoder_bias_max_rad"] = bias.abs().max().item()
        settings["actor_shape"] = list(obs["policy"].shape)
        settings["critic_shape"] = list(obs["critic"].shape)
        settings["actor_noisy_critic_clean"] = True
    with torch.inference_mode():
        for _ in range(args.steps):
            obs, reward, terminated, truncated, _ = env.step(torch.zeros((args.num_envs, 14), device=env.device))
            forces = sensor.data.net_normal_forces_w.torch
            assert tuple(forces.shape) == (args.num_envs, 2, 3), forces.shape
            assert torch.isfinite(forces).all()
            assert all(torch.isfinite(value).all() for value in obs.values()) and torch.isfinite(reward).all()
            peak_force = max(peak_force, forces.norm(dim=-1).max().item())
            contact_steps += int((forces.norm(dim=-1) > 1).any())
            resets += int((terminated | truncated).sum())
    assert contact_steps > 0, "Foot sensors recorded no ground contact"
    if args.training_settings:
        friction = wp.to_torch(NewtonManager._solver.mjw_model.geom_friction)
        # Startup randomization must reach the solver, with one value shared by
        # all fourteen capsules in a robot, and different draws across robots.
        values = friction[:, foot_geom_ids, 0]
        assert values.shape == (args.num_envs, 14), values.shape
        assert torch.all((values >= 0.3) & (values <= 1.2))
        assert (values.max(dim=1).values - values.min(dim=1).values).max() < 1e-6
        assert values[:, 0].std() > 0
        settings["foot_friction_per_env"] = values[:, 0].cpu().tolist()
        settings["shared_foot_friction_verified_in_solver"] = True
        # Validate effective randomization in the solver, including coherent
        # inertia scaling. Checking config ranges alone would miss routing bugs.
        mjw = NewtonManager._solver.mjw_model
        masses = wp.to_torch(mjw.body_mass)
        physical = model.body_mass > 0
        density = masses[:, physical] / torch.as_tensor(model.body_mass[physical], device=env.device)
        assert ((density >= 0.9 - 1e-6) & (density <= 1.1 + 1e-6)).all()
        assert density.std() > 0
        inertia = wp.to_torch(mjw.body_inertia)[:, physical]
        nominal_inertia = torch.as_tensor(model.body_inertia[physical], device=env.device)
        torch.testing.assert_close(inertia.double(), nominal_inertia * density[..., None], rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(
            masses.sum(-1), robot.data.body_mass.torch.sum(-1), rtol=1e-5, atol=1e-4)
        # The template contains nominal gains; startup writes per-world arrays.
        # Newton emits separate position, velocity and feed-forward actuators.
        # Only negative position bias identifies the fourteen Kp elements.
        kp_ids = model.actuator_biasprm[:, 1] < 0
        kp = wp.to_torch(mjw.actuator_gainprm)[:, kp_ids, 0]
        kp_ratio = kp / torch.as_tensor(model.actuator_gainprm[kp_ids, 0], device=env.device)
        kd_ids = model.actuator_biasprm[:, 2] < 0
        kd = wp.to_torch(mjw.actuator_biasprm)[:, kd_ids, 2]
        kd_ratio = kd / torch.as_tensor(model.actuator_biasprm[kd_ids, 2], device=env.device)
        assert kp.shape[-1] == kd.shape[-1] == 14
        torch.testing.assert_close(kp, -wp.to_torch(mjw.actuator_biasprm)[:, kp_ids, 1])
        torch.testing.assert_close(kd, -wp.to_torch(mjw.actuator_gainprm)[:, kd_ids, 0])
        for ratio in (kp_ratio, kd_ratio):
            assert ((ratio >= .9 - 1e-6) & (ratio <= 1.1 + 1e-6)).all()
            assert ratio.std() > 0
        from nest.assets.robots.grasshopper import POLICY_JOINT_NAMES
        dofs = []
        for name in POLICY_JOINT_NAMES:
            joints = [j for j in range(model.njnt)
                      if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").endswith(name)]
            assert len(joints) == 1, (name, joints)
            dofs.append(model.jnt_dofadr[joints[0]])
        dry_friction = wp.to_torch(mjw.dof_frictionloss)[:, dofs]
        assert dry_friction.max() <= 0.5 + 1e-6
        assert dry_friction.min() >= 0 and dry_friction.std() > 0
        delay = env.action_manager.get_term("joint_pos")._delay_buffer
        assert delay is not None and delay.ticks >= args.steps * cfg.decimation
        assert ((delay.lags >= 0) & (delay.lags <= 4)).all()
        settings["body_density_scale_min_max"] = [density.min().item(), density.max().item()]
        settings["kp_scale_min_max"] = [kp_ratio.min().item(), kp_ratio.max().item()]
        settings["kd_scale_min_max"] = [kd_ratio.min().item(), kd_ratio.max().item()]
        settings["joint_dry_friction_max_nm"] = dry_friction.max().item()
        settings["command_delay_lags"] = delay.lags.cpu().tolist()
        settings["delay_physics_ticks"] = delay.ticks
        settings["mass_and_inertia_scaling_verified_in_solver"] = True
        from nest.tasks.velocity.mdp.rewards import angular_momentum_penalty

        solver = NewtonManager._solver
        # Startup density DR makes each world's inertial parameters distinct.
        # Compare against a copy of this exact world's native model.
        import copy
        native_model = copy.copy(model)
        for field in ("body_mass", "body_inertia", "body_iquat", "body_ipos"):
            getattr(native_model, field)[:] = getattr(solver.mjw_model, field).numpy()[0]
        native = mujoco.MjData(native_model)
        mujoco.mj_setConst(native_model, native)
        native.qpos[:] = solver.mjw_data.qpos.numpy()[0]
        native.qvel[:] = solver.mjw_data.qvel.numpy()[0]
        mujoco.mj_forward(native_model, native)
        mujoco.mj_subtreeVel(native_model, native)
        native_l2 = float(np.square(native.subtree_angmom[1]).sum())
        reconstructed_l2 = angular_momentum_penalty(env)[0].item()
        np.testing.assert_allclose(reconstructed_l2, native_l2, rtol=2e-3, atol=1e-3)
        settings["angular_momentum_squared_native"] = native_l2
        settings["angular_momentum_squared_reconstructed"] = reconstructed_l2
    report = {"usd_path": cfg.scene.robot.spawn.usd_path, "num_envs": args.num_envs, "steps": args.steps,
              "simulated_seconds_per_env": args.steps * env.step_dt,
              "feet": feet, "sensor_shape": list(forces.shape), "peak_normal_force_n": peak_force,
              "steps_with_foot_contact": contact_steps, "episode_resets": resets,
              "mass_kg": robot.data.body_mass.torch[0].sum().item(), "finite_observations_and_rewards": True,
              "native_ground_plane": True, "control_dt_s": env.step_dt,
              "solver_iterations": model.opt.iterations, "solver_ls_iterations": model.opt.ls_iterations,
              "training_settings": settings,
              "note": "Zero-action contact smoke check; this does not evaluate a trained gait."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    wrapped.close()
