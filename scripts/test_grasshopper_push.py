#!/usr/bin/env python3
"""Evaluate the saved standing policy under finite, horizontal torso-force pulses."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/nest-push-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nest-push-matplotlib")

import warp as wp

wp.config.enable_backward = False
wp.config.kernel_cache_dir = "/tmp/nest-push-warp-cache"

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "logs/rsl_rl/grasshopper_stand/2026-09-24_18-54-51"
DIRECTIONS = {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0), "-Y": (0, -1, 0)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=RUN / "exported/policy.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/grasshopper_push_test")
    parser.add_argument("--forces", type=float, nargs="+", default=[0, 20, 50, 100, 150, 200, 300])
    parser.add_argument("--directions", nargs="+", choices=list(DIRECTIONS), default=list(DIRECTIONS))
    parser.add_argument("--num-envs", type=int, default=8, help="Reset samples per force/direction.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push-at", type=float, default=2.0)
    parser.add_argument("--push-duration", type=float, default=0.2)
    parser.add_argument("--repeat-pushes", type=int, default=1)
    parser.add_argument("--push-interval", type=float, default=2.0)
    parser.add_argument("--extra-pulse-force", type=float, default=0.0,
                        help="Superimpose one transient force on the sustained-force test.")
    parser.add_argument("--extra-pulse-at", type=float, default=4.0)
    parser.add_argument("--extra-pulse-duration", type=float, default=0.2)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--body", default="TORSO")
    parser.add_argument("--mujoco-cpu", action="store_true", help="Use Newton's native MuJoCo CPU solver.")
    parser.add_argument("--video", action="store_true", help="Render sample 0 of each case with MuJoCo.")
    parser.add_argument("--video-label", default="")
    parser.add_argument("--video-no-overlay", action="store_true", help="Record video without the parameter and status banner.")
    parser.add_argument("--video-sample", type=int, default=0)
    parser.add_argument("--video-renderer", choices=("mujoco", "isaac"), default="mujoco")
    parser.add_argument("--camera-wide", action="store_true", help="Use a wide view that includes displacement and falls.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or any(f < 0 or not math.isfinite(f) for f in args.forces):
        parser.error("num-envs must be positive and forces finite and nonnegative")
    if not 0 <= args.video_sample < args.num_envs:
        parser.error("video-sample must select a valid reset sample")
    last_push = args.push_at + (args.repeat_pushes - 1) * args.push_interval
    if args.repeat_pushes < 1 or (args.repeat_pushes > 1 and args.push_interval <= args.push_duration):
        parser.error("repeat-pushes must be positive; repeated pushes need an interval greater than their duration")
    if not (0 < args.push_at < last_push + args.push_duration < args.seconds):
        parser.error("push must start after zero and end before the trial ends")
    if not math.isfinite(args.extra_pulse_force) or args.extra_pulse_force < 0:
        parser.error("extra-pulse-force must be finite and nonnegative")
    if args.extra_pulse_force and not (args.extra_pulse_at >= args.push_at
            and args.extra_pulse_duration > 0 and args.extra_pulse_at + args.extra_pulse_duration < args.seconds):
        parser.error("extra pulse must start at or after push-at and finish before the trial ends")
    return args


def main():
    args = parse_args()
    import gymnasium as gym
    import numpy as np
    import torch

    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import resolve_task_config
    import nest.tasks  # noqa: F401

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg, _ = resolve_task_config("Nest-Stand-Grasshopper", None, overrides=[])
    # Native MuJoCo supports one Newton world; run reset samples serially on CPU.
    cfg.scene.num_envs = 1 if args.mujoco_cpu else args.num_envs
    cached_ground = Path("/tmp/https") / cfg.scene.ground.spawn.usd_path.removeprefix("https://")
    if cached_ground.is_file():
        cfg.scene.ground.spawn.usd_path = str(cached_ground)
    cfg.seed = args.seed
    cfg.sim.device = "cpu" if args.mujoco_cpu else (args.device or "cuda:0")
    if args.mujoco_cpu:
        from isaaclab_newton.physics import NewtonMJWarpManager, NewtonManager

        class NewtonCpuPushTestManager(NewtonMJWarpManager):
            @classmethod
            def _initialize_contacts(cls):
                # Native MuJoCo solves contacts internally but cannot export them to
                # Newton contact sensors. This task uses only proprioception.
                if cls._report_contacts:
                    raise RuntimeError("CPU push evaluation does not support contact sensors")
                NewtonManager._contacts = None

        cfg.sim.physics.solver_cfg.use_mujoco_cpu = True
        cfg.sim.physics.solver_cfg.class_type = NewtonCpuPushTestManager
        cfg.sim.physics.class_type = NewtonCpuPushTestManager
    # Preserve fallen states for measurement/video. Apply the original fall criteria below,
    # latching the first failure so an automatic reset cannot be mistaken for recovery.
    cfg.terminations.time_out = None
    cfg.terminations.root_height = None
    cfg.terminations.bad_orientation = None
    cfg.sim.visualizer_cfgs = []
    if args.video and args.video_renderer == "isaac":
        from isaaclab_visualizers.kit import KitVisualizerCfg

        args.headless = True
        args.enable_cameras = True
        args.renderer = "RayTracedLighting"
        args.width, args.height = 1280, 720
        cfg.scene.dome_light.spawn.intensity = 650.0
        cfg.sim.visualizer_cfgs = [KitVisualizerCfg(
            headless=True, window_width=1280, window_height=720,
            origin_type="env", origin_env_index=args.video_sample,
            visible_env_indices=[args.video_sample],
            eye=(3.8, 3.8, 2.5) if args.camera_wide else (2.7, 2.7, 1.9),
            lookat=(0.3, 0.3, 0.75) if args.camera_wide else (0.0, 0.0, 0.95),
            focal_length=24.0 if args.camera_wide else 26.0,
            background_color=(0.12, 0.16, 0.22),
        )]
    cfg.validate()
    policy = torch.jit.load(str(args.policy), map_location=cfg.sim.device).eval()
    rows, traces, summaries = [], [], []
    with launch_simulation(cfg, args):
        with contextlib.closing(gym.make("Nest-Stand-Grasshopper", cfg=cfg)) as wrapped:
            env = wrapped.unwrapped
            batch_size = env.num_envs
            robot = env.scene["robot"]
            body_ids, body_names = robot.find_bodies(args.body)
            if len(body_ids) != 1:
                raise ValueError(f"Expected one force body, found {body_names}; bodies={robot.body_names}")
            dt = env.step_dt
            start, duration, steps = round(args.push_at / dt), round(args.push_duration / dt), round(args.seconds / dt)
            push_starts = [start + index * round(args.push_interval / dt) for index in range(args.repeat_pushes)]
            last_push_end = push_starts[-1] + duration
            extra_start = round(args.extra_pulse_at / dt)
            extra_duration = max(1, round(args.extra_pulse_duration / dt))
            if args.extra_pulse_force:
                last_push_end = max(last_push_end, extra_start + extra_duration)
            if duration < 1:
                raise ValueError("Push duration must be at least one control step")
            force = torch.zeros((batch_size, 1, 3), device=env.device)
            torque = torch.zeros_like(force)
            mass = float(robot.data.body_mass.torch[0].sum())
            renderer = None
            if args.video and args.video_renderer == "isaac":
                from pxr import Gf, UsdGeom, UsdLux

                renderer = next(v for v in env.sim._visualizers if v.cfg.visualizer_type == "kit")
                stage = renderer._scene_data_provider.usd_stage
                sun = UsdLux.DistantLight.Define(stage, "/World/PushRenderSun")
                sun.CreateIntensityAttr(1200.0)
                UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-45, 25, -15))
                arrow_root = UsdGeom.Xform.Define(stage, "/World/PushForceArrow")
                arrow_position = arrow_root.AddTranslateOp()
                arrow_rotation = arrow_root.AddOrientOp()
                shaft = UsdGeom.Cylinder.Define(stage, "/World/PushForceArrow/Shaft")
                shaft.CreateRadiusAttr(0.025)
                shaft.CreateHeightAttr(0.55)
                shaft.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.07, 0.03)])
                UsdGeom.Xformable(shaft).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.275))
                tip = UsdGeom.Cone.Define(stage, "/World/PushForceArrow/Tip")
                tip.CreateRadiusAttr(0.07)
                tip.CreateHeightAttr(0.1)
                tip.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.07, 0.03)])
                UsdGeom.Xformable(tip).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.6))
                arrow_root.MakeInvisible()
            elif args.video:
                import mujoco
                from isaaclab_newton.physics import NewtonManager
                solver = NewtonManager._solver
                solver.mj_model.vis.global_.offwidth = 960
                solver.mj_model.vis.global_.offheight = 540
                solver.mj_model.vis.headlight.ambient[:] = 0.5
                solver.mj_model.vis.headlight.diffuse[:] = 0.8
                # Plane size only limits its drawing; collision planes are infinite.
                # Keep the rendered floor large enough that a fall never looks like an edge fall.
                plane_geoms = solver.mj_model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
                solver.mj_model.geom_size[plane_geoms, :2] = 50
                renderer = mujoco.Renderer(solver.mj_model, height=540, width=960)
                render_data = solver.mj_data if args.mujoco_cpu else mujoco.MjData(solver.mj_model)
            try:
                cases = [(f, d, offset) for f in args.forces
                         for d in (args.directions if f else [args.directions[0]])
                         for offset in (range(args.num_envs) if args.mujoco_cpu else [0])]
                for magnitude, direction, offset in cases:
                    obs, _ = env.reset(seed=args.seed + offset)
                    fallen = np.zeros(batch_size, dtype=bool)
                    first_fall = np.full(batch_size, np.nan)
                    peak_tilt = np.zeros(batch_size)
                    min_height = np.full(batch_size, np.inf)
                    peak_speed = np.zeros(batch_size)
                    stable_steps = np.zeros(batch_size, dtype=int)
                    recovery = np.full(batch_size, np.nan)
                    initial_xy = robot.data.root_pos_w.torch[:, :2].cpu().numpy().copy()
                    reference_z = None
                    writer = None
                    video_index = 0 if args.mujoco_cpu else args.video_sample
                    if renderer is not None and offset == (args.video_sample if args.mujoco_cpu else 0):
                        import imageio.v2 as imageio
                        from PIL import Image, ImageDraw, ImageFont
                        writer = imageio.get_writer(str(args.output_dir / f"push_{magnitude:g}N_{direction.replace('+', 'plus').replace('-', 'minus')}.mp4"), fps=30, codec="libx264", macro_block_size=2)
                        if args.video_renderer == "mujoco":
                            camera = mujoco.MjvCamera()
                            camera.distance, camera.azimuth, camera.elevation = 3.8, 135, -20
                        else:
                            rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*DIRECTIONS[direction])).GetQuat()
                            arrow_rotation.Set(Gf.Quatf(rotation.GetReal(), Gf.Vec3f(rotation.GetImaginary())))
                            # Warm the RTX pipeline without advancing physics or policy time.
                            for _ in range(20):
                                env.sim.render()
                                renderer.render_rgb_array()
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
                    try:
                        for step in range(steps):
                            on = any(push_start <= step < push_start + duration for push_start in push_starts)
                            extra_on = bool(args.extra_pulse_force) and extra_start <= step < extra_start + extra_duration
                            applied_magnitude = (magnitude if on else 0.0) + (args.extra_pulse_force if extra_on else 0.0)
                            force[:] = torch.tensor(DIRECTIONS[direction], device=env.device) * applied_magnitude
                            robot.permanent_wrench_composer.set_forces_and_torques_index(
                                forces=force, torques=torque, body_ids=body_ids, is_global=True
                            )
                            with torch.inference_mode():
                                actions = policy(obs["policy"])
                                if not torch.isfinite(actions).all():
                                    raise RuntimeError("Nonfinite policy actions")
                                obs, _, _, _, _ = env.step(actions)
                            z = robot.data.root_pos_w.torch[:, 2].cpu().numpy().copy()
                            gravity = robot.data.projected_gravity_b.torch.cpu().numpy()
                            tilt = np.arccos(np.clip(-gravity[:, 2], -1, 1))
                            speed = robot.data.root_lin_vel_w.torch[:, :2].norm(dim=1).cpu().numpy()
                            angular_speed = robot.data.root_ang_vel_w.torch.norm(dim=1).cpu().numpy()
                            finite = np.isfinite(z) & np.isfinite(tilt) & np.isfinite(speed)
                            failed = (z < 0.7) | (tilt > 0.8) | ~finite
                            first_fall[failed & ~fallen] = (step + 1) * dt
                            fallen |= failed
                            if step == start - 1:
                                reference_z = z.copy()
                            if step >= start:
                                peak_tilt = np.maximum(peak_tilt, tilt)
                                min_height = np.minimum(min_height, z)
                                peak_speed = np.maximum(peak_speed, speed)
                            if step >= last_push_end:
                                stable = (~fallen & finite & (tilt < math.radians(5)) & (speed < 0.1)
                                          & (angular_speed < 0.2) & (np.abs(z - reference_z) < 0.03))
                                stable_steps = np.where(stable, stable_steps + 1, 0)
                                just_recovered = (stable_steps >= math.ceil(0.5 / dt)) & np.isnan(recovery)
                                recovery[just_recovered] = (step + 1 - last_push_end) * dt - 0.5
                            if step % 6 == 0 or step == steps - 1:
                                for sample in range(batch_size):
                                    traces.append({"force_N": magnitude, "direction": direction, "sample": offset + sample,
                                                   "time_s": (step + 1) * dt, "force_active": bool(applied_magnitude),
                                                   "applied_force_N": applied_magnitude,
                                                   "extra_pulse_active": extra_on,
                                                   "height_m": float(z[sample]), "tilt_deg": float(np.degrees(tilt[sample])),
                                                   "speed_m_s": float(speed[sample]), "fallen": bool(fallen[sample])})
                            if writer is not None and step % 2 == 1:
                                if args.video_renderer == "isaac":
                                    if applied_magnitude:
                                        position = robot.data.body_com_pos_w.torch[video_index, body_ids[0]].cpu().numpy().astype(float)
                                        origin = position - np.array(DIRECTIONS[direction]) * 0.65
                                        arrow_position.Set(Gf.Vec3d(*origin))
                                        arrow_root.MakeVisible()
                                    else:
                                        arrow_root.MakeInvisible()
                                    # Headless Kit capture does not make env.step() enter its
                                    # render loop. Sync the current Newton body poses to Fabric
                                    # before the RTX annotator captures this frame.
                                    env.sim.render()
                                    frame = Image.fromarray(renderer.render_rgb_array())
                                else:
                                    camera.lookat[:] = (*initial_xy[video_index], 0.85)
                                if args.video_renderer == "mujoco" and not args.mujoco_cpu:
                                    # Render the GPU solver's state without stepping a second simulation.
                                    render_data.qpos[:] = solver.mjw_data.qpos.numpy()[video_index]
                                    render_data.qvel[:] = solver.mjw_data.qvel.numpy()[video_index]
                                    mujoco.mj_forward(solver.mj_model, render_data)
                                if args.video_renderer == "mujoco":
                                    renderer.update_scene(render_data, camera=camera)
                                if args.video_renderer == "mujoco" and applied_magnitude:
                                    position = robot.data.body_com_pos_w.torch[video_index, body_ids[0]].cpu().numpy().astype(float)
                                    arrow = renderer.scene.geoms[renderer.scene.ngeom]
                                    mujoco.mjv_initGeom(arrow, mujoco.mjtGeom.mjGEOM_ARROW,
                                                       np.zeros(3), position, np.eye(3).ravel(),
                                                       np.array([1.0, 0.2, 0.1, 1.0], dtype=np.float32))
                                    mujoco.mjv_connector(arrow, mujoco.mjtGeom.mjGEOM_ARROW, 0.025,
                                                        position - np.array(DIRECTIONS[direction]) * 0.65, position)
                                    renderer.scene.ngeom += 1
                                if args.video_renderer == "mujoco":
                                    frame = Image.fromarray(renderer.render())
                                if not args.video_no_overlay:
                                    draw = ImageDraw.Draw(frame)
                                    draw.rectangle((0, 0, frame.width, 90), fill=(18, 23, 34))
                                    label = f"{args.video_label} | " if args.video_label else ""
                                    extra_label = " | + pulse" if extra_on else ""
                                    draw.text((20, 12), f"{label}{applied_magnitude:g} N {direction}{extra_label} | t={(step+1)*dt:.2f} s | {'PUSH ON' if applied_magnitude else 'force off'}", font=font, fill="white")
                                    draw.text((20, 48), f"tilt={np.degrees(tilt[video_index]):.1f} deg | speed={speed[video_index]:.2f} m/s | {'FALL DETECTED' if fallen[video_index] else 'standing'}", font=font, fill=(255, 110, 95) if fallen[video_index] else (120, 230, 175))
                                writer.append_data(np.asarray(frame))
                    finally:
                        if writer is not None:
                            writer.close()
                        robot.permanent_wrench_composer.reset()
                    displacement = np.linalg.norm(robot.data.root_pos_w.torch[:, :2].cpu().numpy() - initial_xy, axis=1)
                    final_stable = ~fallen & (stable_steps >= math.ceil(0.5 / dt))
                    for sample in range(batch_size):
                        rows.append({"force_N": magnitude, "direction": direction, "sample": offset + sample,
                                     "survived": bool(~fallen[sample]), "recovered_at_end": bool(final_stable[sample]),
                                     "first_fall_s": None if np.isnan(first_fall[sample]) else float(first_fall[sample]),
                                     "recovery_s": None if np.isnan(recovery[sample]) or fallen[sample] else float(max(0, recovery[sample])),
                                     "max_tilt_deg": float(np.degrees(peak_tilt[sample])), "min_height_m": float(min_height[sample]),
                                     "peak_speed_m_s": float(peak_speed[sample]), "final_displacement_m": float(displacement[sample])})
                    group = [r for r in rows if r["force_N"] == magnitude and r["direction"] == direction]
                    recovered_times = [r["recovery_s"] for r in group if r["recovered_at_end"]]
                    summary = {"force_N": magnitude, "direction": direction, "trials": len(group),
                               "survived": sum(r["survived"] for r in group),
                               "recovered": sum(r["recovered_at_end"] for r in group),
                               "max_tilt_deg": max(r["max_tilt_deg"] for r in group),
                               "mean_recovery_s": float(np.mean(recovered_times)) if recovered_times else None}
                    summaries = [s for s in summaries if (s["force_N"], s["direction"]) != (magnitude, direction)]
                    summaries.append(summary)
                    if len(group) == args.num_envs:
                        print(json.dumps(summary), flush=True)
                    metadata = {"task": "Nest-Stand-Grasshopper", "policy": str(args.policy.resolve()),
                                "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                                "solver": "Newton/MuJoCo CPU" if args.mujoco_cpu else "Newton/MJWarp",
                                "device": str(env.device), "seed": args.seed, "robot_mass_kg": mass,
                                "samples_per_case": args.num_envs, "serial_reset_seeds":
                                list(range(args.seed, args.seed + args.num_envs)) if args.mujoco_cpu else None,
                                "video_sample": args.video_sample if args.video else None,
                                "video_renderer": args.video_renderer if args.video else None,
                                "video_overlay": not args.video_no_overlay if args.video else None,
                                "camera_wide": args.camera_wide if args.video else None,
                                "body": body_names[0], "force_frame": "world", "force_point": "torso center of mass",
                                "control_dt_s": dt, "push_start_s": start * dt, "push_duration_s": duration * dt,
                                "push_times_s": [s * dt for s in push_starts],
                                "extra_pulse": {"force_N": args.extra_pulse_force,
                                                "start_s": extra_start * dt, "duration_s": extra_duration * dt,
                                                "direction": "same as main force"},
                                "trial_duration_s": steps * dt, "fall_thresholds": {"height_m": 0.7, "tilt_rad": 0.8},
                                "recovery_thresholds": {"tilt_deg": 5, "horizontal_speed_m_s": 0.1,
                                                       "angular_speed_rad_s": 0.2, "height_error_m": 0.03, "stable_duration_s": 0.5}}
                    (args.output_dir / "results.json").write_text(json.dumps({"metadata": metadata, "summary": summaries, "trials": rows}, indent=2) + "\n")
            finally:
                if renderer is not None and args.video_renderer == "mujoco":
                    renderer.close()
    for name, data in (("trials.csv", rows), ("trajectories.csv", traces)):
        with (args.output_dir / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)


if __name__ == "__main__":
    main()
