#!/usr/bin/env python3
"""Replay current or historical walking policies with optional RTX recording."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nest-walk-replay-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/nest-walk-replay-cache")
import warp as wp

wp.config.enable_backward = False
wp.config.kernel_cache_dir = "/tmp/nest-walk-warp-cache"

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "logs/rsl_rl/grasshopper_walk_flat_1ms/2026-10-19_4096env_5000iter"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=RUN / "exports/iteration_05000/policy.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speeds", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--yaw-rates", type=float, nargs="+", default=[0.0],
                        help="Body yaw commands in rad/s; evaluate every speed/yaw pair")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-yaw", type=float, help="Fix the initial heading in radians for comparable views")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--capture-states", action="store_true", help="Save simulated robot poses for an independent Isaac RTX render")
    parser.add_argument("--cpu-mujoco", action="store_true", help="Single-robot native MuJoCo CPU replay; supports Mesa software rendering")
    parser.add_argument("--export-mjcf", action="store_true", help="Save the nominal compiled MuJoCo model for standalone policy testing")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.seconds <= 2 or any(not 0 <= s <= 1 for s in args.speeds):
        parser.error("Use positive num-envs, seconds > 2 and speed commands in [0, 1] m/s")
    if any(not -0.7 <= yaw <= 0.7 for yaw in args.yaw_rates):
        parser.error("Use yaw-rate commands in [-0.7, 0.7] rad/s")
    if args.video and not args.cpu_mujoco:
        parser.error("Direct Kit recording omits the Newton robot. Use --capture-states, then render_grasshopper_walk_replay.py")

    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import resolve_task_config
    from isaaclab.utils.io import dump_yaml
    import nest.tasks  # noqa: F401

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cfg, _ = resolve_task_config("Nest-Walk-Flat-Grasshopper", None, overrides=["physics=newton_mjwarp"])
    policy = torch.jit.load(str(args.policy), map_location="cpu").eval()
    wm_policy = "enc.0.weight" in policy.state_dict() or "obs_norm.mean" in policy.state_dict()
    if wm_policy:
        from nest.mbrl.supermimic.inference import HistoryPolicy
        metadata = json.loads(args.policy.with_suffix('.json').read_text())
        expected_history = {'nest_grasshopper_reference_v3': 10, 'nest_grasshopper_small_mlp_bptt': 5}.get(metadata['architecture']['variant'])
        if (expected_history is None
                or metadata['input_history'] != [expected_history, 51] or metadata['input_goal'] != 3):
            raise ValueError('Unsupported WM actor schema')
        policy = HistoryPolicy(policy, history=expected_history)
        policy_size = 54
    else:
        policy_size = policy.state_dict()["mlp.0.weight"].shape[1]
    if policy_size == 55:
        cfg.use_legacy_policy_observations()
    elif policy_size != 54:
        parser.error(f"Expected a 54- or 55-input walking actor, got {policy_size}")
    cfg.play_mode()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 8.0
    cfg.seed = args.seed
    if args.initial_yaw is not None:
        cfg.events.reset_base.params["pose_range"]["yaw"] = (args.initial_yaw, args.initial_yaw)
    cfg.sim.device = args.device or "cuda:0"
    cfg.sim.visualizer_cfgs = []
    if args.export_mjcf:
        if args.num_envs != 1:
            parser.error("Export a single environment with --num-envs 1")
        cfg.sim.physics.solver_cfg.save_to_mjcf = str(output / "training_nominal.xml")
    cfg.terminations.time_out = None
    cfg.terminations.root_height = None
    cfg.terminations.bad_orientation = None
    cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.scene.dome_light.spawn.intensity = 650.0
    cfg.scene.dome_light.spawn.color = (0.95, 0.97, 1.0)
    if args.cpu_mujoco:
        from isaaclab_newton.physics import NewtonManager, NewtonMJWarpManager

        if args.num_envs != 1:
            parser.error("Native CPU replay requires --num-envs 1")

        class NewtonCpuReplayManager(NewtonMJWarpManager):
            @classmethod
            def _initialize_contacts(cls):
                if cls._report_contacts:
                    raise RuntimeError("Native CPU contacts are not available to Isaac Lab sensors")
                NewtonManager._contacts = None

        cfg.sim.device = "cpu"
        cfg.scene.feet_contact = None
        cfg.scene.robot.spawn.activate_contact_sensors = False
        cfg.rewards.foot_swing_height = None
        cfg.rewards.foot_slip = None
        cfg.rewards.soft_landing = None
        cfg.observations.critic.foot_air_time = None
        cfg.observations.critic.foot_contact = None
        cfg.observations.critic.foot_contact_forces = None
        cfg.sim.physics.solver_cfg.use_mujoco_cpu = True
        cfg.sim.physics.solver_cfg.class_type = NewtonCpuReplayManager
        cfg.sim.physics.class_type = NewtonCpuReplayManager
    if args.video and not args.cpu_mujoco:
        from isaaclab_visualizers.kit import KitVisualizerCfg

        args.headless = True
        args.enable_cameras = True
        args.renderer = "RayTracedLighting"
        cfg.sim.visualizer_cfgs = [KitVisualizerCfg(
            headless=True, window_width=args.width, window_height=args.height,
            origin_type="world", visible_env_indices=[0],
            eye=(3.5, -3.5, 2.3), lookat=(0.0, 0.0, 0.8),
            focal_length=26.0, background_color=(0.18, 0.22, 0.28))]
    cfg.validate()
    dump_yaml(str(output / "env.yaml"), cfg)
    policy = policy.to(cfg.sim.device)
    with torch.inference_mode():
        assert policy(torch.zeros((args.num_envs, policy_size), device=cfg.sim.device)).shape == (args.num_envs, 14)
    report = {
        "status": "initializing", "task": "Nest-Walk-Flat-Grasshopper",
        "policy": str(args.policy.resolve()),
        "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        "policy_type": metadata["architecture"]["variant"] if wm_policy else "ppo_mlp",
        "physics": "Newton native MuJoCo CPU" if args.cpu_mujoco else "Isaac Lab Newton/MJWarp",
        "renderer": ("MuJoCo software OpenGL" if args.cpu_mujoco else "Isaac Sim RTX") if args.video else None,
        "num_envs": args.num_envs, "seed": args.seed, "seconds_per_speed": args.seconds,
        "speeds_m_s": args.speeds, "observation_size": policy_size,
        "yaw_rates_rad_s": args.yaw_rates,
        "initial_yaw_rad": args.initial_yaw,
        "velocity_measurement": "Root-link linear velocity and body-frame root-link yaw rate, matching tracking rewards",
        "auto_reset": False, "video_sample_index": 0 if args.video else None,
        "fall_criteria": {"root_height_m": 0.7, "tilt_rad": 0.8}, "cases": [],
        "note": "Playback uses the current robot contacts and task physics; historical actors use their legacy observation/action layout.",
    }
    if args.cpu_mujoco:
        report["cpu_replay_note"] = f"Native CPU solver uses {policy_size} policy observations and 14 policy actions. Foot contact sensors and their unused replay rewards are disabled; physical ground contacts remain enabled."

    def save_report():
        (output / "replay.json").write_text(json.dumps(report, indent=2) + "\n")

    save_report()
    started = time.monotonic()
    traces = []
    captured_positions = []
    captured_quaternions = []
    captured_joints = []
    captured_commands = []
    captured_yaw_commands = []
    captured_yaw_velocities = []
    captured_body_velocities = []
    captured_linear_velocities_w = []
    captured_linear_velocities_b = []
    captured_case_indices = []
    captured_velocities = []
    captured_times = []
    captured_foot_heights = []
    captured_foot_contacts = []
    try:
        with launch_simulation(cfg, args):
            with contextlib.closing(gym.make(report["task"], cfg=cfg)) as wrapped:
                env = wrapped.unwrapped
                env.render_enabled = False
                robot = env.scene["robot"]
                if args.export_mjcf:
                    import mujoco
                    from isaaclab_newton.physics import NewtonManager
                    if args.num_envs != 1:
                        raise ValueError("Export a single environment with --num-envs 1")
                    path = output / "training_nominal.mjb"
                    # Newton builds an MjSpec directly, so mj_saveLastXML has
                    # no loaded XML. Save its XML via save_to_mjcf above and
                    # snapshot the live compiled parameters in a native MJB.
                    NewtonManager._solver.mj_model.opt.timestep = cfg.sim.dt / cfg.sim.physics.num_substeps
                    mujoco.mj_saveModel(NewtonManager._solver.mj_model, str(path))
                    report["exported_mjcf"] = str(output / "training_nominal.xml")
                    report["exported_native_model"] = str(path)
                if args.capture_states:
                    from nest.tasks.velocity.mdp.rewards import _foot_site_positions
                    foot_params = env.reward_manager.get_term_cfg("foot_clearance").params
                    foot_sensor_cfg = None if args.cpu_mujoco else env.reward_manager.get_term_cfg("foot_slip").params["sensor_cfg"]
                renderer = None
                if args.video:
                    import imageio.v2 as imageio
                    from PIL import Image, ImageDraw, ImageFont
                    if args.cpu_mujoco:
                        import mujoco
                        from OpenGL import GL
                        from isaaclab_newton.physics import NewtonManager

                        solver = NewtonManager._solver
                        solver.mj_model.vis.global_.offwidth = args.width
                        solver.mj_model.vis.global_.offheight = args.height
                        solver.mj_model.vis.headlight.ambient[:] = 0.5
                        solver.mj_model.vis.headlight.diffuse[:] = 0.8
                        renderer = mujoco.Renderer(solver.mj_model, height=args.height, width=args.width)
                        camera = mujoco.MjvCamera()
                        camera.distance, camera.azimuth, camera.elevation = 3.2, 45, -20
                        report["opengl_renderer"] = GL.glGetString(GL.GL_RENDERER).decode()
                    else:
                        from pxr import Gf, UsdGeom, UsdLux

                        renderer = next(v for v in env.sim._visualizers if v.cfg.visualizer_type == "kit")
                        stage = renderer._scene_data_provider.usd_stage
                        sun = UsdLux.DistantLight.Define(stage, "/World/WalkReplaySun")
                        sun.CreateIntensityAttr(1200.0)
                        UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45, 25, -30))
                        origin = env.scene.env_origins[0].cpu().numpy()
                        # World-fixed visual markers make travel visible with a following camera.
                        for index in range(-5, 36):
                            line = UsdGeom.Cube.Define(stage, f"/World/WalkReplayGrid/Line_{index + 5}")
                            line.CreateDisplayColorAttr([Gf.Vec3f(0.42, 0.46, 0.50)])
                            xf = UsdGeom.Xformable(line.GetPrim())
                            xf.AddTranslateOp().Set(Gf.Vec3d(float(origin[0]) + index, float(origin[1]), 0.001))
                            xf.AddScaleOp().Set(Gf.Vec3f(0.007, 2.0, 0.001))
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 21)

                command = env.command_manager.get_term("base_velocity")
                dt = env.step_dt
                steps = round(args.seconds / dt)
                stride = 2
                video_fps = 1.0 / (stride * dt)
                for case_index, (speed, yaw_rate) in enumerate(itertools.product(args.speeds, args.yaw_rates)):
                    command.cfg.ranges.lin_vel_x = (speed, speed)
                    command.cfg.ranges.ang_vel_z = (yaw_rate, yaw_rate)
                    obs, _ = env.reset(seed=args.seed)
                    if wm_policy:
                        policy.reset()
                    assert obs["policy"].shape == (args.num_envs, policy_size)
                    if not torch.allclose(command.command[:, 0], torch.full_like(command.command[:, 0], speed)):
                        raise RuntimeError("The requested fixed velocity command was not applied")
                    if not torch.allclose(command.command[:, 2], torch.full_like(command.command[:, 2], yaw_rate)):
                        raise RuntimeError("The requested fixed yaw command was not applied")
                    fallen = np.zeros(args.num_envs, dtype=bool)
                    first_fall = np.full(args.num_envs, np.nan)
                    peak_tilt = np.zeros(args.num_envs)
                    min_height = np.full(args.num_envs, np.inf)
                    speed_sum = np.zeros(args.num_envs)
                    speed_count = np.zeros(args.num_envs, dtype=int)
                    xy_error_sum = np.zeros(args.num_envs)
                    yaw_error_sum = np.zeros(args.num_envs)
                    yaw_sum = np.zeros(args.num_envs)
                    body_speed_sum = np.zeros(args.num_envs)
                    initial_xy = robot.data.root_pos_w.torch[:, :2].cpu().numpy().copy()
                    writer = None
                    frames = 0
                    case_name = f"walk_{speed:g}ms" + (f"_yaw{yaw_rate:g}" if yaw_rate else "")
                    video_path = output / f"{case_name}_{'mujoco_cpu' if args.cpu_mujoco else 'isaac'}.mp4"
                    if renderer is not None:
                        position = robot.data.root_pos_w.torch[0].cpu().numpy()
                        if args.cpu_mujoco:
                            camera.lookat[:] = (float(position[0]), float(position[1]), 0.8)
                            renderer.update_scene(solver.mj_data, camera=camera)
                            renderer.render()
                        else:
                            renderer.set_camera_view((float(position[0]) + 3.5, float(position[1]) - 3.5, 2.3),
                                                     (float(position[0]), float(position[1]), 0.8))
                            for _ in range(20):
                                env.sim.render()
                                renderer.render_rgb_array()
                        writer = imageio.get_writer(str(video_path), fps=video_fps, codec="libx264",
                                                    quality=8, macro_block_size=2,
                                                    ffmpeg_params=["-movflags", "+faststart"])
                    try:
                        for step in range(steps):
                            with torch.inference_mode():
                                actions = policy(obs["policy"])
                                if not torch.isfinite(actions).all():
                                    raise RuntimeError("Nonfinite policy action")
                                obs, _, _, _, _ = env.step(actions)
                            pos = robot.data.root_pos_w.torch.cpu().numpy().copy()
                            velocity = robot.data.root_link_lin_vel_w.torch.cpu().numpy().copy()
                            body_velocity = robot.data.root_link_lin_vel_b.torch.cpu().numpy()
                            body_yaw_rate = robot.data.root_link_ang_vel_b.torch[:, 2].cpu().numpy()
                            xy_error = np.linalg.norm(body_velocity[:, :2] - np.array([speed, 0.0]), axis=1)
                            gravity = robot.data.projected_gravity_b.torch.cpu().numpy()
                            tilt = np.arccos(np.clip(-gravity[:, 2], -1.0, 1.0))
                            finite = np.isfinite(pos).all(axis=1) & np.isfinite(velocity).all(axis=1)
                            failed = (pos[:, 2] < 0.7) | (tilt > 0.8) | ~finite
                            first_fall[failed & ~fallen] = (step + 1) * dt
                            fallen |= failed
                            peak_tilt = np.maximum(peak_tilt, tilt)
                            min_height = np.minimum(min_height, pos[:, 2])
                            valid = ~fallen & ((step + 1) * dt >= 2.0)
                            speed_sum[valid] += velocity[valid, 0]
                            speed_count[valid] += 1
                            xy_error_sum[valid] += xy_error[valid]
                            yaw_error_sum[valid] += np.abs(body_yaw_rate[valid] - yaw_rate)
                            yaw_sum[valid] += body_yaw_rate[valid]
                            body_speed_sum[valid] += body_velocity[valid, 0]
                            if step % 6 == 5:
                                for sample in range(args.num_envs):
                                    traces.append({"command_m_s": speed, "sample": sample,
                                                   "command_yaw_rad_s": yaw_rate,
                                                   "time_s": (step + 1) * dt, "x_m": float(pos[sample, 0]),
                                                   "y_m": float(pos[sample, 1]), "height_m": float(pos[sample, 2]),
                                                   "forward_velocity_m_s": float(velocity[sample, 0]),
                                                   "body_forward_velocity_m_s": float(body_velocity[sample, 0]),
                                                   "xy_tracking_error_m_s": float(xy_error[sample]),
                                                   "yaw_rate_rad_s": float(body_yaw_rate[sample]),
                                                   "tilt_deg": float(np.degrees(tilt[sample])),
                                                   "fallen": bool(fallen[sample])})
                            if args.capture_states and step % stride == stride - 1:
                                captured_positions.append(pos[0].copy())
                                captured_quaternions.append(robot.data.root_quat_w.torch[0].cpu().numpy().copy())
                                captured_joints.append(robot.data.joint_pos.torch[0].cpu().numpy().copy())
                                captured_commands.append(speed)
                                captured_yaw_commands.append(yaw_rate)
                                captured_yaw_velocities.append(body_yaw_rate[0])
                                captured_body_velocities.append(body_velocity[0, 0])
                                captured_linear_velocities_w.append(velocity[0].copy())
                                captured_linear_velocities_b.append(body_velocity[0].copy())
                                captured_case_indices.append(case_index)
                                captured_velocities.append(velocity[0, 0])
                                captured_times.append((step + 1) * dt)
                                sites = _foot_site_positions(env, foot_params["asset_cfg"], foot_params["foot_offsets"])
                                captured_foot_heights.append(sites[0, :, 2].cpu().numpy().copy())
                                if foot_sensor_cfg is not None:
                                    force = env.scene[foot_sensor_cfg.name].data.net_normal_forces_w.torch[0, foot_sensor_cfg.body_ids]
                                    captured_foot_contacts.append((force.norm(dim=-1) > 0).cpu().numpy().copy())
                                else:
                                    captured_foot_contacts.append(np.full(2, np.nan))
                            if writer is not None and step % stride == stride - 1:
                                if args.cpu_mujoco:
                                    camera.lookat[:] = (float(pos[0, 0]), float(pos[0, 1]), 0.8)
                                    renderer.update_scene(solver.mj_data, camera=camera)
                                    # Rendering-only ground references; they do not enter physics.
                                    for axis, indices in ((0, range(-5, 36)), (1, range(-5, 6))):
                                        for index in indices:
                                            geom = renderer.scene.geoms[renderer.scene.ngeom]
                                            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE,
                                                               np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                                               np.array([0.75, 0.78, 0.82, 1.0]))
                                            start = np.array([index, -5, 0.003]) if axis == 0 else np.array([-5, index, 0.003])
                                            end = np.array([index, 5, 0.003]) if axis == 0 else np.array([35, index, 0.003])
                                            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 1.0, start, end)
                                            renderer.scene.ngeom += 1
                                    pixels = renderer.render()
                                else:
                                    renderer.set_camera_view((float(pos[0, 0]) + 3.5, float(pos[0, 1]) - 3.5, 2.3),
                                                             (float(pos[0, 0]), float(pos[0, 1]), 0.8))
                                    env.sim.render()
                                    pixels = renderer.render_rgb_array()
                                if pixels.size == 0 or float(pixels.std()) < 8:
                                    raise RuntimeError("Isaac RTX returned an empty or blank frame")
                                frame = Image.fromarray(pixels)
                                draw = ImageDraw.Draw(frame)
                                draw.rectangle((0, 0, args.width, 68), fill=(20, 26, 33))
                                draw.text((18, 7), f"NEST WALK | vx {body_velocity[0, 0]:.2f}/{speed:.2f} m/s | yaw {body_yaw_rate[0]:+.2f}/{yaw_rate:+.2f} rad/s", font=font, fill="white")
                                draw.text((18, 36), f"t={(step + 1)*dt:.2f}s | tilt={np.degrees(tilt[0]):.1f} deg | {'FALL DETECTED' if fallen[0] else 'walking'}",
                                          font=font, fill=(255, 130, 110) if fallen[0] else (130, 235, 180))
                                writer.append_data(np.asarray(frame))
                                frames += 1
                                if frames in (30, 120):
                                    frame.save(output / f"{case_name}_frame_{frames:04d}.png")
                            if (step + 1) % 120 == 0:
                                print(f"WALK_REPLAY speed={speed:g} yaw={yaw_rate:g} t={(step+1)*dt:.1f}/{args.seconds:g}s survived={(~fallen).sum()}/{args.num_envs}", flush=True)
                    finally:
                        if writer is not None:
                            writer.close()
                    final_pos = robot.data.root_pos_w.torch.cpu().numpy()
                    rows = [{"sample": i, "survived": not bool(fallen[i]),
                             "first_fall_s": None if np.isnan(first_fall[i]) else float(first_fall[i]),
                             "forward_displacement_m": float(final_pos[i, 0] - initial_xy[i, 0]),
                             "lateral_displacement_m": float(final_pos[i, 1] - initial_xy[i, 1]),
                             "mean_forward_velocity_after_2s_m_s": float(speed_sum[i]/speed_count[i]) if speed_count[i] else None,
                             "mean_body_forward_velocity_after_2s_m_s": float(body_speed_sum[i]/speed_count[i]) if speed_count[i] else None,
                             "mean_yaw_rate_after_2s_rad_s": float(yaw_sum[i]/speed_count[i]) if speed_count[i] else None,
                             "mean_xy_tracking_error_after_2s_m_s": float(xy_error_sum[i]/speed_count[i]) if speed_count[i] else None,
                             "mean_yaw_tracking_error_after_2s_rad_s": float(yaw_error_sum[i]/speed_count[i]) if speed_count[i] else None,
                             "max_tilt_deg": float(np.degrees(peak_tilt[i])), "min_root_height_m": float(min_height[i])}
                            for i in range(args.num_envs)]
                    case = {"command_m_s": speed, "survived": int((~fallen).sum()),
                            "command_yaw_rad_s": yaw_rate,
                            "samples": rows, "video": str(video_path) if writer is not None else None,
                            "video_frames": frames, "video_fps": video_fps if frames else None}
                    if writer is not None and (frames != steps // stride or video_path.stat().st_size < 50000):
                        raise RuntimeError("Incomplete replay recording")
                    report["cases"].append(case)
                    save_report()
                if args.capture_states:
                    state_path = output / "walking_states.npz"
                    np.savez_compressed(state_path,
                                        root_pos_w=np.asarray(captured_positions),
                                        root_quat_w=np.asarray(captured_quaternions),
                                        joint_pos=np.asarray(captured_joints),
                                        joint_names=np.asarray(robot.joint_names),
                                        command_m_s=np.asarray(captured_commands),
                                        command_yaw_rad_s=np.asarray(captured_yaw_commands),
                                        yaw_rate_rad_s=np.asarray(captured_yaw_velocities),
                                        body_forward_velocity_m_s=np.asarray(captured_body_velocities),
                                        linear_velocity_w_m_s=np.asarray(captured_linear_velocities_w),
                                        linear_velocity_b_m_s=np.asarray(captured_linear_velocities_b),
                                        case_index=np.asarray(captured_case_indices),
                                        forward_velocity_m_s=np.asarray(captured_velocities),
                                        time_s=np.asarray(captured_times),
                                        foot_height_m=np.asarray(captured_foot_heights),
                                        foot_contact=np.asarray(captured_foot_contacts),
                                        foot_names=np.asarray(foot_params["asset_cfg"].body_names),
                                        physics_backend=report["physics"],
                                        fps=video_fps)
                    report["state_trace"] = str(state_path)
                    report["state_frames"] = len(captured_positions)
                with (output / "trace.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(traces[0]))
                    writer.writeheader()
                    writer.writerows(traces)
                report.update(status="complete", elapsed_seconds=round(time.monotonic()-started, 2))
                save_report()
                if args.cpu_mujoco and renderer is not None:
                    renderer.close()
        print(f"WALK_REPLAY_COMPLETE output={output}", flush=True)
    except BaseException as error:
        import traceback
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        save_report()
        raise


if __name__ == "__main__":
    main()
