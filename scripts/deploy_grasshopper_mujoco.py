#!/usr/bin/env python3
"""Run the exported standing policy directly in native MuJoCo, without Isaac Lab."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

if "--viewer" not in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
import yaml

if __package__:
    from .urdf_mjcf_dynamics import apply_joint_damping, read_joint_damping
else:
    from urdf_mjcf_dynamics import apply_joint_damping, read_joint_damping


ROOT = Path(__file__).resolve().parents[1]
TRAINING_RUN = ROOT / "logs/rsl_rl/grasshopper_stand/2026-09-24_18-54-51"
PREFIX = "grasshopper_lower_body_"


class TrainingConfigLoader(yaml.SafeLoader):
    """Allow the two inert data tags emitted by Isaac Lab's config dumper."""


TrainingConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node))
)
TrainingConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice",
    lambda loader, node: slice(*loader.construct_sequence(node)),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=TRAINING_RUN / "exported/policy.pt")
    parser.add_argument("--training-config", type=Path, default=TRAINING_RUN / "params/env.yaml")
    parser.add_argument("--robot", type=Path, default=ROOT / "outputs/grasshopper_torso/mjcf/grasshopper_lower_body.xml")
    parser.add_argument("--urdf", type=Path, default=ROOT / "assets/robots/grasshopper_torso/grasshopper_lower_body.urdf")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/grasshopper_torso/mujoco_deployment")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-trial", type=int, default=0, help="Trial index to record; 0 uses the nominal pose.")
    parser.add_argument("--viewer", action="store_true", help="Open the native MuJoCo viewer in real time.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    if args.seconds <= 0 or args.trials < 1:
        parser.error("seconds and trials must be positive")
    if not 0 <= args.video_trial < args.trials:
        parser.error("video-trial must be between 0 and trials - 1")
    return args


def load_deployment_config(path, urdf_path=None):
    if urdf_path is None:
        urdf_path = ROOT / "assets/robots/grasshopper_torso/grasshopper_lower_body.urdf"
    cfg = yaml.load(path.read_text(), Loader=TrainingConfigLoader)
    action = cfg["actions"]["joint_pos"]
    observed = cfg["observations"]["policy"]
    terms = [key for key, value in observed.items() if isinstance(value, dict) and "func" in value]
    expected = ["base_height", "base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions"]
    if terms != expected or any(observed[name].get("scale") is not None or observed[name].get("clip") is not None for name in terms):
        raise ValueError("The deployment expects the unscaled 52-dimensional standing observations")
    if not action["preserve_order"] or not action["use_default_offset"] or action.get("clip") is not None:
        raise ValueError("Unsupported action ordering, offset, or clipping configuration")
    robot = cfg["scene"]["robot"]
    actuator = robot["actuators"]["driven_joints"]
    sim = cfg["sim"]
    names = action["joint_names"]
    control_dt = sim["dt"] * cfg["decimation"]
    physics_dt = sim["dt"] / sim["physics"]["num_substeps"]
    return {
        "joint_names": names, "default_joint_pos": robot["init_state"]["joint_pos"],
        "source_urdf": str(urdf_path.resolve()),
        "joint_passive_damping": read_joint_damping(urdf_path),
        "default_root_pos": list(robot["init_state"]["pos"]),
        "default_root_quat_xyzw": list(robot["init_state"]["rot"]),
        "kp": [actuator["stiffness"][name] for name in names],
        "kd": [actuator["damping"][name] for name in names],
        "effort_limits": [actuator["joint_effort_limit"][name] for name in names],
        "action_scale": action["scale"], "observation_terms": terms,
        "control_dt": control_dt, "physics_dt": physics_dt,
        "steps_per_action": round(control_dt / physics_dt),
        "gravity": list(sim["gravity"]),
        "solver_options": sim["physics"]["solver_cfg"],
        "shape_defaults": sim["physics"]["default_shape_cfg"],
        "ground_friction": sim["physics_material"]["dynamic_friction"],
        "reset_pose_range": cfg["events"]["reset_base"]["params"]["pose_range"],
        "root_body": PREFIX + "PELVIS",
        "fall_height": cfg["terminations"]["root_height"]["params"]["minimum_height"],
        "fall_tilt": cfg["terminations"]["bad_orientation"]["params"]["limit_angle"],
    }


def prepare_model(robot_path, output_path, cfg):
    tree = ET.parse(robot_path)
    root = tree.getroot()
    apply_joint_damping(root, cfg["joint_passive_damping"], PREFIX)
    actuator = root.find("actuator")
    if actuator is not None:
        root.remove(actuator)
    actuator = ET.SubElement(root, "actuator")
    for name, kp, kd, effort in zip(cfg["joint_names"], cfg["kp"], cfg["kd"], cfg["effort_limits"], strict=True):
        ET.SubElement(actuator, "position", name="policy_" + name, joint=PREFIX + name,
                      kp=str(kp), kv=str(kd), forcelimited="true", forcerange=f"{-effort} {effort}")
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(cfg["physics_dt"]))
    option.set("gravity", " ".join(map(str, cfg["gravity"])))
    for key in ("integrator", "solver", "iterations", "ls_iterations", "tolerance", "cone", "impratio", "ccd_iterations"):
        value = cfg["solver_options"][key]
        option.set(key, str(value).title() if key == "solver" else str(value))
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("autoreset", "disable")
    for geom in root.findall("worldbody//geom"):
        if geom.get("group") == "3":
            # Training USD disables self-collision. Robot-ground contact remains enabled.
            geom.set("contype", "1")
            geom.set("conaffinity", "2")
            geom.set("margin", str(cfg["shape_defaults"]["margin"]))
            geom.set("gap", str(cfg["shape_defaults"]["gap"]))
    for joint in root.findall("worldbody//joint"):
        if joint.get("type", "hinge") != "free":
            # Training USD's MjcJointAPI has no authored limit response.
            joint.set("solreflimit", "0.02 1")
    world = root.find("worldbody")
    asset = root.find("asset")
    ET.SubElement(asset, "texture", name="deployment_ground_texture", type="2d", builtin="checker",
                  rgb1="0.25 0.27 0.30", rgb2="0.45 0.47 0.50", width="512", height="512")
    ET.SubElement(asset, "material", name="deployment_ground_material", texture="deployment_ground_texture",
                  texrepeat="2 2", texuniform="true", reflectance="0.05", shininess="0.2")
    ET.SubElement(asset, "texture", name="deployment_sky", type="skybox", builtin="gradient",
                  rgb1="0.5 0.6 0.7", rgb2="0.2 0.25 0.3", width="512", height="3072")
    ET.SubElement(world, "geom", name="ground", type="plane", size="0 0 0.1",
                  material="deployment_ground_material",
                  friction=f"{cfg['ground_friction']} 0.005 0.0001",
                  contype="2", conaffinity="1", group="0",
                  margin=str(cfg["shape_defaults"]["margin"]), gap=str(cfg["shape_defaults"]["gap"]))
    ET.SubElement(world, "light", directional="true", pos="-2 -3 4", dir="0.4 0.4 -1",
                  diffuse="0.65 0.65 0.65", castshadow="true")
    ET.SubElement(world, "light", directional="true", pos="2 2 3", dir="-0.4 -0.4 -1",
                  diffuse="0.25 0.25 0.25", castshadow="false")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="2048", offheight="2048")
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="4")
    ET.SubElement(visual, "headlight", active="1", ambient="0.3 0.3 0.3",
                  diffuse="0.5 0.5 0.5", specular="0.1 0.1 0.1")
    ET.SubElement(root, "statistic", center="0 0 0.9", extent="1.7")
    ET.indent(tree)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return mujoco.MjModel.from_xml_path(str(output_path))


def policy_addresses(model, cfg):
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PREFIX + name) for name in cfg["joint_names"]]
    if any(index < 0 for index in ids):
        raise ValueError("Some policy joints are absent from the MuJoCo model")
    return model.jnt_qposadr[ids], model.jnt_dofadr[ids]


def reset(model, data, cfg, trial):
    mujoco.mj_resetData(model, data)
    for name, value in cfg["default_joint_pos"].items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PREFIX + name)
        if joint < 0:
            raise ValueError(f"Missing joint {name}")
        data.qpos[model.jnt_qposadr[joint]] = value
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["root_body"])
    quat_xyzw = np.asarray(cfg["default_root_quat_xyzw"])
    data.qpos[3:7] = quat_xyzw[[3, 0, 1, 2]]
    target_pos = np.array(cfg["default_root_pos"], dtype=float)
    if trial > 0:
        rng = np.random.default_rng(cfg["seed"] + trial)
        bounds = cfg["reset_pose_range"]
        target_pos[:2] += [rng.uniform(*bounds[axis]) for axis in ("x", "y")]
        euler = np.array([rng.uniform(*bounds[axis]) for axis in ("roll", "pitch", "yaw")])
        delta = np.empty(4)
        mujoco.mju_euler2Quat(delta, euler, "xyz")
        initial = data.qpos[3:7].copy()
        mujoco.mju_mulQuat(data.qpos[3:7], delta, initial)
    mujoco.mj_forward(model, data)
    # URDF has a massless base_link 0.1714 m above PELVIS. The policy uses PELVIS.
    data.qpos[:3] += target_pos - data.xpos[root]
    data.ctrl[:] = [cfg["default_joint_pos"][name] for name in cfg["joint_names"]]
    mujoco.mj_forward(model, data)
    return root


def observation(model, data, root, qpos_ids, dof_ids, default_pos, last_action):
    rotation = data.xmat[root].reshape(3, 3)
    velocity = np.empty(6)
    # Isaac Lab observes root COM velocity in the root link's orientation.
    # BODY/local would instead use the principal-inertia orientation.
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, root, velocity, 0)
    gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
    obs = np.concatenate(([data.xpos[root, 2]], rotation.T @ velocity[3:], rotation.T @ velocity[:3],
                          gravity, data.qpos[qpos_ids] - default_pos, data.qvel[dof_ids], last_action))
    if obs.shape != (52,) or not np.isfinite(obs).all():
        raise RuntimeError("Invalid standing-policy observation")
    return obs.astype(np.float32), gravity


def run_trial(model, policy, cfg, args, trial):
    data = mujoco.MjData(model)
    root = reset(model, data, cfg, trial)
    qpos_ids, dof_ids = policy_addresses(model, cfg)
    default_pos = np.array([cfg["default_joint_pos"][name] for name in cfg["joint_names"]])
    last_action = np.zeros(14)
    initial_xy = data.xpos[root, :2].copy()
    trace = []
    first_fall = None
    first_warning = None
    render = trial == getattr(args, "video_trial", 0) and not args.no_video
    video_path = args.output_dir / "standing_mujoco.mp4"
    with contextlib.ExitStack() as stack:
        renderer = writer = viewer = None
        if render:
            import imageio.v2 as imageio
            from PIL import Image
            renderer = stack.enter_context(mujoco.Renderer(model, height=args.height, width=args.width))
            writer = stack.enter_context(imageio.get_writer(str(video_path), fps=30, codec="libx264", macro_block_size=2))
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0, 0, 0.85]
            camera.distance, camera.azimuth, camera.elevation = 2.8, 35, -12
            option = mujoco.MjvOption()
            option.geomgroup[:] = 0
            option.geomgroup[0] = option.geomgroup[2] = 1
        if args.viewer and trial == 0:
            import mujoco.viewer as mjviewer
            viewer = stack.enter_context(mjviewer.launch_passive(model, data))
            viewer.cam.lookat[:] = [0, 0, 0.85]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.8, 35, -12
        for step in range(round(args.seconds / cfg["control_dt"])):
            wall_start = time.perf_counter()
            obs, _ = observation(model, data, root, qpos_ids, dof_ids, default_pos, last_action)
            with torch.inference_mode():
                action = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
            if action.shape != (14,) or not np.isfinite(action).all():
                raise RuntimeError("Invalid policy action")
            data.ctrl[:] = default_pos + cfg["action_scale"] * action
            last_action = action.copy()
            for _ in range(cfg["steps_per_action"]):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            _, gravity = observation(model, data, root, qpos_ids, dof_ids, default_pos, last_action)
            height = float(data.xpos[root, 2])
            tilt = float(np.arccos(np.clip(-gravity[2], -1, 1)))
            mimic = data.joint(PREFIX + "WAIST_P_RIGHT_out").qpos[0] + data.joint(PREFIX + "WAIST_P_out").qpos[0]
            failed = height < cfg["fall_height"] or tilt > cfg["fall_tilt"]
            if failed and first_fall is None:
                first_fall = float(data.time)
            warnings = [int(w.number) for w in data.warning]
            if any(warnings) and first_warning is None:
                first_warning = {"time": float(data.time), "counts": warnings}
            trace.append({"time_s": float(data.time), "height_m": height, "tilt_deg": math.degrees(tilt),
                          "x_m": float(data.xpos[root, 0]), "y_m": float(data.xpos[root, 1]),
                          "max_abs_action": float(np.abs(action).max()),
                          "max_abs_torque_Nm": float(np.abs(data.actuator_force).max()),
                          "mimic_error_rad": float(abs(mimic)), "contacts": int(data.ncon)})
            if render and step % 2 == 0:
                renderer.update_scene(data, camera=camera, scene_option=option)
                frame = Image.fromarray(renderer.render())
                writer.append_data(np.asarray(frame))
                if step == 0 or abs(data.time - args.seconds / 2) < cfg["control_dt"]:
                    frame.save(args.output_dir / "standing_mujoco_preview.png")
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(max(0.0, cfg["control_dt"] - (time.perf_counter() - wall_start)))
    end_xy = data.xpos[root, :2]
    return {
        "trial": trial, "initial_pose": "nominal" if trial == 0 else "randomized within training ranges",
        "duration_s": float(data.time), "survived": first_fall is None,
        "first_fall_s": first_fall, "first_warning": first_warning,
        "minimum_height_m": min(row["height_m"] for row in trace),
        "maximum_tilt_deg": max(row["tilt_deg"] for row in trace),
        "final_height_m": trace[-1]["height_m"], "final_tilt_deg": trace[-1]["tilt_deg"],
        "displacement_m": float(np.linalg.norm(end_xy - initial_xy)),
        "maximum_mimic_error_rad": max(row["mimic_error_rad"] for row in trace),
        "maximum_torque_Nm": max(row["max_abs_torque_Nm"] for row in trace),
    }, trace


def main():
    args = parse_args()
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_deployment_config(args.training_config, args.urdf)
    cfg["seed"] = args.seed
    model_path = args.output_dir / "grasshopper_standing.xml"
    model = prepare_model(args.robot, model_path, cfg)
    if model.nu != 14 or model.nv != 21:
        raise ValueError(f"Unexpected model dimensions: nu={model.nu}, nv={model.nv}")
    policy = torch.jit.load(str(args.policy), map_location="cpu").eval()
    trials = []
    for trial in range(args.trials):
        result, trace = run_trial(model, policy, cfg, args, trial)
        trials.append(result)
        with (args.output_dir / f"trace_{trial:02d}.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
            writer.writeheader()
            writer.writerows(trace)
        print(json.dumps(result), flush=True)
    report = {
        "runtime": "Native mujoco.mj_step and TorchScript policy; no Isaac Lab or Newton runtime",
        "versions": {name: version(name) for name in ("mujoco", "torch")},
        "policy": str(args.policy.resolve()), "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        "source_robot": str(args.robot.resolve()), "deployment_model": str(model_path.resolve()),
        "training_config": str(args.training_config.resolve()), "config": cfg,
        "summary": {"trials": len(trials), "survived": sum(t["survived"] for t in trials),
                    "maximum_tilt_deg": max(t["maximum_tilt_deg"] for t in trials),
                    "minimum_height_m": min(t["minimum_height_m"] for t in trials)},
        "trials": trials,
        "notes": ["Native MuJoCo CPU differs numerically from the training MuJoCo Warp backend.",
                  "Policy weights are unchanged; controller/model settings come from the saved training configuration.",
                  "Trial 0 is nominal; later trials use independently sampled training reset ranges.",
                  "No external pushes, automatic resets, body supports, or gravity compensation."],
    }
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "deployment_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
