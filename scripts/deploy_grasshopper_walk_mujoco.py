#!/usr/bin/env python3
"""Test a 54-input walking policy using standalone mujoco.mj_step and TorchScript.

The physics model is a snapshot of the nominal training model. Isaac Lab and
Newton are used only to export that snapshot, never by this test process.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
import yaml

from deploy_grasshopper_mujoco import TrainingConfigLoader

ROOT = Path(__file__).resolve().parents[1]


def find_body(model, suffix):
    ids = [i for i in range(1, model.nbody)
           if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").endswith("_" + suffix)]
    if len(ids) != 1:
        raise ValueError(f"Expected one body ending in {suffix!r}, got {ids}")
    return ids[0]


def load_config(path):
    cfg = yaml.load(path.read_text(), Loader=TrainingConfigLoader)
    terms = cfg["observations"]["policy"]
    names = [k for k, v in terms.items() if isinstance(v, dict) and "func" in v]
    expected = ["base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "velocity_commands"]
    if names != expected:
        raise ValueError(f"Expected the 54-input walking observation layout, got {names}")
    if any(terms[n].get("clip") is not None or terms[n].get("scale") is not None for n in names):
        raise ValueError("Observation scaling/clipping must be handled explicitly")
    action = cfg["actions"]["joint_pos"]
    if not action["preserve_order"] or not action["use_default_offset"] or action.get("clip") is not None:
        raise ValueError("Unsupported joint action ordering/offset/clipping")
    robot = cfg["scene"]["robot"]
    joint_names = action["joint_names"]
    pd = robot["actuators"]["driven_joints"]
    scale = action["scale"]
    agent = yaml.load((path.parent / "agent.yaml").read_text(), Loader=TrainingConfigLoader)
    return {
        "joint_names": joint_names,
        "default_joint_pos": robot["init_state"]["joint_pos"],
        "root_position": robot["init_state"]["pos"], "root_quaternion_xyzw": robot["init_state"]["rot"],
        "action_scale": [scale[n] if isinstance(scale, dict) else scale for n in joint_names],
        "kp": [pd["stiffness"][n] for n in joint_names],
        "kd": [pd["damping"][n] for n in joint_names],
        "effort_limit": [pd["joint_effort_limit"][n] for n in joint_names],
        "physics_dt": cfg["sim"]["dt"], "decimation": cfg["decimation"],
        "clip_actions": agent.get("clip_actions"),
        "reset_pose_range": cfg["events"]["reset_base"]["params"]["pose_range"],
        "training_fall_tilt": cfg["terminations"]["bad_orientation"]["params"]["limit_angle"],
    }


def prepare_model(binary_path, xml_path, visual_path, output):
    """Add massless display meshes and retain the exact compiled physics arrays."""
    reference = mujoco.MjModel.from_binary_path(str(binary_path))
    source = mujoco.MjModel.from_xml_path(str(visual_path))
    source_data, ref_data = mujoco.MjData(source), mujoco.MjData(reference)
    mujoco.mj_forward(source, source_data)
    mujoco.mj_forward(reference, ref_data)
    source_root, ref_root = find_body(source, "PELVIS"), find_body(reference, "PELVIS")
    source_data.qpos[:3] += ref_data.xpos[ref_root] - source_data.xpos[source_root]
    mujoco.mj_forward(source, source_data)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    source_xml = ET.parse(visual_path).getroot()
    for mesh in source_xml.findall("asset/mesh"):
        display = copy.deepcopy(mesh)
        display.set("name", "display_" + display.attrib["name"])
        asset.append(display)
    ET.SubElement(asset, "texture", name="native_sky", type="skybox", builtin="gradient",
                  rgb1="0.55 0.65 0.75", rgb2="0.85 0.88 0.92", width="512", height="3072")
    bodies = {b.attrib["name"]: b for b in root.findall("worldbody//body")}
    visual_count = 0
    for body in source_xml.findall("worldbody//body"):
        geoms = [g for g in body.findall("geom") if g.get("group") == "2"]
        if not geoms:
            continue
        suffix = body.attrib["name"].removeprefix("grasshopper_lower_body_")
        target_id = find_body(reference, suffix)
        source_id = find_body(source, suffix)
        target = bodies[mujoco.mj_id2name(reference, mujoco.mjtObj.mjOBJ_BODY, target_id)]
        sr = source_data.xmat[source_id].reshape(3, 3)
        tr = ref_data.xmat[target_id].reshape(3, 3)
        for geom in geoms:
            display = copy.deepcopy(geom)
            p = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
            q = np.fromstring(geom.get("quat", "1 0 0 0"), sep=" ")
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, q)
            local_rotation = tr.T @ sr @ rotation.reshape(3, 3)
            local_position = tr.T @ (source_data.xpos[source_id] + sr @ p - ref_data.xpos[target_id])
            local_quat = np.empty(4)
            mujoco.mju_mat2Quat(local_quat, local_rotation.ravel())
            display.set("pos", " ".join(map(str, local_position)))
            display.set("quat", " ".join(map(str, local_quat)))
            display.set("name", "display_" + display.attrib["name"])
            display.set("mesh", "display_" + display.attrib["mesh"])
            display.set("mass", "0")
            display.set("contype", "0")
            display.set("conaffinity", "0")
            target.append(display)
            visual_count += 1
    world = root.find("worldbody")
    for geom in world.findall("geom"):
        if geom.get("type") == "plane":
            geom.set("rgba", "0.42 0.47 0.53 1")
    ET.SubElement(world, "light", pos="-2 -3 5", dir=".4 .4 -1", directional="true", diffuse=".8 .8 .8")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", "1920")
    global_visual.set("offheight", "1080")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.set("ambient", ".35 .35 .35")
    headlight.set("diffuse", ".5 .5 .5")
    ET.indent(tree)
    display_path = output / "grasshopper_walk.xml"
    tree.write(display_path, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(display_path))
    if (model.nbody, model.njnt, model.nu) != (reference.nbody, reference.njnt, reference.nu):
        raise AssertionError("Display meshes changed the physics topology")
    # XML serialization may round numbers, and Newton applies some parameters
    # after compilation. Restore the live snapshot exactly, including limits.
    physical = (
        "body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia", "body_gravcomp",
        "jnt_pos", "jnt_axis", "jnt_range", "jnt_solref", "jnt_solimp", "jnt_margin", "jnt_stiffness",
        "jnt_actfrcrange", "jnt_actfrclimited", "dof_damping", "dof_armature", "dof_frictionloss",
        "dof_solref", "dof_solimp", "actuator_gainprm", "actuator_biasprm", "actuator_dynprm",
        "actuator_gear", "actuator_ctrlrange", "actuator_forcerange", "actuator_ctrllimited",
        "actuator_forcelimited", "eq_data", "eq_solref", "eq_solimp", "qpos0", "qpos_spring",
    )
    copied = []
    for name in physical:
        if hasattr(reference, name):
            a, b = getattr(model, name), getattr(reference, name)
            if a.shape != b.shape:
                raise AssertionError(f"Physics array shape changed: {name}")
            a[:] = b
            copied.append(name)
    for name in ("timestep", "gravity", "integrator", "solver", "iterations", "ls_iterations", "tolerance",
                 "ls_tolerance", "cone", "impratio", "ccd_iterations", "ccd_tolerance", "disableflags", "enableflags"):
        value = getattr(reference.opt, name)
        if isinstance(value, np.ndarray):
            getattr(model.opt, name)[:] = value
        else:
            setattr(model.opt, name, value)
    geom_fields = ("type", "size", "pos", "quat", "friction", "solref", "solimp", "margin", "gap",
                   "condim", "priority", "contype", "conaffinity")
    for i in range(reference.ngeom):
        name = mujoco.mj_id2name(reference, mujoco.mjtObj.mjOBJ_GEOM, i)
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if j < 0:
            raise AssertionError(f"Lost physics geom {name}")
        for field in geom_fields:
            getattr(model, "geom_" + field)[j] = getattr(reference, "geom_" + field)[i]
        model.geom_group[j] = 0 if model.geom_type[j] == mujoco.mjtGeom.mjGEOM_PLANE else 3
        if model.geom_type[j] == mujoco.mjtGeom.mjGEOM_PLANE:
            # MuJoCo planes collide infinitely; these dimensions affect display only.
            model.geom_size[j, :2] = 50.
    mujoco.mj_setConst(model, mujoco.MjData(model))
    for name in copied:
        np.testing.assert_array_equal(getattr(model, name), getattr(reference, name))
    mujoco.mj_saveModel(model, str(output / "grasshopper_walk.mjb"))
    return model, {"visual_mesh_geoms": visual_count, "physics_arrays_verified_exact": copied,
                   "physics_source": str(binary_path.resolve()),
                   "model_with_exact_parameters": str((output / "grasshopper_walk.mjb").resolve())}


def addresses(model, cfg):
    joint_ids = []
    for name in cfg["joint_names"]:
        ids = [i for i in range(model.njnt)
               if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "").endswith("_" + name)]
        if len(ids) != 1:
            raise ValueError(f"Joint mapping failed for {name}: {ids}")
        joint_ids.append(ids[0])
    positions = []
    for j, kp, kd in zip(joint_ids, cfg["kp"], cfg["kd"], strict=True):
        aids = np.flatnonzero(model.actuator_trnid[:, 0] == j)
        pa = [int(i) for i in aids if model.actuator_biasprm[i, 1] < 0]
        va = [int(i) for i in aids if model.actuator_biasprm[i, 2] < 0]
        if len(pa) != 1 or len(va) != 1:
            raise ValueError("Expected the exported separate position and velocity servos")
        np.testing.assert_allclose(model.actuator_gainprm[pa[0], 0], kp)
        np.testing.assert_allclose(-model.actuator_biasprm[va[0], 2], kd)
        positions.append(pa[0])
    return model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids], np.asarray(positions)


def observe(model, data, root, qids, vids, default, last_action, command):
    rotation = data.xmat[root].reshape(3, 3)
    com, link = np.empty(6), np.empty(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, root, com, 0)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, root, link, 0)
    gravity = rotation.T @ np.array([0., 0., -1.])
    # mdp.base_lin_vel observes COM velocity, whereas tracking rewards use the
    # link origin. Both are expressed in the root link's orientation.
    obs = np.concatenate((rotation.T @ com[3:], rotation.T @ com[:3], gravity,
                          data.qpos[qids] - default, data.qvel[vids], last_action, command)).astype(np.float32)
    if obs.shape != (54,) or not np.isfinite(obs).all():
        raise RuntimeError("Invalid walking observation")
    return obs, gravity, link[3:], rotation.T @ link[3:], (rotation.T @ link[:3])[2]


def reset(model, data, cfg, root, qids, pids, trial, seed):
    mujoco.mj_resetData(model, data)
    for name, value in cfg["default_joint_pos"].items():
        ids = [i for i in range(model.njnt)
               if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "").endswith("_" + name)]
        if len(ids) == 1:
            data.qpos[model.jnt_qposadr[ids[0]]] = value
    data.qpos[3:7] = np.asarray(cfg["root_quaternion_xyzw"])[[3, 0, 1, 2]]
    target = np.asarray(cfg["root_position"], dtype=float).copy()
    bounds = cfg["reset_pose_range"]
    rng = np.random.default_rng(seed + trial)
    target[2] += rng.uniform(*bounds["z"])
    if trial:
        target[:2] += [rng.uniform(*bounds[a]) for a in ("x", "y")]
    mujoco.mj_forward(model, data)
    data.qpos[:3] += target - data.xpos[root]
    data.ctrl[:] = 0
    data.ctrl[pids] = data.qpos[qids]
    mujoco.mj_forward(model, data)


def draw_arrow(scene, start, end, color, width=.018):
    if np.linalg.norm(end - start) < .02:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(), color)
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width, start, end)
    scene.ngeom += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--model-export", type=Path, required=True)
    parser.add_argument("--visual-model", type=Path, default=ROOT / "outputs/grasshopper_torso/mjcf/grasshopper_lower_body.xml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speeds", type=float, nargs="+", default=[.5])
    parser.add_argument("--yaw-rates", type=float, nargs="+", default=[0., .5, -.5])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if args.trials < 1 or args.seconds <= 2:
        parser.error("Use at least one trial and seconds > 2")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cfg = load_config(args.training_config)
    model, model_report = prepare_model(args.model_export / "training_nominal.mjb",
        args.model_export / "training_nominal.xml", args.visual_model, args.output_dir)
    # Newton sets dt immediately before stepping; an export taken before its
    # first step can still contain MuJoCo's default 2 ms instead of training dt.
    model.opt.timestep = cfg["physics_dt"]
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    mujoco.mj_saveModel(model, str(args.output_dir / "grasshopper_walk.mjb"))
    dt = cfg["physics_dt"] * cfg["decimation"]
    qids, vids, pids = addresses(model, cfg)
    root = find_body(model, "PELVIS")
    foot_ids = [find_body(model, name) for name in ("R_FOOT", "L_FOOT")]
    for foot in foot_ids:
        g = np.flatnonzero((model.geom_bodyid == foot) & ((model.geom_contype | model.geom_conaffinity) != 0))
        assert len(g) == 7 and (model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE).all()
        np.testing.assert_allclose(model.geom_size[g, 0], .01)
    policy = torch.jit.load(str(args.policy), map_location="cpu").eval()
    if policy.state_dict()["mlp.0.weight"].shape[1] != 54:
        raise ValueError("This runner requires a 54-input walking policy")
    default = np.asarray([cfg["default_joint_pos"][n] for n in cfg["joint_names"]])
    report = {"status": "testing", "runtime": "Standalone mujoco.mj_step + CPU TorchScript; no Isaac Lab or Newton imports",
              "policy": str(args.policy.resolve()), "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
              "mujoco_version": mujoco.__version__, "config": cfg, "model": model_report,
              "control_dt_s": dt, "seconds_per_trial": args.seconds, "warmup_s": 2., "auto_reset": False,
              "conditions": "Nominal dynamics; no noise, pushes, latency or startup hardware randomization; initial heading zero",
              "cases": []}
    result_path = args.output_dir / "results.json"
    renderer = writer = None
    if not args.no_video:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw, ImageFont
        renderer = mujoco.Renderer(model, height=720, width=1280)
        writer = imageio.get_writer(str(args.output_dir / "mujoco_command_vs_actual.mp4"), fps=25,
                                   codec="libx264", quality=8, macro_block_size=2,
                                   ffmpeg_params=["-movflags", "+faststart"])
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 3.5, 40, -18
        option = mujoco.MjvOption()
        option.geomgroup[:] = 0
        option.geomgroup[0] = option.geomgroup[2] = 1
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    frames = 0
    started = time.monotonic()
    try:
        for speed, yaw in itertools.product(args.speeds, args.yaw_rates):
            command = np.array([speed, 0., yaw])
            results = []
            for trial in range(args.trials):
                data = mujoco.MjData(model)
                reset(model, data, cfg, root, qids, pids, trial, args.seed)
                last_action = np.zeros(14)
                rows, failed, training_failed, path = [], False, False, []
                for step in range(round(args.seconds / dt)):
                    obs, *_ = observe(model, data, root, qids, vids, default, last_action, command)
                    with torch.inference_mode():
                        action = policy(torch.from_numpy(obs).unsqueeze(0))[0].numpy()
                    if not np.isfinite(action).all():
                        raise RuntimeError("Non-finite action")
                    if cfg["clip_actions"] is not None:
                        action = np.clip(action, -cfg["clip_actions"], cfg["clip_actions"])
                    data.ctrl[pids] = default + np.asarray(cfg["action_scale"]) * action
                    last_action = action.copy()
                    for _ in range(cfg["decimation"]):
                        mujoco.mj_step(model, data)
                    mujoco.mj_forward(model, data)
                    _, gravity, world_velocity, body_velocity, actual_yaw = observe(
                        model, data, root, qids, vids, default, last_action, command)
                    tilt = math.acos(float(np.clip(-gravity[2], -1, 1)))
                    height = data.xpos[root, 2]
                    failed |= height < .7 or tilt > .8
                    training_failed |= tilt > cfg["training_fall_tilt"]
                    row = {"time_s": (step + 1) * dt, "vx_m_s": float(body_velocity[0]), "yaw_rad_s": float(actual_yaw),
                           "xy_error_m_s": float(np.linalg.norm(body_velocity[:2] - command[:2])),
                           "yaw_error_rad_s": float(abs(actual_yaw - yaw)), "height_m": float(height),
                           "tilt_deg": math.degrees(tilt), "fallen": bool(failed),
                           "torque_limit_ratio": float(np.max(np.abs(data.qfrc_actuator[vids]) / cfg["effort_limit"]))}
                    rows.append(row)
                    if writer is not None and trial == 0 and step % 2 == 1 and (step + 1) * dt <= 12.:
                        camera.lookat[:] = data.xpos[root]
                        renderer.update_scene(data, camera=camera, scene_option=option)
                        scene = renderer.scene
                        for axis in range(2):
                            for meter in range(-15, 21):
                                a = np.array([meter, -15, .003]) if axis == 0 else np.array([-15, meter, .003])
                                b = np.array([meter, 20, .003]) if axis == 0 else np.array([20, meter, .003])
                                geom = scene.geoms[scene.ngeom]
                                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), [.6, .65, .7, 1])
                                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 1., a, b)
                                scene.ngeom += 1
                        path.append(np.array([data.xpos[root, 0], data.xpos[root, 1], .012]))
                        for a, b in zip(path[::4][:-1], path[::4][1:]):
                            geom = scene.geoms[scene.ngeom]
                            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), [.85, .85, .85, 1])
                            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2., a, b)
                            scene.ngeom += 1
                        rotation = data.xmat[root].reshape(3, 3)
                        for vel, dz, color in ((rotation @ np.array([speed, 0., 0.]) * np.array([1, 1, 0]), .62, [0.05, .85, 1, 1]),
                                               (world_velocity * np.array([1, 1, 0]), .50, [1, .38, .06, 1])):
                            start = data.xpos[root] + [0, 0, dz]
                            draw_arrow(scene, start, start + vel * 1.25, color)
                        heading = math.atan2(rotation[1, 0], rotation[0, 0])
                        for rate, radius, color in ((yaw, .66, [0.05, .85, 1, 1]), (actual_yaw, .84, [1, .38, .06, 1])):
                            if abs(rate) > .015:
                                angles = np.linspace(heading, heading + np.clip(rate * 2.5, -2.5, 2.5), 24)
                                points = np.column_stack((data.xpos[root, 0] + radius * np.cos(angles),
                                                          data.xpos[root, 1] + radius * np.sin(angles), np.full(24, .075)))
                                for a, b in zip(points[:-1], points[1:]):
                                    geom = scene.geoms[scene.ngeom]
                                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), color)
                                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .012, a, b)
                                    scene.ngeom += 1
                                tangent = np.array([-math.sin(angles[-1]), math.cos(angles[-1]), 0.]) * np.sign(rate)
                                draw_arrow(scene, points[-1] - .1 * tangent, points[-1] + .04 * tangent, color)
                        frame = Image.fromarray(renderer.render())
                        draw = ImageDraw.Draw(frame)
                        draw.rectangle((0, 0, 1280, 95), fill=(20, 26, 33))
                        label = "LEFT" if yaw > 0 else "RIGHT" if yaw < 0 else "STRAIGHT"
                        draw.text((18, 7), f"NATIVE MUJOCO | 10k policy | {label} | t={(step+1)*dt:.2f}s", font=font, fill="white")
                        draw.text((18, 37), f"CMD vx {speed:+.2f} m/s  yaw {yaw:+.2f} rad/s", font=font, fill=(40, 220, 255))
                        draw.text((660, 37), f"ACT vx {body_velocity[0]:+.2f} m/s  yaw {actual_yaw:+.2f} rad/s", font=font, fill=(255, 145, 65))
                        draw.text((18, 67), "CYAN = COMMAND  |  ORANGE = ACTUAL  |  Straight: velocity  |  Curved: yaw rate", font=font, fill=(205, 215, 225))
                        writer.append_data(np.asarray(frame))
                        frames += 1
                        if frames in (120, 450, 750):
                            frame.save(args.output_dir / f"mujoco_frame_{frames:04d}.png")
                warm = [r for r in rows if r["time_s"] >= 2.]
                result = {"trial": trial, "survived": not failed, "survived_training_tilt": not training_failed,
                          "mean_vx_m_s": float(np.mean([r["vx_m_s"] for r in warm])),
                          "mean_yaw_rad_s": float(np.mean([r["yaw_rad_s"] for r in warm])),
                          "mean_xy_error_m_s": float(np.mean([r["xy_error_m_s"] for r in warm])),
                          "mean_yaw_error_rad_s": float(np.mean([r["yaw_error_rad_s"] for r in warm])),
                          "max_tilt_deg": max(r["tilt_deg"] for r in rows),
                          "max_torque_limit_ratio": max(r["torque_limit_ratio"] for r in rows),
                          "warning_counts": [int(w.number) for w in data.warning]}
                results.append(result)
                with (args.output_dir / f"trace_vx{speed:g}_yaw{yaw:g}_{trial:02d}.csv").open("w") as stream:
                    csv_writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    csv_writer.writeheader()
                    csv_writer.writerows(rows)
            case = {"command_vx_m_s": speed, "command_yaw_rad_s": yaw,
                    "survived": sum(r["survived"] for r in results), "samples": args.trials,
                    **{k: float(np.mean([r[k] for r in results])) for k in
                       ("mean_vx_m_s", "mean_yaw_rad_s", "mean_xy_error_m_s", "mean_yaw_error_rad_s")},
                    "trials": results}
            report["cases"].append(case)
            result_path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: v for k, v in case.items() if k != "trials"}), flush=True)
        report.update(status="complete", elapsed_s=time.monotonic() - started, video_frames=frames,
                      video=str((args.output_dir / "mujoco_command_vs_actual.mp4").resolve()) if writer else None)
        result_path.write_text(json.dumps(report, indent=2) + "\n")
    except BaseException as error:
        report.update(status="failed", error=str(error))
        result_path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        if writer is not None:
            writer.close()
            renderer.close()


if __name__ == "__main__":
    main()
