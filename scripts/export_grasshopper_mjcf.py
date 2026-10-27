#!/usr/bin/env python3
"""Export Grasshopper through Newton, restore URDF colors, and render MJCF views."""

from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import warp as wp

import newton
from newton.solvers import SolverMuJoCo

if __package__:
    from .urdf_mjcf_dynamics import apply_joint_damping, read_joint_damping
else:
    from urdf_mjcf_dynamics import apply_joint_damping, read_joint_damping


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--urdf", type=Path,
        default=ROOT / "assets/robots/grasshopper_torso/grasshopper_lower_body.urdf",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs/grasshopper_torso/mjcf",
    )
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--collapse-fixed-joints", action="store_true",
                        help="Merge fixed links and use only explicitly declared URDF masses")
    return parser.parse_args()


def numbers(values) -> str:
    return " ".join(format(float(value), ".17g") for value in values)


def visual_bounds(model, data) -> tuple[np.ndarray, np.ndarray]:
    points = []
    for geom in np.flatnonzero(model.geom_group == 2):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("This preview expects mesh-based visual geometry")
        mesh = model.geom_dataid[geom]
        start = model.mesh_vertadr[mesh]
        vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
        rotation = data.geom_xmat[geom].reshape(3, 3)
        points.append(vertices @ rotation.T + data.geom_xpos[geom])
    if not points:
        raise ValueError("No visual mesh geometry was exported")
    points = np.concatenate(points)
    return points.min(axis=0), points.max(axis=0)


def preview_scene(path: Path, robot_xml: Path, floor_z: float) -> None:
    scene = ET.Element("mujoco", model="grasshopper_visual_preview")
    ET.SubElement(scene, "include", file=robot_xml.name)
    assets = ET.SubElement(scene, "asset")
    ET.SubElement(assets, "texture", name="preview_sky", type="skybox", builtin="gradient", rgb1="0.78 0.83 0.90", rgb2="0.94 0.96 0.98", width="512", height="3072")
    visual = ET.SubElement(scene, "visual")
    ET.SubElement(visual, "global", offwidth="2048", offheight="2048")
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="4")
    ET.SubElement(visual, "headlight", ambient="0.25 0.25 0.25", diffuse="0.3 0.3 0.3", specular="0.05 0.05 0.05")
    world = ET.SubElement(scene, "worldbody")
    ET.SubElement(world, "light", directional="true", pos="-2 -3 4", dir="0.4 0.4 -1", diffuse="0.45 0.45 0.45", castshadow="true")
    ET.SubElement(world, "light", directional="true", pos="2 2 3", dir="-0.4 -0.4 -1", diffuse="0.2 0.2 0.22", castshadow="false")
    ET.SubElement(world, "geom", name="preview_floor", type="plane", pos=f"0 0 {floor_z:.9g}", size="0 0 0.1", rgba="0.7 0.73 0.78 1", contype="0", conaffinity="0", group="0")
    ET.indent(scene)
    ET.ElementTree(scene).write(path, encoding="utf-8", xml_declaration=True)


def render_views(scene_path, output_dir, center, span, width, height) -> None:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[0] = 1  # Preview floor.
    option.geomgroup[2] = 1  # Visuals; collision geometry is group 3.
    camera = mujoco.MjvCamera()
    camera.lookat[:] = center
    camera.distance = float(span * 1.72)
    camera.elevation = -8
    views = [("front", 0), ("rear", 180), ("left", 90), ("hero", 35)]
    frames = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for name, azimuth in views:
            camera.azimuth = azimuth
            camera.elevation = -12 if name == "hero" else -6
            renderer.update_scene(data, camera=camera, scene_option=option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
            frame = Image.fromarray(renderer.render())
            frame.save(output_dir / f"grasshopper_mjcf_{name}.png")
            frames.append((name.title(), frame))
            print(f"Rendered {name}", flush=True)
    label_height = 48
    montage = Image.new("RGB", (2 * width, 2 * (height + label_height)), "#edf0f4")
    draw = ImageDraw.Draw(montage)
    font = ImageFont.load_default(size=24)
    for index, (name, frame) in enumerate(frames):
        x = (index % 2) * width
        y = (index // 2) * (height + label_height)
        draw.text((x + 20, y + 10), name, font=font, fill="#243044")
        montage.paste(frame, (x, y + label_height))
    montage.save(output_dir / "grasshopper_mjcf_four_views.png")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(Path(tempfile.gettempdir()) / "nest_mjcf_warp_cache")
    wp.init()
    builder = newton.ModelBuilder()
    if args.collapse_fixed_joints:
        # Fixed shells without <inertial> must not acquire an arbitrary density.
        builder.default_shape_cfg.density = 0.0
    SolverMuJoCo.register_custom_attributes(builder)
    builder.add_urdf(
        str(args.urdf.resolve()), floating=True,
        collapse_fixed_joints=False, ignore_inertial_definitions=False,
        hide_visuals=False, parse_visuals_as_colliders=False,
    )
    # Standard URDF damping is passive. Newton's importer represents it as
    # velocity actuation; use joint damping in MJCF and avoid counting it twice.
    builder.joint_target_mode[:] = [int(newton.JointTargetMode.NONE)] * len(builder.joint_target_mode)
    imported_visuals = Counter(
        builder.body_label[builder.shape_body[index]].rsplit("/", 1)[-1]
        for index, flag in enumerate(builder.shape_flags) if flag & int(newton.ShapeFlags.VISIBLE)
    )
    if args.collapse_fixed_joints:
        builder.collapse_fixed_joints()
    model = builder.finalize(device="cpu")
    raw_path = args.output_dir / "grasshopper_lower_body_raw.xml"
    robot_path = args.output_dir / "grasshopper_lower_body.xml"
    solver = SolverMuJoCo(
        model, use_mujoco_cpu=True, skip_visual_only_geoms=False,
        save_to_mjcf=str(raw_path),
    )
    tree = ET.parse(raw_path)
    root = tree.getroot()
    passive_damping = read_joint_damping(args.urdf)
    robot_name = ET.parse(args.urdf).getroot().attrib["name"]
    apply_joint_damping(root, passive_damping, robot_name + "_")
    shape_lookup = {f"{label}_{i}": i for i, label in enumerate(model.shape_label)}
    flags = model.shape_flags.numpy()
    shape_bodies = model.shape_body.numpy()
    colors = model.shape_color.numpy()
    opacity = model.shape_opacity.numpy()
    color_counts = Counter()
    visual_count = collision_count = foot_capsule_count = 0
    for geom in root.findall(".//geom"):
        index = shape_lookup[geom.get("name")]
        is_visual = bool(flags[index] & int(newton.ShapeFlags.VISIBLE))
        geom.set("group", "2" if is_visual else "3")
        if is_visual:
            rgba = [*colors[index], opacity[index]]
            geom.set("rgba", numbers(rgba))
            color_counts[" ".join(f"{value:.3g}" for value in rgba)] += 1
            visual_count += 1
        if flags[index] & int(newton.ShapeFlags.COLLIDE_SHAPES):
            collision_count += 1
            body_name = model.body_label[int(shape_bodies[index])]
            if geom.get("type") == "capsule" and body_name.endswith(("R_FOOT", "L_FOOT")):
                foot_capsule_count += 1
                geom.set("priority", "1")
                geom.set("condim", "3")
                geom.set("friction", "0.6 0.005 0.0001")
                geom.set("solref", "0.02 1")
                geom.set("margin", "0")
                geom.set("gap", "0")
    source_robot = ET.parse(args.urdf).getroot()
    expected_foot_capsules = sum(
        len(source_robot.findall(f"link[@name='{foot}']/collision/geometry/capsule"))
        for foot in ("R_FOOT", "L_FOOT")
    )
    assert foot_capsule_count == expected_foot_capsules, "Foot contact defaults were not applied to every capsule"

    # Snapshot the actual compiled limit response for use in native MuJoCo.
    # MJCF cannot preserve Newton's force-space gain semantics for future edits.
    limit_count = 0
    for joint in root.findall("worldbody//joint"):
        index = mujoco.mj_name2id(solver.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint.get("name"))
        if index >= 0 and solver.mj_model.jnt_limited[index]:
            joint.set("solreflimit", numbers(solver.mj_model.jnt_solref[index]))
            limit_count += 1
    ET.indent(tree)
    tree.write(robot_path, encoding="utf-8", xml_declaration=True)

    exported = mujoco.MjModel.from_xml_path(str(robot_path))
    data = mujoco.MjData(exported)
    mujoco.mj_forward(exported, data)
    lower, upper = visual_bounds(exported, data)
    limited = exported.jnt_limited.astype(bool)
    limit_error = float(np.max(np.abs(exported.jnt_solref[limited] - solver.mj_model.jnt_solref[limited])))
    # A Collada visual can expand into several meshes (e.g. the Tesollo hands).
    # Check coverage per link instead of assuming one shape per URDF visual.
    source_visuals = {
        link.get("name"): len(link.findall("visual"))
        for link in source_robot.findall("link") if link.find("visual") is not None
    }
    missing_visuals = {
        name: count - imported_visuals[name]
        for name, count in source_visuals.items() if imported_visuals[name] < count
    }
    if missing_visuals:
        raise ValueError(f"Visual geometry is missing from URDF links: {missing_visuals}")
    assert limit_error < 1e-12, "Runtime joint-limit response was not preserved"
    assert np.isfinite(data.geom_xpos).all(), "Non-finite geometry transforms"
    scene_path = args.output_dir / "grasshopper_preview.xml"
    preview_scene(scene_path, robot_path, float(lower[2] - 0.015))
    render_views(scene_path, args.output_dir, (lower + upper) / 2, np.max(upper - lower), args.width, args.height)

    report = {
        "input": str(args.urdf.resolve()), "output": str(robot_path.resolve()),
        "versions": {name: version(name) for name in ("newton", "mujoco", "mujoco-warp")},
        "backend": "MuJoCo CPU; EGL rendering",
        "pose": "Authored zero joint pose, evaluated with mj_forward; no dynamics rollout",
        "visual_geoms": visual_count, "collision_geoms": collision_count,
        "source_visuals": sum(source_visuals.values()),
        "visual_link_coverage_verified": len(source_visuals),
        "g1_style_foot_capsules": foot_capsule_count,
        "rgba_counts": dict(color_counts),
        "counts": {name: int(getattr(exported, name)) for name in ("nbody", "njnt", "nq", "nv", "nu", "ngeom", "nmesh")},
        "mass_kg": float(exported.body_mass.sum()),
        "fixed_joints_collapsed": args.collapse_fixed_joints,
        "bounds": {"min": lower.tolist(), "max": upper.tolist()},
        "runtime_limit_responses_copied": limit_count,
        "runtime_limit_response_max_error": limit_error,
        "passive_joint_damping": passive_damping,
        "xml_bytes": robot_path.stat().st_size,
        "notes": ["Visual group 2; hidden collision group 3.", "Visual mesh data is embedded in the robot XML.", "Preview floor and lighting are in a separate scene XML.", "URDF has no textures; this script restores uniform RGBA only.", "Runtime limit response is a snapshot, not editable Newton force-space gains.", "URDF passive damping is written on joints; no duplicate velocity drives are generated."],
    }
    (args.output_dir / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
