#!/usr/bin/env python3
"""Render recorded physical walking states using the actual CAD USD in Isaac RTX."""

import argparse
import json
import math
from pathlib import Path
from xml.etree import ElementTree

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--state", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--prefix", default="nest_walk")
parser.add_argument("--width", type=int, default=2560)
parser.add_argument("--height", type=int, default=1440)
parser.add_argument("--grid-spacing", type=float, default=2.2)
args = parser.parse_args()
if args.width < 1 or args.height < 1 or args.grid_spacing <= 0:
    parser.error("Resolution and grid spacing must be positive")

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "width": args.width, "height": args.height,
                     "renderer": "RayTracedLighting", "fast_shutdown": True})


def look_at(eye, target):
    from pxr import Gf
    eye, target = Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target))
    forward = (target - eye).GetNormalized()
    right = (forward ^ Gf.Vec3d(0, 0, 1)).GetNormalized()
    up = (right ^ forward).GetNormalized()
    matrix = Gf.Matrix4d()
    matrix[0], matrix[1] = [*right, 0], [*up, 0]
    matrix[2] = [-forward[0], -forward[1], -forward[2], 0]
    matrix[3] = [*eye, 1]
    return matrix


def main():
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom, UsdLux
    from nest.assets.robots.grasshopper import USD_PATH, URDF_PATH

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = np.load(args.state)
    positions, quaternions, angles = state["root_pos_w"], state["root_quat_w"], state["joint_pos"]
    names = list(state["joint_names"])
    child_links = {j.attrib["name"]: j.find("child").attrib["link"]
                   for j in ElementTree.parse(URDF_PATH).getroot().findall("joint")
                   if j.attrib["name"] in names}
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    for index, (pos, quat, joint_positions) in enumerate(zip(positions, quaternions, angles)):
        root = UsdGeom.Xform.Define(stage, f"/World/Robot_{index:03d}").GetPrim()
        root.GetReferences().AddReference(str(USD_PATH))
        root.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
        geometry = stage.GetPrimAtPath(f"{root.GetPath()}/Geometry")
        by_name = {p.GetName(): p for p in Usd.PrimRange(geometry)}
        for name, angle in zip(names, joint_positions):
            child = by_name[child_links[name]]
            orient = child.GetAttribute("xformOp:orient")
            half = float(angle) * 0.5
            orient.Set(orient.Get() * Gf.Quatf(math.cos(half), Gf.Vec3f(0, 0, math.sin(half))))
        # Recorded root is the PELVIS link frame; account for the USD's fixed
        # parent chain rather than guessing an offset or changing the posture.
        pelvis = by_name["PELVIS"]
        cache = UsdGeom.XformCache()
        local_pelvis = cache.GetLocalToWorldTransform(pelvis) * cache.GetLocalToWorldTransform(root).GetInverse()
        desired = Gf.Matrix4d()
        desired.SetRotate(Gf.Quatd(float(quat[3]), Gf.Vec3d(*map(float, quat[:3]))))
        desired.SetTranslateOnly(Gf.Vec3d(*map(float, pos)))
        UsdGeom.Xformable(root).MakeMatrixXform().Set(local_pelvis.GetInverse() * desired)
        if (index + 1) % 128 == 0:
            print(f"ISAAC_SCENE_ROBOTS {index + 1}/{len(positions)}", flush=True)

    floor = UsdGeom.Cube.Define(stage, "/Render/Floor")
    floor.CreateDisplayColorAttr([Gf.Vec3f(0.46, 0.49, 0.54)])
    xform = UsdGeom.Xformable(floor.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.025))
    xform.AddScaleOp().Set(Gf.Vec3f(100, 100, 0.025))
    dome = UsdLux.DomeLight.Define(stage, "/Render/Dome")
    dome.CreateIntensityAttr(520.0)
    dome.CreateColorAttr(Gf.Vec3f(0.95, 0.97, 1.0))
    key = UsdLux.DistantLight.Define(stage, "/Render/Key")
    key.CreateIntensityAttr(1200.0)
    key.CreateAngleAttr(1.0)
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-42, 28, -38))

    # Thin visual cell boundaries express the original parallel environment layout.
    side = math.ceil(math.sqrt(len(positions)))
    extent = side * args.grid_spacing / 2
    for axis in (0, 1):
        for i in range(side + 1):
            line = UsdGeom.Cube.Define(stage, f"/Render/Grid_{axis}_{i}")
            line.CreateDisplayColorAttr([Gf.Vec3f(0.65, 0.68, 0.72)])
            t = [-extent + i * args.grid_spacing, 0, 0.001] if axis == 0 else [0, -extent + i * args.grid_spacing, 0.001]
            scale = [0.008, extent, 0.001] if axis == 0 else [extent, 0.008, 0.001]
            xf = UsdGeom.Xformable(line.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*t))
            xf.AddScaleOp().Set(Gf.Vec3f(*scale))

    camera = UsdGeom.Camera.Define(stage, "/Render/Camera")
    camera.CreateFocalLengthAttr(42.0)
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateVerticalApertureAttr(36.0 * args.height / args.width)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 300))
    camera_op = UsdGeom.Xformable(camera.GetPrim()).MakeMatrixXform()
    center = (positions.min(axis=0) + positions.max(axis=0)) / 2
    span = max(float(np.ptp(positions[:, 0])), float(np.ptp(positions[:, 1])), 4)
    views = {
        "overview": ((center[0] + span * 1.5, center[1] + span * 1.4, span * 1.3),
                     (center[0], center[1], 0.55)),
        "hero": ((positions[:, 0].max() + 5.5, positions[:, 1].max() + 6.0, 3.3),
                 (positions[:, 0].max() - args.grid_spacing, positions[:, 1].max() - args.grid_spacing, 0.75)),
    }
    rep.orchestrator.set_capture_on_play(False)
    product = rep.create.render_product(str(camera.GetPath()), (args.width, args.height))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach([product])
    report = {"renderer": "Isaac Sim 6.1 RTX RayTracedLighting", "state": str(args.state.resolve()),
              "robot_usd": str(USD_PATH), "num_robots": len(positions),
              "resolution": [args.width, args.height],
              "pose_source": "Recorded physical simulation state; no authored poses",
              "physics_backend": str(state["physics_backend"]) if "physics_backend" in state else "PhysX",
              "training_iteration": int(state["training_iteration"]) if "training_iteration" in state else None,
              "images": []}
    for name, (eye, target) in views.items():
        camera_op.Set(look_at(eye, target))
        for _ in range(45):
            app.update()
        for _ in range(4):
            rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=True)
        data = np.asarray(rgb_annotator.get_data())
        if data.size == 0 or data[..., :3].std() < 8:
            raise RuntimeError(f"Invalid RTX capture for {name}")
        path = output / f"{args.prefix}_isaac_{name}.png"
        Image.fromarray(data[..., :3].astype(np.uint8)).save(path)
        report["images"].append(str(path))
        print(f"ISAAC_RENDER_SAVED {path}", flush=True)
    stage.Export(str(output / f"{args.prefix}_scene.usdc"))
    (output / "render.json").write_text(json.dumps(report, indent=2) + "\n")
    rgb_annotator.detach()
    product.destroy()
    rep.orchestrator.wait_until_complete()


try:
    main()
finally:
    app.close()
