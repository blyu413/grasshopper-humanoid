#!/usr/bin/env python3
"""Render the Grasshopper standing reference from front and side."""

import math
from pathlib import Path
from xml.etree import ElementTree

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {"headless": True, "width": 960, "height": 720, "renderer": "RayTracedLighting", "fast_shutdown": True}
)


def look_at_matrix(eye, target):
    from pxr import Gf

    eye = Gf.Vec3d(*eye)
    target = Gf.Vec3d(*target)
    forward = (target - eye).GetNormalized()
    right = (forward ^ Gf.Vec3d(0, 0, 1)).GetNormalized()
    up = (right ^ forward).GetNormalized()
    matrix = Gf.Matrix4d()
    matrix[0] = [right[0], right[1], right[2], 0]
    matrix[1] = [up[0], up[1], up[2], 0]
    matrix[2] = [-forward[0], -forward[1], -forward[2], 0]
    matrix[3] = [eye[0], eye[1], eye[2], 1]
    return matrix


def main():
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom, UsdLux

    from nest.assets.robots.grasshopper import STAND_JOINT_POS, STAND_ROOT_HEIGHT, URDF_PATH, USD_PATH

    if not simulation_app.context.open_stage(str(USD_PATH)):
        raise RuntimeError(f"Could not open {USD_PATH}")
    context = omni.usd.get_context()
    for _ in range(600):
        simulation_app.update()
        loading = context.get_stage_loading_status()
        if not loading or loading[2] == 0:
            break
    else:
        raise RuntimeError("USD stage did not finish loading")

    stage = context.get_stage()
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        raise RuntimeError("USD has no valid default prim")
    root.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
    geometry = stage.GetPrimAtPath(f"{root.GetPath()}/Geometry")
    child_links = {
        joint.attrib["name"]: joint.find("child").attrib["link"]
        for joint in ElementTree.parse(URDF_PATH).getroot().findall("joint")
        if joint.attrib["name"] in STAND_JOINT_POS
    }
    for joint_name, position in STAND_JOINT_POS.items():
        child_link = child_links[joint_name]
        matches = [prim for prim in Usd.PrimRange(geometry) if prim.GetName() == child_link]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one geometry prim for {child_link}, found {len(matches)}")
        orient_attr = matches[0].GetAttribute("xformOp:orient")
        half_angle = position / 2.0
        joint_rotation = Gf.Quatf(math.cos(half_angle), Gf.Vec3f(0, 0, math.sin(half_angle)))
        orient_attr.Set(orient_attr.Get() * joint_rotation)
    pelvis = stage.GetPrimAtPath(f"{root.GetPath()}/Geometry/base_link/PELVIS")
    if not pelvis.IsValid():
        raise RuntimeError("Could not find PELVIS link")
    pelvis_offset_z = UsdGeom.Xformable(pelvis).GetLocalTransformation().ExtractTranslation()[2]
    spawn_z = STAND_ROOT_HEIGHT - pelvis_offset_z
    UsdGeom.Xformable(root).AddTranslateOp().Set(Gf.Vec3d(0, 0, spawn_z))

    floor = UsdGeom.Cube.Define(stage, "/Render/Floor")
    floor.CreateSizeAttr(2.0)
    floor.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.16, 0.21)])
    floor_xform = UsdGeom.Xformable(floor.GetPrim())
    floor_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.02))
    floor_xform.AddScaleOp().Set(Gf.Vec3f(4.0, 4.0, 0.02))

    dome = UsdLux.DomeLight.Define(stage, "/Render/DomeLight")
    dome.CreateIntensityAttr(300.0)
    dome.CreateColorAttr(Gf.Vec3f(0.82, 0.88, 1.0))
    sun = UsdLux.DistantLight.Define(stage, "/Render/DistantLight")
    sun.CreateIntensityAttr(1200.0)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-50, 20, -25))

    camera = UsdGeom.Camera.Define(stage, "/Render/Camera")
    camera.CreateFocalLengthAttr(28.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_xform.ClearXformOpOrder()
    camera_op = camera_xform.MakeMatrixXform()

    rep.orchestrator.set_capture_on_play(False)
    render_product = rep.create.render_product(str(camera.GetPath()), (960, 720))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach([render_product])

    output_dir = Path(__file__).resolve().parents[1] / "outputs/grasshopper_torso"
    output_dir.mkdir(parents=True, exist_ok=True)
    views = {
        "front": (3.6, 0.0, 0.9),
        "side": (0.0, 3.6, 0.9),
    }
    target = (0.0, 0.0, 0.8)
    for name, eye in views.items():
        camera_op.Set(look_at_matrix(eye, target))
        for _ in range(45):
            simulation_app.update()
        for _ in range(4):
            rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=True)
        data = rgb_annotator.get_data()
        if not isinstance(data, np.ndarray) or data.size == 0:
            raise RuntimeError(f"No RGB data for {name} view")
        rgb = np.asarray(data)[..., :3].astype(np.uint8, copy=False)
        if rgb.max() == 0 or rgb.std() < 10:
            raise RuntimeError(f"Invalid {name} render: mean={rgb.mean():.1f}, std={rgb.std():.1f}")
        output = output_dir / f"grasshopper_stand_pose_{name}.png"
        Image.fromarray(rgb, mode="RGB").save(output)
        print(f"{name}: {output}, root_z={STAND_ROOT_HEIGHT:.3f}, min_rgb={rgb.min()}, max_rgb={rgb.max()}", flush=True)

    rgb_annotator.detach()
    render_product.destroy()
    rep.orchestrator.wait_until_complete()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
