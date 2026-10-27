#!/usr/bin/env python3
"""Render presentation-ready standing views of the NEST Grasshopper robot."""

import math
from pathlib import Path
from xml.etree import ElementTree

from isaacsim import SimulationApp


WIDTH = 1200
HEIGHT = 1600

simulation_app = SimulationApp(
    {
        "headless": True,
        "width": WIDTH,
        "height": HEIGHT,
        "renderer": "RayTracedLighting",
        "fast_shutdown": True,
    }
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


def make_contact_sheet(images: dict[str, "Image.Image"], output: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (3840, 2160), (243, 245, 247))
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 76)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 42)
        meta_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
    except OSError:
        title_font = label_font = meta_font = ImageFont.load_default()

    draw.text((150, 90), "NEST  /  GRASSHOPPER", fill=(26, 32, 40), font=title_font)
    draw.text((154, 190), "STANDING REFERENCE  ·  CAD-ACCURATE RENDER", fill=(91, 101, 115), font=meta_font)

    labels = {
        "front": "FRONT",
        "left": "LEFT SIDE",
        "rear": "REAR",
        "hero": "THREE-QUARTER",
    }
    margin_x = 120
    gap = 36
    panel_w = (3840 - 2 * margin_x - 3 * gap) // 4
    panel_y = 300
    panel_h = 1680

    for index, (name, image) in enumerate(images.items()):
        x = margin_x + index * (panel_w + gap)
        draw.rounded_rectangle(
            (x, panel_y, x + panel_w, panel_y + panel_h),
            radius=28,
            fill=(231, 235, 239),
            outline=(210, 216, 223),
            width=3,
        )
        fitted = image.copy()
        fitted.thumbnail((panel_w - 24, panel_h - 130), Image.Resampling.LANCZOS)
        paste_x = x + (panel_w - fitted.width) // 2
        paste_y = panel_y + 20 + (panel_h - 130 - fitted.height) // 2
        canvas.paste(fitted, (paste_x, paste_y))
        label = labels[name]
        box = draw.textbbox((0, 0), label, font=label_font)
        text_w = box[2] - box[0]
        draw.text(
            (x + (panel_w - text_w) // 2, panel_y + panel_h - 88),
            label,
            fill=(47, 55, 66),
            font=label_font,
        )

    draw.line((150, 2070, 3690, 2070), fill=(196, 203, 211), width=3)
    draw.text((150, 2090), "Standing pose · neutral studio lighting", fill=(107, 116, 128), font=meta_font)
    canvas.save(output, quality=95)


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
    floor.CreateDisplayColorAttr([Gf.Vec3f(0.73, 0.76, 0.80)])
    floor_xform = UsdGeom.Xformable(floor.GetPrim())
    floor_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.025))
    floor_xform.AddScaleOp().Set(Gf.Vec3f(5.0, 5.0, 0.025))

    dome = UsdLux.DomeLight.Define(stage, "/Render/DomeLight")
    dome.CreateIntensityAttr(520.0)
    dome.CreateColorAttr(Gf.Vec3f(0.95, 0.97, 1.0))
    key = UsdLux.DistantLight.Define(stage, "/Render/KeyLight")
    key.CreateIntensityAttr(950.0)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.94, 0.86))
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-42, 28, -38))
    rim = UsdLux.DistantLight.Define(stage, "/Render/RimLight")
    rim.CreateIntensityAttr(500.0)
    rim.CreateColorAttr(Gf.Vec3f(0.76, 0.86, 1.0))
    UsdGeom.Xformable(rim.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-25, -145, 15))

    camera = UsdGeom.Camera.Define(stage, "/Render/Camera")
    camera.CreateFocalLengthAttr(92.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_xform.ClearXformOpOrder()
    camera_op = camera_xform.MakeMatrixXform()

    rep.orchestrator.set_capture_on_play(False)
    render_product = rep.create.render_product(str(camera.GetPath()), (WIDTH, HEIGHT))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach([render_product])

    output_dir = Path(__file__).resolve().parents[1] / "outputs/grasshopper_torso/presentation_views"
    output_dir.mkdir(parents=True, exist_ok=True)
    views = {
        "front": (8.0, 0.0, 0.86),
        "left": (0.0, 8.0, 0.86),
        "rear": (-8.0, 0.0, 0.86),
        "hero": (4.9, 4.9, 1.15),
    }
    target = (0.0, 0.0, 0.82)
    rendered: dict[str, Image.Image] = {}
    for name, eye in views.items():
        camera.CreateFocalLengthAttr(72.0 if name == "hero" else 92.0)
        camera_op.Set(look_at_matrix(eye, target))
        for _ in range(50):
            simulation_app.update()
        for _ in range(5):
            rep.orchestrator.step(rt_subframes=6, delta_time=0.0, pause_timeline=True)
        data = rgb_annotator.get_data()
        if not isinstance(data, np.ndarray) or data.size == 0:
            raise RuntimeError(f"No RGB data for {name} view")
        rgb = np.asarray(data)[..., :3].astype(np.uint8, copy=False)
        if rgb.max() == 0 or rgb.std() < 10:
            raise RuntimeError(f"Invalid {name} render: mean={rgb.mean():.1f}, std={rgb.std():.1f}")
        image = Image.fromarray(rgb, mode="RGB")
        output = output_dir / f"nest_grasshopper_standing_{name}.png"
        image.save(output, quality=95)
        rendered[name] = image
        print(f"{name}: {output}", flush=True)

    sheet_output = output_dir / "nest_grasshopper_standing_four_views_4k.png"
    make_contact_sheet(rendered, sheet_output)
    print(f"sheet: {sheet_output}", flush=True)

    rgb_annotator.detach()
    render_product.destroy()
    rep.orchestrator.wait_until_complete()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
