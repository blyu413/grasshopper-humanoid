#!/usr/bin/env python3
"""Render an Isaac Lab walking state trace with the real robot USD in Isaac RTX."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from xml.etree import ElementTree

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--states", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--video-name", default="walk_1ms_isaac_rtx.mp4")
parser.add_argument("--title", default="GRASSHOPPER | Policy replay")
parser.add_argument("--velocity-arrows", action="store_true",
                    help="Show cyan commanded/orange measured planar velocity and yaw-rate arrows")
args = parser.parse_args()

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


def annotate_legacy(draw, states, index, font):
    """Keep the existing text overlay for traces without 3D velocity arrows."""
    draw.rectangle((0, 0, args.width, 68), fill=(20, 26, 33))
    draw.text((18, 7), args.title, font=font, fill="white")
    if "command_yaw_rad_s" in states:
        text = (
            f"vx {states['body_forward_velocity_m_s'][index]:.2f}/{states['command_m_s'][index]:.2f} m/s | "
            f"yaw {states['yaw_rate_rad_s'][index]:+.2f}/{states['command_yaw_rad_s'][index]:+.2f} rad/s | "
        )
    else:
        text = (
            f"Command {states['command_m_s'][index]:.2f} m/s | "
            f"Actual world vx {states['forward_velocity_m_s'][index]:.2f} m/s | "
        )
    draw.text((18, 36), text + f"t={states['time_s'][index]:.2f}s", font=font, fill=(130, 235, 180))


def main():
    import imageio.v2 as imageio
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image, ImageDraw, ImageFont
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade
    from nest.assets.robots.grasshopper import USD_PATH, URDF_PATH

    state_path = args.states.resolve()
    states = np.load(state_path)
    positions = states["root_pos_w"]
    quaternions = states["root_quat_w"]
    angles = states["joint_pos"]
    names = list(states["joint_names"])
    fps = int(states["fps"])
    annotated = all(key in states for key in ("command_m_s", "forward_velocity_m_s", "time_s"))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22) if annotated else None
    if len(positions) == 0 or len(positions) != len(quaternions) or len(positions) != len(angles):
        raise ValueError("Invalid walking state trace")
    if args.velocity_arrows:
        required = ("linear_velocity_w_m_s", "linear_velocity_b_m_s", "command_yaw_rad_s", "yaw_rate_rad_s")
        if any(key not in states for key in required):
            raise ValueError("Velocity arrows need full root-link velocity vectors; capture a new replay trace")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/World/Robot").GetPrim()
    root.GetReferences().AddReference(str(USD_PATH))
    root.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
    geometry = stage.GetPrimAtPath(f"{root.GetPath()}/Geometry")
    by_name = {prim.GetName(): prim for prim in Usd.PrimRange(geometry)}
    child_links = {joint.attrib["name"]: joint.find("child").attrib["link"]
                   for joint in ElementTree.parse(URDF_PATH).getroot().findall("joint")
                   if joint.attrib["name"] in names}
    orient_attrs = {name: by_name[child_links[name]].GetAttribute("xformOp:orient") for name in names}
    neutral_orients = {name: attr.Get() for name, attr in orient_attrs.items()}
    pelvis = by_name["PELVIS"]
    cache = UsdGeom.XformCache()
    pelvis_to_root = (cache.GetLocalToWorldTransform(pelvis)
                      * cache.GetLocalToWorldTransform(root).GetInverse())
    root_op = UsdGeom.Xformable(root).MakeMatrixXform()

    floor = UsdGeom.Cube.Define(stage, "/Render/Floor")
    floor.CreateDisplayColorAttr([Gf.Vec3f(0.31, 0.36, 0.43)])
    floor_xf = UsdGeom.Xformable(floor.GetPrim())
    floor_xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.025))
    floor_xf.AddScaleOp().Set(Gf.Vec3f(100, 100, 0.025))
    dome = UsdLux.DomeLight.Define(stage, "/Render/Dome")
    dome.CreateIntensityAttr(520.0)
    dome.CreateColorAttr(Gf.Vec3f(0.90, 0.94, 1.0))
    key = UsdLux.DistantLight.Define(stage, "/Render/Key")
    key.CreateIntensityAttr(1100.0)
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-42, 28, -38))
    for axis in range(2):
        for index in range(-15, 21):
            line = UsdGeom.Cube.Define(stage, f"/Render/Grid_{axis}_{index + 15}")
            line.CreateDisplayColorAttr([Gf.Vec3f(0.46, 0.53, 0.62)])
            xf = UsdGeom.Xformable(line.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(index, 0, 0.002) if axis == 0 else Gf.Vec3d(0, index, 0.002))
            xf.AddScaleOp().Set(Gf.Vec3f(0.005, 30, 0.002) if axis == 0 else Gf.Vec3f(30, 0.005, 0.002))

    def make_material(name, color):
        material = UsdShade.Material.Define(stage, f"/Render/Materials/{name}")
        shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        from pxr import Sdf
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color) * 0.4)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    def make_arrow(name, material):
        shaft = UsdGeom.Cylinder.Define(stage, f"/Render/Arrows/{name}/Shaft")
        shaft.CreateRadiusAttr(0.016)
        tip = UsdGeom.Cone.Define(stage, f"/Render/Arrows/{name}/Tip")
        tip.CreateRadiusAttr(0.055)
        for shape in (shaft, tip):
            UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material)
        return (shaft, UsdGeom.Xformable(shaft).MakeMatrixXform(),
                tip, UsdGeom.Xformable(tip).MakeMatrixXform())

    def update_arrow(arrow, origin, vector, tip_length=0.12):
        shaft, shaft_op, tip, tip_op = arrow
        length = vector.GetLength()
        visible = UsdGeom.Tokens.inherited if length > 0.025 else UsdGeom.Tokens.invisible
        shaft.GetVisibilityAttr().Set(visible)
        tip.GetVisibilityAttr().Set(visible)
        if length <= 0.025:
            return
        direction = vector / length
        head = min(tip_length, length * 0.4)
        shaft.CreateHeightAttr(length - head)
        tip.CreateHeightAttr(head)
        rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), direction)
        for op, distance in ((shaft_op, (length - head) / 2), (tip_op, length - head / 2)):
            transform = Gf.Matrix4d().SetRotate(rotation)
            transform.SetTranslateOnly(origin + direction * distance)
            op.Set(transform)

    def make_curve(name, material, width=0.018):
        curve = UsdGeom.BasisCurves.Define(stage, f"/Render/Arrows/{name}")
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        curve.CreateWidthsAttr([width])
        curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(material)
        return curve

    def update_yaw(curve, tip, center, heading, rate, radius):
        # Both rates use the same angle scale. Radius separates the two glyphs.
        sweep = max(-2.5, min(2.5, rate * 2.5))
        curve.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if abs(rate) > .015 else UsdGeom.Tokens.invisible)
        angle = np.linspace(heading, heading + sweep, 40)
        points = [Gf.Vec3f(float(center[0] + radius * math.cos(a)),
                           float(center[1] + radius * math.sin(a)), 0.075) for a in angle]
        curve.CreateCurveVertexCountsAttr([len(points)])
        curve.CreatePointsAttr(points)
        end = Gf.Vec3d(*points[-1])
        tangent = Gf.Vec3d(-math.sin(angle[-1]), math.cos(angle[-1]), 0) * (1 if rate >= 0 else -1)
        update_arrow(tip, end - tangent * .10, tangent * (.14 if abs(rate) > .015 else 0), .14)

    if args.velocity_arrows:
        command_material = make_material("Command", (0.05, 0.85, 1.0))
        actual_material = make_material("Actual", (1.0, 0.38, 0.06))
        command_arrow = make_arrow("CommandLinear", command_material)
        actual_arrow = make_arrow("ActualLinear", actual_material)
        command_arc = make_curve("CommandYaw", command_material, .024)
        actual_arc = make_curve("ActualYaw", actual_material, .024)
        command_arc_tip = make_arrow("CommandYawTip", command_material)
        actual_arc_tip = make_arrow("ActualYawTip", actual_material)
        trail = make_curve("MeasuredPath", make_material("Path", (.65, .72, .8)), .012)
        trail_points, previous_case = [], None
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)

    camera = UsdGeom.Camera.Define(stage, "/Render/Camera")
    camera.CreateFocalLengthAttr(40.0)
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateVerticalApertureAttr(36.0 * args.height / args.width)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100))
    camera_op = UsdGeom.Xformable(camera.GetPrim()).MakeMatrixXform()

    rep.orchestrator.set_capture_on_play(False)
    product = rep.create.render_product(str(camera.GetPath()), (args.width, args.height))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([product])
    video_path = output / args.video_name
    report = {"status": "rendering", "renderer": "Isaac Sim RTX RayTracedLighting",
              "physics_state_source": str(state_path), "robot_usd": str(USD_PATH),
              "frames_expected": len(positions), "fps": fps,
              "resolution": [args.width, args.height], "video": str(video_path)}
    if args.velocity_arrows:
        report["arrows"] = {
            "cyan": "Command", "orange": "Actual measured root-link velocity",
            "linear_vector_frame": "World XY; command rotated from body X by the recorded quaternion",
            "linear_scale_m_per_m_s": 1.25, "yaw_angle_scale_s": 2.5,
            "velocity_smoothing": "None; frame-sampled instantaneous measurements",
            "path": "Measured root-link position projected onto the floor",
        }
    report_path = output / "render.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    try:
        with imageio.get_writer(str(video_path), fps=fps, codec="libx264", quality=8,
                                macro_block_size=2, ffmpeg_params=["-movflags", "+faststart"]) as writer:
            for index, (position, quaternion, joint_angles) in enumerate(zip(positions, quaternions, angles)):
                for name, angle in zip(names, joint_angles):
                    half = float(angle) * 0.5
                    rotation = Gf.Quatf(math.cos(half), Gf.Vec3f(0, 0, math.sin(half)))
                    orient_attrs[name].Set(neutral_orients[name] * rotation)
                desired = Gf.Matrix4d()
                desired.SetRotate(Gf.Quatd(float(quaternion[3]), Gf.Vec3d(*map(float, quaternion[:3]))))
                desired.SetTranslateOnly(Gf.Vec3d(*map(float, position)))
                root_op.Set(pelvis_to_root.GetInverse() * desired)
                if args.velocity_arrows:
                    world_quat = Gf.Quatd(float(quaternion[3]), Gf.Vec3d(*map(float, quaternion[:3])))
                    command_vector = world_quat.Transform(Gf.Vec3d(float(states["command_m_s"][index]), 0, 0))
                    command_vector[2] = 0.0
                    measured = states["linear_velocity_w_m_s"][index]
                    actual_vector = Gf.Vec3d(float(measured[0]), float(measured[1]), 0)
                    update_arrow(command_arrow, Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]) + .62), command_vector * 1.25)
                    update_arrow(actual_arrow, Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]) + .50), actual_vector * 1.25)
                    forward = world_quat.Transform(Gf.Vec3d(1, 0, 0))
                    heading = math.atan2(forward[1], forward[0])
                    update_yaw(command_arc, command_arc_tip, position, heading, float(states["command_yaw_rad_s"][index]), .66)
                    update_yaw(actual_arc, actual_arc_tip, position, heading, float(states["yaw_rate_rad_s"][index]), .84)
                    case_id = int(states["case_index"][index]) if "case_index" in states else 0
                    if case_id != previous_case:
                        trail_points = []
                        previous_case = case_id
                    trail_points.append(Gf.Vec3f(float(position[0]), float(position[1]), .018))
                    path_points = trail_points if len(trail_points) >= 2 else trail_points * 2
                    trail.CreateCurveVertexCountsAttr([len(path_points)])
                    trail.CreatePointsAttr(path_points)
                eye = (float(position[0]) + 3.0, float(position[1]) - 3.6, 2.2)
                target = (float(position[0]), float(position[1]), 0.85)
                camera_op.Set(look_at(eye, target))
                if index == 0:
                    for _ in range(45):
                        app.update()
                rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=True)
                pixels = np.asarray(annotator.get_data())
                if pixels.size == 0:
                    raise RuntimeError(f"Empty RTX frame {index}")
                frame = Image.fromarray(pixels[..., :3].astype(np.uint8))
                if annotated:
                    draw = ImageDraw.Draw(frame)
                    if args.velocity_arrows:
                        command_yaw = float(states["command_yaw_rad_s"][index])
                        case_name = "LEFT TURN" if command_yaw > .01 else "RIGHT TURN" if command_yaw < -.01 else "STRAIGHT"
                        draw.rectangle((0, 0, args.width, 100), fill=(20, 26, 33))
                        draw.text((18, 7), f"{args.title} | {case_name} | t={states['time_s'][index]:.2f}s", font=font, fill="white")
                        draw.text((18, 37), f"CMD  vx {states['command_m_s'][index]:+.2f} m/s   yaw {command_yaw:+.2f} rad/s", font=font, fill=(40, 220, 255))
                        draw.text((660, 37), f"ACT  vx {states['body_forward_velocity_m_s'][index]:+.2f} m/s   yaw {states['yaw_rate_rad_s'][index]:+.2f} rad/s", font=font, fill=(255, 145, 65))
                        draw.text((18, 73), "CYAN = COMMAND   |   ORANGE = ACTUAL   |   Straight arrows: planar velocity   |   Curved arrows: yaw rate", font=small_font, fill=(205, 215, 225))
                    else:
                        annotate_legacy(draw, states, index, font)
                writer.append_data(np.asarray(frame))
                if index in (29, 119, len(positions) - 1):
                    frame.save(output / f"walk_frame_{index + 1:04d}.png")
                if (index + 1) % 60 == 0:
                    print(f"ISAAC_RTX_FRAMES {index + 1}/{len(positions)}", flush=True)
        report.update(status="complete", frames_rendered=len(positions),
                      video_bytes=video_path.stat().st_size)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"ISAAC_RTX_VIDEO_COMPLETE {video_path}", flush=True)
    except BaseException:
        import traceback

        report.update(status="failed", traceback=traceback.format_exc())
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        annotator.detach()
        product.destroy()
        rep.orchestrator.wait_until_complete()


try:
    main()
finally:
    app.close()
