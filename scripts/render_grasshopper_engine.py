#!/usr/bin/env python3
"""Load a multi-physics robot USD, run one backend, and render an orbit video."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", required=True, help="Imported multi-physics robot USD")
    parser.add_argument("--engine", required=True, choices=("physx", "newton"))
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--frames-dir", help="Directory for source PNG frames")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--settle-frames", type=int, default=200)
    parser.add_argument(
        "--fall-view",
        action="store_true",
        help="Use a lower, wider camera composition that keeps a fallen robot in frame",
    )
    return parser.parse_args()


ARGS = parse_args()
simulation_app = SimulationApp(
    {
        "headless": True,
        "width": ARGS.width,
        "height": ARGS.height,
        "renderer": "RayTracedLighting",
        "fast_shutdown": True,
    }
)


def look_at_matrix(eye, target):
    from pxr import Gf

    eye = Gf.Vec3d(*eye)
    target = Gf.Vec3d(*target)
    up = Gf.Vec3d(0, 0, 1)
    forward = (target - eye).GetNormalized()
    if abs(forward * up) > 0.99:
        up = Gf.Vec3d(0, 1, 0)
    right = (forward ^ up).GetNormalized()
    camera_up = (right ^ forward).GetNormalized()
    matrix = Gf.Matrix4d()
    matrix[0] = [right[0], right[1], right[2], 0]
    matrix[1] = [camera_up[0], camera_up[1], camera_up[2], 0]
    matrix[2] = [-forward[0], -forward[1], -forward[2], 0]
    matrix[3] = [eye[0], eye[1], eye[2], 1]
    return matrix


def wait_for_stage() -> None:
    import omni.usd

    context = omni.usd.get_context()
    for _ in range(600):
        simulation_app.update()
        loading = context.get_stage_loading_status()
        if not loading or loading[2] == 0:
            return
    raise RuntimeError("USD stage did not finish loading")


def compute_robot_bounds(stage, root):
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    aligned = cache.ComputeWorldBound(root).ComputeAlignedRange()
    minimum = aligned.GetMin()
    maximum = aligned.GetMax()
    if aligned.IsEmpty():
        raise RuntimeError("Robot has no renderable world bounds")
    return tuple(float(v) for v in minimum), tuple(float(v) for v in maximum)


def create_render_scene(stage, bounds, fall_view=False):
    import carb
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

    minimum, maximum = bounds
    center = tuple((minimum[i] + maximum[i]) * 0.5 for i in range(3))
    size = tuple(maximum[i] - minimum[i] for i in range(3))
    max_span = max(size)
    horizontal_span = max(size[0], size[1])
    camera_distance = max(3.4 * horizontal_span, 3.2 * size[2], 3.5)
    if fall_view:
        camera_distance *= 1.2
    floor_half_extent = max(camera_distance * 1.3, max_span * 2.0, 4.0)

    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "RayTracedLighting")
    settings.set("/rtx/post/tonemap/op", 4)
    settings.set("/rtx/post/tonemap/filmIso", 200.0)
    settings.set("/rtx/post/tonemap/whitepoint", 6500.0)
    settings.set("/rtx/post/tonemap/enabled", True)
    settings.set("/rtx/post/aa/op", 3)

    render_root = UsdGeom.Xform.Define(stage, "/Render")

    dome = UsdLux.DomeLight.Define(stage, "/Render/DomeLight")
    dome.CreateIntensityAttr(300.0)
    dome.CreateColorAttr(Gf.Vec3f(0.82, 0.88, 1.0))

    sun = UsdLux.DistantLight.Define(stage, "/Render/DistantLight")
    sun.CreateIntensityAttr(1200.0)
    sun.CreateAngleAttr(1.0)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-50, 20, -25))

    floor_parent = UsdGeom.Xform.Define(stage, "/Render/Floor")
    UsdGeom.Xformable(floor_parent.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(center[0], center[1], minimum[2] - 0.04)
    )
    floor = UsdGeom.Cube.Define(stage, "/Render/Floor/Surface")
    floor.CreateSizeAttr(2.0)
    floor.CreateDisplayColorAttr([Gf.Vec3f(0.10, 0.13, 0.18)])
    UsdGeom.Xformable(floor.GetPrim()).AddScaleOp().Set(Gf.Vec3f(floor_half_extent, floor_half_extent, 0.02))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())

    camera = UsdGeom.Camera.Define(stage, "/Render/Camera")
    camera.CreateFocalLengthAttr(35.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, max(100.0, camera_distance * 10.0)))
    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_xform.ClearXformOpOrder()
    camera_op = camera_xform.MakeMatrixXform()

    if fall_view:
        target = (center[0], center[1], minimum[2] + size[2] * 0.30)
    else:
        target = (center[0], center[1], center[2] + size[2] * 0.05)
    return camera, camera_op, center, size, camera_distance, target


def count_stage_content(stage):
    from pxr import Usd, UsdGeom, UsdPhysics

    prims = tuple(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))
    return {
        "meshes": sum(prim.IsA(UsdGeom.Mesh) for prim in prims),
        "joints": sum(prim.IsA(UsdPhysics.Joint) for prim in prims),
        "rigid_bodies": sum(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in prims),
        "articulation_roots": sum(prim.HasAPI(UsdPhysics.ArticulationRootAPI) for prim in prims),
    }


def frame_metrics(array) -> dict:
    import numpy as np

    rgb = np.asarray(array)[..., :3]
    return {
        "mean_rgb": round(float(rgb.mean()), 3),
        "max_rgb": int(rgb.max()),
        "std_rgb": round(float(rgb.std()), 3),
    }


def validate_metrics(metrics: dict, label: str) -> None:
    if metrics["max_rgb"] == 0:
        raise RuntimeError(f"{label} frame is black")
    if metrics["mean_rgb"] < 30:
        raise RuntimeError(f"{label} frame is too dark: {metrics}")
    if metrics["mean_rgb"] > 220:
        raise RuntimeError(f"{label} frame is overexposed: {metrics}")
    if metrics["std_rgb"] < 15:
        raise RuntimeError(f"{label} frame is too flat: {metrics}")


def encode_video(frame_paths, output_path: Path, fps: int) -> None:
    from video_encoding import get_video_encoding_interface

    encoder = get_video_encoding_interface()
    if encoder is None:
        raise RuntimeError("Isaac Sim video encoding interface is unavailable")
    if not encoder.start_encoding(str(output_path), fps, len(frame_paths), True):
        raise RuntimeError("NVENC could not start MP4 encoding")
    try:
        for path in frame_paths:
            encoder.encode_next_frame_from_file(str(path))
    finally:
        encoder.finalize_encoding()
    if not output_path.is_file() or output_path.stat().st_size < 150_000:
        raise RuntimeError(f"Encoded video is missing or too small: {output_path}")


def run() -> dict:
    import carb
    import numpy as np
    import omni.kit.app
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from PIL import Image
    from isaacsim.core.simulation_manager import SimulationManager

    usd_path = Path(ARGS.usd).expanduser().resolve()
    output_path = Path(ARGS.output).expanduser().resolve()
    frames_dir = (
        Path(ARGS.frames_dir).expanduser().resolve()
        if ARGS.frames_dir
        else output_path.with_name(f"{output_path.stem}_frames")
    )
    if not usd_path.is_file():
        raise FileNotFoundError(usd_path)
    print(f"[render] opening {usd_path}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("frame_*.png"):
        old_frame.unlink()

    if ARGS.engine == "newton":
        extension_manager = omni.kit.app.get_app().get_extension_manager()
        extension_manager.set_extension_enabled_immediate("isaacsim.physics.newton", True)
        extension_manager.set_extension_enabled_immediate("isaacsim.physics.newton.tensors", True)
        for _ in range(10):
            simulation_app.update()

    if not simulation_app.context.open_stage(str(usd_path)):
        raise RuntimeError(f"Could not open USD stage: {usd_path}")
    wait_for_stage()
    stage = omni.usd.get_context().get_stage()
    print("[render] stage loaded", flush=True)
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        raise RuntimeError("Imported USD has no valid default prim")

    physics_variants = root.GetVariantSets().GetVariantSet("Physics")
    available_variants = list(physics_variants.GetVariantNames())
    requested_variant = "physx" if ARGS.engine == "physx" else "mujoco"
    if requested_variant not in available_variants:
        raise RuntimeError(
            f"Physics variant {requested_variant!r} is unavailable; found {available_variants}"
        )
    if not physics_variants.SetVariantSelection(requested_variant):
        raise RuntimeError(f"Could not select Physics={requested_variant}")
    for _ in range(5):
        simulation_app.update()

    if not SimulationManager.switch_physics_engine(ARGS.engine, verbose=True):
        raise RuntimeError(f"Could not switch to {ARGS.engine}")
    active_engine = SimulationManager.get_active_physics_engine()
    if active_engine != ARGS.engine:
        raise RuntimeError(f"Requested {ARGS.engine}, active engine is {active_engine}")

    counts = count_stage_content(stage)
    print(f"[render] active_engine={active_engine} counts={counts}", flush=True)
    if counts["meshes"] < 1 or counts["joints"] < 1 or counts["rigid_bodies"] < 1:
        raise RuntimeError(f"Incomplete robot load: {counts}")

    bounds = compute_robot_bounds(stage, root)
    camera, camera_op, center, size, camera_distance, target = create_render_scene(
        stage, bounds, fall_view=ARGS.fall_view
    )

    rep.orchestrator.set_capture_on_play(False)
    render_product = rep.create.render_product(str(camera.GetPath()), (ARGS.width, ARGS.height))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach([render_product])

    SimulationManager.initialize_physics()
    print("[render] physics initialized", flush=True)
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()

    start_angle = math.radians(-40.0)
    initial_eye = (
        center[0] + camera_distance * math.cos(start_angle),
        center[1] + camera_distance * math.sin(start_angle),
        center[2] + max(size[2] * 0.28, camera_distance * 0.12),
    )
    camera_op.Set(look_at_matrix(initial_eye, target))
    for _ in range(ARGS.settle_frames):
        simulation_app.update()
    print("[render] RTX warm-up complete", flush=True)
    timeline.play()

    frame_paths = []
    selected_metrics = {}
    metric_indices = {0: "start", ARGS.frames // 2: "middle", ARGS.frames - 1: "end"}
    physics_hz = 60
    total_physics_steps = 0

    for frame_index in range(ARGS.frames):
        angle = start_angle + 2.0 * math.pi * frame_index / ARGS.frames
        camera_z = center[2] + max(size[2] * 0.28, camera_distance * 0.12)
        camera_z += size[2] * 0.05 * math.sin(angle * 2.0)
        eye = (
            center[0] + camera_distance * math.cos(angle),
            center[1] + camera_distance * math.sin(angle),
            camera_z,
        )
        camera_op.Set(look_at_matrix(eye, target))

        desired_total = math.floor((frame_index + 1) * physics_hz / ARGS.fps)
        steps = max(1, desired_total - total_physics_steps)
        SimulationManager.step(steps=steps)
        total_physics_steps += steps
        simulation_app.update()

        data = None
        for attempt in range(4):
            rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=False)
            candidate = rgb_annotator.get_data()
            if isinstance(candidate, np.ndarray) and candidate.size:
                data = candidate
                break
            carb.log_warn(f"Empty RGB frame {frame_index}, retry {attempt + 1}/4")
        if data is None:
            raise RuntimeError(f"No RGB data for frame {frame_index}")

        rgb = np.asarray(data)[..., :3].astype(np.uint8, copy=False)
        frame_path = frames_dir / f"frame_{frame_index:05d}.png"
        Image.fromarray(rgb, mode="RGB").save(frame_path, compress_level=3)
        frame_paths.append(frame_path)
        if frame_index in metric_indices:
            label = metric_indices[frame_index]
            selected_metrics[label] = frame_metrics(rgb)
            validate_metrics(selected_metrics[label], label)
        if frame_index == 0 or (frame_index + 1) % max(1, ARGS.fps) == 0:
            print(f"[render] captured {frame_index + 1}/{ARGS.frames}", flush=True)

    rep.orchestrator.wait_until_complete()
    rgb_annotator.detach()
    render_product.destroy()
    timeline.stop()

    expected_steps = math.floor(ARGS.frames * physics_hz / ARGS.fps)
    if total_physics_steps < expected_steps or total_physics_steps / physics_hz < 3.0:
        raise RuntimeError(f"Physics ran for only {total_physics_steps / physics_hz:.3f}s")

    encode_video(frame_paths, output_path, ARGS.fps)
    report = {
        "status": "passed",
        "engine": ARGS.engine,
        "active_engine": active_engine,
        "physics_variant": requested_variant,
        "available_physics_variants": available_variants,
        "usd": str(usd_path),
        "output": str(output_path),
        "frames_dir": str(frames_dir),
        "resolution": [ARGS.width, ARGS.height],
        "fps": ARGS.fps,
        "frames": len(frame_paths),
        "duration_seconds": len(frame_paths) / ARGS.fps,
        "physics_seconds": total_physics_steps / physics_hz,
        "fall_view": ARGS.fall_view,
        "counts": counts,
        "bounds": {"minimum": bounds[0], "maximum": bounds[1]},
        "frame_metrics": selected_metrics,
        "video_bytes": output_path.stat().st_size,
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("RESULT_JSON=" + json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    failure = None
    try:
        run()
    except BaseException as exc:
        import traceback

        failure = exc
        traceback.print_exc()
    finally:
        if failure is None:
            simulation_app.close()
        else:
            simulation_app.close(exit_code=1)
    if failure is not None:
        raise failure
