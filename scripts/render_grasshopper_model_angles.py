#!/usr/bin/env python3
"""Render additional camera angles of the Newton-exported full-body Grasshopper."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
if Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json").is_file():
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "outputs/grasshopper_model_comparison_20261022/latest"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT)
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    output = model_dir / "more_angles"
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((model_dir / "export_report.json").read_text())
    lower, upper = np.array(report["bounds"]["min"]), np.array(report["bounds"]["max"])
    center, span = (lower + upper) / 2, float(np.max(upper - lower))
    model = mujoco.MjModel.from_xml_path(str(model_dir / "grasshopper_preview.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[0] = option.geomgroup[2] = 1
    camera = mujoco.MjvCamera()
    views = [
        ("01_front", "Front", 0, -10),
        ("02_front_left", "Front left", 45, -10),
        ("03_left", "Left", 90, -10),
        ("04_rear_left", "Rear left", 135, -10),
        ("05_rear", "Rear", 180, -10),
        ("06_rear_right", "Rear right", 225, -10),
        ("07_right", "Right", 270, -10),
        ("08_front_right", "Front right", 315, -10),
        ("09_high_front", "High front", 35, -35),
        ("10_high_rear", "High rear", 215, -35),
        ("11_top", "Top", 0, -75),
        ("12_low_front", "Low front", 325, 10),
    ]
    metadata = {"source_scene": str(model_dir / "grasshopper_preview.xml"),
                "renderer": "MuJoCo EGL/Mesa, using the Newton-exported model",
                "pose": "Authored zero joint pose; camera changes only", "views": []}
    frames = []
    with mujoco.Renderer(model, height=960, width=720) as renderer:
        for filename, title, azimuth, elevation in views:
            camera.lookat[:] = center
            camera.distance = span * 1.72
            camera.azimuth, camera.elevation = azimuth, elevation
            renderer.update_scene(data, camera=camera, scene_option=option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
            pixels = renderer.render().copy()
            if not np.isfinite(pixels).all() or pixels.std() < 10:
                raise RuntimeError(f"Invalid render: {filename}")
            frame = Image.fromarray(pixels)
            frame.save(output / f"{filename}.png")
            frames.append((title, frame))
            metadata["views"].append({"file": f"{filename}.png", "azimuth": azimuth, "elevation": elevation,
                                      "lookat": center.tolist(), "distance": camera.distance})
            print(f"Rendered {filename}", flush=True)
    tile_w, tile_h, header = 360, 480, 40
    sheet = Image.new("RGB", (tile_w * 4, (tile_h + header) * 3), "#edf0f4")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=20)
    for i, (title, frame) in enumerate(frames):
        x, y = (i % 4) * tile_w, (i // 4) * (tile_h + header)
        draw.text((x + 14, y + 9), title, fill="#243044", font=font)
        sheet.paste(frame.resize((tile_w, tile_h), Image.Resampling.LANCZOS), (x, y + header))
    sheet.save(output / "grasshopper_12_angles.png")
    (output / "render_report.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved: {output / 'grasshopper_12_angles.png'}", flush=True)


if __name__ == "__main__":
    main()
