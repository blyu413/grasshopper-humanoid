#!/usr/bin/env python3
"""Compare the shipped Grasshopper URDFs and render them through Newton's MJCF export."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "assets/robots/grasshopper_torso/grasshopper_lower_body.urdf"
NEW = ROOT / "assets/robots/grasshopper_torso_arm/grasshopper_latest_tesollo_description/grasshopper_latest_tesollo.urdf"


def canonical(element):
    if element is None:
        return None
    attributes = {}
    for key, value in element.attrib.items():
        try:
            attributes[key] = [round(float(v), 12) for v in value.split()]
        except ValueError:
            attributes[key] = value
    return {"tag": element.tag, "attributes": attributes,
            "children": [canonical(child) for child in element]}


def mesh_path(source, filename):
    if (filename.startswith("file:") or Path(filename).is_absolute()) and "/meshes/" in filename:
        return source.parent / "meshes" / filename.split("/meshes/", 1)[1]
    return source.parent / filename


def summary(source):
    root = ET.parse(source).getroot()
    links, joints = root.findall("link"), root.findall("joint")
    movable = [j for j in joints if j.get("type") != "fixed"]
    return {
        "source": str(source), "robot_name": root.get("name"),
        "links": len(links), "joints": len(joints),
        "joint_types": dict(Counter(j.get("type") for j in joints)),
        "independent_joints": sum(j.find("mimic") is None for j in movable),
        "mimic_joints": {j.get("name"): j.find("mimic").attrib for j in movable if j.find("mimic") is not None},
        "mass_kg": sum(float(m.get("value")) for m in root.findall("link/inertial/mass")),
        "visuals": len(root.findall("link/visual")),
        "collisions": len(root.findall("link/collision")),
        "collision_geometry": dict(Counter(g[0].tag for g in root.findall("link/collision/geometry"))),
        "unresolved_authored_mesh_references": sum(not (source.parent / m.get("filename")).is_file() for m in root.findall(".//mesh")),
    }


def compare():
    roots = [ET.parse(p).getroot() for p in (OLD, NEW)]
    old_links, new_links = [{e.get("name"): e for e in r.findall("link")} for r in roots]
    old_joints, new_joints = [{e.get("name"): e for e in r.findall("joint")} for r in roots]
    changes = {}
    for name in sorted(old_joints.keys() & new_joints.keys()):
        a, b = old_joints[name], new_joints[name]
        fields = {}
        for tag in ("origin", "axis", "parent", "child", "limit", "dynamics", "mimic"):
            if canonical(a.find(tag)) != canonical(b.find(tag)):
                fields[tag] = {"old": canonical(a.find(tag)), "new": canonical(b.find(tag))}
        if fields:
            changes[name] = fields
    inertial_changes = [name for name in sorted(old_links.keys() & new_links.keys())
                        if canonical(old_links[name].find("inertial")) != canonical(new_links[name].find("inertial"))]
    mesh_changes = {}
    for name in sorted(old_links.keys() & new_links.keys()):
        a = old_links[name].findall("visual/geometry/mesh")
        b = new_links[name].findall("visual/geometry/mesh")
        def hashes(meshes, source):
            return [hashlib.sha256(mesh_path(source, m.get("filename")).read_bytes()).hexdigest() for m in meshes]
        if hashes(a, OLD) != hashes(b, NEW):
            mesh_changes[name] = {"old": [m.get("filename") for m in a], "new": [m.get("filename") for m in b]}
    additions = [j for n, j in new_joints.items() if n not in old_joints and j.get("type") != "fixed"]
    groups = Counter("hands" if "_hand_" in j.get("name") else "head" if j.get("name").startswith("HEAD_")
                     else "wrist_mimics" if j.find("mimic") is not None else "arms" for j in additions)
    return {"old": summary(OLD), "latest": summary(NEW),
            "added_links": sorted(new_links.keys() - old_links.keys()),
            "removed_links": sorted(old_links.keys() - new_links.keys()),
            "added_joints": sorted(new_joints.keys() - old_joints.keys()),
            "removed_joints": sorted(old_joints.keys() - new_joints.keys()),
            "added_movable_joint_groups": dict(groups),
            "shared_joint_changes": changes, "shared_link_inertial_changes": inertial_changes,
            "shared_link_visual_mesh_content_changes": mesh_changes,
            "arm_limits": {n: j.find("limit").attrib for n, j in new_joints.items()
                           if n.startswith(("LA_", "RA_")) and "_hand_" not in n and j.get("type") != "fixed"}}


def localize(source, target):
    tree = ET.parse(source)
    for mesh in tree.getroot().findall(".//mesh"):
        path = mesh_path(source, mesh.get("filename"))
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set("filename", os.path.relpath(path, target.parent))
    tree.write(target, encoding="utf-8", xml_declaration=True)


def smoke_test(source, output):
    import numpy as np
    import warp as wp
    import newton
    from newton.solvers import SolverMuJoCo

    wp.config.kernel_cache_dir = "/tmp/nest_mjcf_warp_cache"
    wp.init()
    builder = newton.ModelBuilder()
    collapse = source.name == "grasshopper_latest_local.urdf"
    if collapse:
        builder.default_shape_cfg.density = 0.0
    SolverMuJoCo.register_custom_attributes(builder)
    builder.add_urdf(str(source), floating=True, collapse_fixed_joints=False,
                     ignore_inertial_definitions=False, hide_visuals=False)
    builder.joint_target_mode[:] = [int(newton.JointTargetMode.NONE)] * len(builder.joint_target_mode)
    if collapse:
        builder.collapse_fixed_joints()
    model = builder.finalize(device="cpu")
    solver = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True, skip_visual_only_geoms=False)
    source_mass = summary(source)["mass_kg"]
    for value in (float(model.body_mass.numpy().sum()), float(solver.mj_model.body_mass.sum())):
        if not np.isclose(value, source_mass, rtol=1e-6):
            raise RuntimeError(f"Runtime mass {value} does not match declared URDF mass {source_mass}")
    state, next_state, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    initial = state.body_q.numpy().copy()
    dt, steps = 0.001, 100
    for _ in range(steps):
        state.clear_forces()
        solver.step(state, next_state, control, None, dt)
        state, next_state = next_state, state
        if not np.isfinite(state.body_q.numpy()).all() or not np.isfinite(state.joint_q.numpy()).all():
            raise RuntimeError("Newton smoke test produced non-finite state")
    report = {"backend": "Newton SolverMuJoCo, CPU", "steps": steps, "dt": dt,
              "seconds": steps * dt, "contacts_enabled": False,
              "scenario": "Uncontrolled free fall without ground; load/integration validation only",
              "all_states_finite": True, "bodies": model.body_count,
              "joint_coordinates": model.joint_coord_count, "joint_dofs": model.joint_dof_count,
              "total_mass_kg": float(model.body_mass.numpy().sum()),
              "solver_mass_kg": float(solver.mj_model.body_mass.sum()),
              "fixed_joints_collapsed": collapse,
              "base_z_displacement_m": float(state.body_q.numpy()[0, 2] - initial[0, 2]),
              "mimic_equalities": int(solver.mj_model.neq)}
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def montage(output):
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    width, height, label = 640, 900, 60
    sheet = Image.new("RGB", (width * 2, height + label), "#edf0f4")
    draw = ImageDraw.Draw(sheet)
    bounds = [json.loads((output / name / "export_report.json").read_text())["bounds"] for name in ("old", "latest")]
    lower = np.min([b["min"] for b in bounds], axis=0)
    upper = np.max([b["max"] for b in bounds], axis=0)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (lower + upper) / 2
    camera.distance = float(np.max(upper - lower) * 1.72)
    camera.azimuth, camera.elevation = 35, -12
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[0] = option.geomgroup[2] = 1
    for i, (folder, caption) in enumerate((("old", "Original: lower body + torso"), ("latest", "Latest: full body + Tesollo hands"))):
        model = mujoco.MjModel.from_xml_path(str(output / folder / "grasshopper_preview.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            renderer.update_scene(data, camera=camera, scene_option=option)
            frame = Image.fromarray(renderer.render())
        frame.save(output / folder / "matched_scale.png")
        sheet.paste(frame, (i * width, label))
        draw.text((i * width + 16, 18), caption, fill="#243044", font=ImageFont.load_default(size=22))
    sheet.save(output / "old_vs_latest.png")


def write_report(output, report):
    old, new = report["old"], report["latest"]
    lines = ["# Grasshopper 新旧模型对比与 Newton 加载验证", "",
             "对比日期：2026-10-22。旧模型为项目当前使用的 `grasshopper_lower_body.urdf`；新版为解压包中的 `grasshopper_latest_tesollo.urdf`。", "",
             "| 项目 | 原模型 | 新版 |", "|---|---:|---:|"]
    for label, key in (("连杆", "links"), ("关节总数", "joints"), ("独立可动关节（排除 mimic）", "independent_joints"),
                       ("视觉元素（URDF）", "visuals"), ("碰撞元素（URDF）", "collisions")):
        lines.append(f"| {label} | {old[key]} | {new[key]} |")
    lines.extend([f"| 总质量（kg） | {old['mass_kg']:.5f} | {new['mass_kg']:.5f} |", "",
                  f"质量增加 {new['mass_kg'] - old['mass_kg']:.5f} kg（{(new['mass_kg'] / old['mass_kg'] - 1) * 100:.2f}%）。",
                  "可动关节由 15 个增至 73 个；mimic 由 1 个增至 3 个。新增头部 2 个、双臂 14 个、双手手指 40 个独立关节，以及 2 个腕部 mimic。自由基座另外提供 6 个速度自由度。", "",
                  "## 影响控制的变化", "",
                  "以下关节的局部轴由 `(0, 0, 1)` 变为 `(0, 0, -1)`：", ""])
    lines.extend(f"- `{name}`" for name in report["shared_joint_changes"])
    lines.extend(["", "这些关节的角度、角速度和力矩符号约定随之改变；迁移旧参考姿态或策略时需要核对符号映射。共享关节的原点、父子连接、限位、effort、velocity、阻尼、摩擦和腰部 mimic 参数均保持一致。", "",
                  "两脚原有的 14 个胶囊（每脚 7 个）变为两块脚部网格。原模型碰撞为 30 mesh + 14 capsule；新版为 116 mesh + 24 box + 18 cylinder。新版不会继承项目此前定制的脚底接触布局。", "",
                  "原模型的 47 个同名连杆，其惯性定义及视觉网格 SHA-256 均未发生变化。旧 `TORSO` 替换为 `torso`，二者仍为 20 kg，惯性及视觉网格内容一致；新版通过 `LB_TORSO_IFC` 连接。目录从 `v0_1` 改成 `v1_0` 本身不代表这些网格发生了变化。", "",
                  "新版左右臂限位明显不对称，应分别使用其 URDF 参数。比如肩俯仰：左 194 N·m、3.979 rad/s、±3.14 rad；右 110 N·m、20.944 rad/s、±1.5208 rad。这是源文件现状，不能假设左右通用。", "",
                  "现有步行/站立策略为 14 维下半身策略。新版包含额外关节且质量、脚部接触、部分轴方向发生变化，不能直接替换模型并认为原策略兼容；本次未改动训练配置。", "",
                  "## Newton 加载与渲染", "",
                  "新版源 URDF 有 242 处 mesh 引用指向导出者机器的绝对路径。生成本地适配 URDF，将所有引用改为指向项目资源的相对路径，并逐个确认文件存在；原始模型保持原样。", "",
                  "加载链路：Newton ModelBuilder.add_urdf → finalize(cpu) → Newton SolverMuJoCo。新旧模型均进行 100 步、每步 0.001 s 的无控制自由落体测试，关闭接触并且无地面，只验证模型加载与积分；该检查不证明站立、行走或接触稳定性。", ""])
    lines.extend(["新版 `torso_casing` 和 `HEAD_front_cover` 有碰撞几何但无惯性声明。默认导入会按密度增加质量，MuJoCo 还可能对零质量节点再次推算惯性。当前适配将默认几何密度设为零，并合并固定关节；可动关节、形状和显式质量保持，导入后同时检查 Newton 与求解器质量等于 URDF 声明质量。128.22 kg 是已声明惯性之和，不代表已核实这两处外壳的真实质量。", ""])
    for folder in ("old", "latest"):
        result = json.loads((output / folder / "newton_smoke_test.json").read_text())
        exported = json.loads((output / folder / "export_report.json").read_text())
        lines.append(f"- {folder}：Newton bodies={result['bodies']}，速度自由度={result['joint_dofs']}，mimic 等式={result['mimic_equalities']}，状态均有限；基座下落 {abs(result['base_z_displacement_m']):.6f} m。导入视觉 shape={exported['visual_geoms']}，碰撞 shape={exported['collision_geoms']}，视觉覆盖校验通过。")
    lines.extend(["", "当前环境无法访问 NVIDIA 驱动，因此实际运行的是 Newton 的 MuJoCo CPU 后端；未验证 CUDA/MJWarp。图片为经过 Newton 导出的几何与材料，使用 MuJoCo EGL/Mesa 离屏渲染，不是 Newton 原生 Viewer 的截图。", "",
                  "渲染采用源文件零关节姿态，无动态 rollout。`old_vs_latest.png` 使用相同相机、相同米制比例；`latest/grasshopper_mjcf_four_views.png` 为新版四视图。部分 Tesollo Collada 文件含多个子网格，所以一个 URDF visual 可导入为多个 shape；导出校验已改为逐连杆确认覆盖。", "",
                  "## 复现", "", "```bash", ".venv/bin/python scripts/compare_grasshopper_models.py", "```", "",
                  "结构化完整差异见 `comparison.json`；每个模型目录内有适配 URDF、Newton 导出 MJCF、渲染场景、四视图、加载日志、export_report.json 和 newton_smoke_test.json。"])
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/grasshopper_model_comparison_20261022")
    parser.add_argument("--skip-export", action="store_true", help="Reuse previously exported/rendered MJCFs")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    if Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json").exists():
        os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = compare()
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    for label, source in (("old", OLD), ("latest", NEW)):
        folder = output / label
        folder.mkdir(exist_ok=True)
        local = folder / f"grasshopper_{label}_local.urdf"
        localize(source, local)
        if not args.skip_export:
            env = dict(os.environ, MUJOCO_GL="egl", EGL_PLATFORM="surfaceless")
            if Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json").exists():
                env["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
            with (folder / "load_render.log").open("w") as log:
                subprocess.run([sys.executable, str(ROOT / "scripts/export_grasshopper_mjcf.py"),
                                "--urdf", str(local), "--output-dir", str(folder), "--width", "640", "--height", "900",
                                *(["--collapse-fixed-joints"] if label == "latest" else [])],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(label, smoke_test(local, folder / "newton_smoke_test.json"), flush=True)
    montage(output)
    write_report(output, report)
    print("Saved comparison and renders:", output, flush=True)


if __name__ == "__main__":
    main()
