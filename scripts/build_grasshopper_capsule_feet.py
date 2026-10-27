#!/usr/bin/env python3
"""Build the shared USD foot-collision override from the Grasshopper URDF.

MJLab G1 uses seven longitudinal capsules per foot, with a 1 cm radius.
The URDF fits that layout to Grasshopper's mirrored foot frames and sole bounds.
This layer reuses the existing visual/physics asset and overrides only foot
colliders, preserving authored rigid-body mass properties and all joints.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from xml.etree import ElementTree

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/robots/grasshopper_torso/grasshopper_lower_body.urdf"
BASE_USD = ROOT / "assets/robots/grasshopper_torso/usd/grasshopper_lower_body/grasshopper_lower_body.usda"
CAPSULE_USD = ROOT / "assets/robots/grasshopper_torso/grasshopper_lower_body_capsule_feet.usda"


def rpy_quaternion(rpy):
    roll, pitch, yaw = (value / 2 for value in rpy)
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return Gf.Quatf(cr * cp * cy + sr * sp * sy, Gf.Vec3f(
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ))


def build_capsule_feet(base_usd=BASE_USD, output_usd=CAPSULE_USD, urdf=URDF):
    base_usd, output_usd = Path(base_usd).resolve(), Path(output_usd).resolve()
    if base_usd == output_usd:
        raise ValueError("The capsule override must be separate from the base USD")
    if not base_usd.is_file():
        raise FileNotFoundError(base_usd)
    output_usd.parent.mkdir(parents=True, exist_ok=True)
    # Construct in memory so failed builds cannot truncate the existing asset.
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/grasshopper_lower_body").GetPrim()
    root.GetReferences().AddReference(str(base_usd))
    root.GetVariantSet("Physics").SetVariantSelection("mujoco")
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetMetadata("kilogramsPerUnit", 1.0)
    material = UsdShade.Material.Define(stage, "/grasshopper_lower_body/sole_contact_material")
    contact = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    contact.CreateStaticFrictionAttr(0.6)
    contact.CreateDynamicFrictionAttr(0.6)
    contact.CreateRestitutionAttr(0.0)
    material.GetPrim().CreateAttribute("mjc:priority", Sdf.ValueTypeNames.Int).Set(1)
    robot_xml = ElementTree.parse(urdf).getroot()
    for name in ("R_FOOT", "L_FOOT"):
        feet = [prim for prim in stage.Traverse() if prim.GetName() == name]
        if len(feet) != 1:
            raise ValueError(f"Expected one {name} body, found {len(feet)}")
        foot = feet[0]
        # The original convex mesh is inside an instance. De-instance its
        # collision branch before authoring a stronger collisionEnabled=false.
        mesh_colliders = [prim for prim in Usd.PrimRange(foot, Usd.TraverseInstanceProxies())
                          if prim.HasAPI(UsdPhysics.CollisionAPI)]
        if len(mesh_colliders) != 1 or not mesh_colliders[0].IsA(UsdGeom.Mesh):
            raise ValueError(f"Expected one original convex foot mesh on {name}")
        old_path = mesh_colliders[0].GetPath()
        ancestor = mesh_colliders[0].GetParent()
        while ancestor != foot:
            if ancestor.IsInstance() and not ancestor.IsInstanceProxy():
                ancestor.SetInstanceable(False)
            ancestor = ancestor.GetParent()
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(old_path)).CreateCollisionEnabledAttr(False)
        collisions = robot_xml.findall(f"link[@name='{name}']/collision")
        if len(collisions) != 7:
            raise ValueError(f"Expected seven URDF sole capsules on {name}")
        for index, collision in enumerate(collisions, 1):
            shape = collision.find("geometry/capsule")
            if shape is None:
                raise ValueError(f"Non-capsule foot collision on {name}")
            origin = collision.find("origin")
            xyz = [float(value) for value in origin.get("xyz", "0 0 0").split()]
            rpy = [float(value) for value in origin.get("rpy", "0 0 0").split()]
            capsule = UsdGeom.Capsule.Define(stage, foot.GetPath().AppendChild(f"sole_capsule_{index}"))
            capsule.CreateRadiusAttr(float(shape.get("radius")))
            capsule.CreateHeightAttr(float(shape.get("height")))
            capsule.CreateAxisAttr(UsdGeom.Tokens.z)
            capsule.AddTranslateOp().Set(Gf.Vec3d(*xyz))
            capsule.AddOrientOp().Set(rpy_quaternion(rpy))
            capsule.CreatePurposeAttr(UsdGeom.Tokens.guide)
            capsule.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.8, 0.5)])
            UsdPhysics.CollisionAPI.Apply(capsule.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdShade.MaterialBindingAPI.Apply(capsule.GetPrim()).Bind(
                material, materialPurpose="physics")
            # MJLab G1 contact defaults. Foot priority ensures sampled friction
            # wins over the plane's material during MuJoCo contact mixing.
            for key, kind, value in (
                ("priority", Sdf.ValueTypeNames.Int, 1),
                ("condim", Sdf.ValueTypeNames.Int, 3),
                ("margin", Sdf.ValueTypeNames.Float, 0.0),
                ("gap", Sdf.ValueTypeNames.Float, 0.0),
                ("solref", Sdf.ValueTypeNames.Float2, Gf.Vec2f(0.02, 1.0)),
            ):
                capsule.GetPrim().CreateAttribute(f"mjc:{key}", kind).Set(value)
    stage.GetRootLayer().customLayerData = {
        "generator": "scripts/build_grasshopper_capsule_feet.py",
        "collisionSource": "grasshopper_lower_body.urdf",
        "layoutReference": "MJLab Unitree G1: seven longitudinal sole capsules per foot",
    }
    temporary = output_usd.with_name(output_usd.stem + ".building.usda")
    stage.GetRootLayer().Export(str(temporary))
    # Anchor the relative reference in a real layer, after exporting the
    # anonymous stage. This avoids unresolved relative paths during composition.
    layer = Sdf.Layer.FindOrOpen(str(temporary))
    layer.GetPrimAtPath(root.GetPath()).referenceList.prependedItems = [
        Sdf.Reference(os.path.relpath(base_usd, output_usd.parent))
    ]
    layer.Save()
    verification = Usd.Stage.Open(str(temporary))
    for name in ("R_FOOT", "L_FOOT"):
        feet = [prim for prim in verification.Traverse() if prim.GetName() == name]
        if len(feet) != 1 or sum(child.IsA(UsdGeom.Capsule) for child in feet[0].GetChildren()) != 7:
            raise ValueError(f"Failed to compose seven capsules on {name}")
    temporary.replace(output_usd)
    return output_usd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-usd", type=Path, default=BASE_USD)
    parser.add_argument("--output-usd", type=Path, default=CAPSULE_USD)
    args = parser.parse_args()
    print(build_capsule_feet(args.base_usd, args.output_usd))
