"""Check the composed foot contacts and preservation of rigid-body dynamics."""

import math
import unittest
from xml.etree import ElementTree

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from nest.assets.robots.grasshopper import MESH_FEET_USD_PATH, URDF_PATH, USD_PATH


class TestGrasshopperFootCollisions(unittest.TestCase):
    def test_urdf_has_mirrored_capsules_with_the_original_sole_bounds(self):
        robot = ElementTree.parse(URDF_PATH).getroot()
        feet = []
        for name in ("R_FOOT", "L_FOOT"):
            collisions = robot.findall(f"link[@name='{name}']/collision")
            self.assertEqual(len(collisions), 7)
            centers, lengths = [], []
            for collision in collisions:
                geometry = collision.find("geometry")
                self.assertEqual([child.tag for child in geometry], ["capsule"])
                self.assertAlmostEqual(float(geometry[0].get("radius")), 0.01)
                origin = collision.find("origin")
                np.testing.assert_allclose(list(map(float, origin.get("rpy").split())), [0, math.pi / 2, 0])
                centers.append(list(map(float, origin.get("xyz").split())))
                lengths.append(float(geometry[0].get("height")))
            centers, lengths = np.asarray(centers), np.asarray(lengths)
            np.testing.assert_allclose(centers[:, 2] - 0.01, -0.0773)
            self.assertAlmostEqual(centers[:, 1].min() - 0.01, -0.043)
            self.assertAlmostEqual(centers[:, 1].max() + 0.01, 0.043)
            self.assertAlmostEqual(np.max(centers[:, 0] + lengths / 2 + 0.01)
                                   - np.min(centers[:, 0] - lengths / 2 - 0.01), 0.258)
            feet.append((centers, lengths))
        np.testing.assert_allclose(feet[0][0] * [-1, 1, 1], feet[1][0])
        np.testing.assert_allclose(feet[0][1], feet[1][1])

    def test_both_usd_variants_use_native_capsules_and_preserve_other_physics(self):
        base = Usd.Stage.Open(str(MESH_FEET_USD_PATH))
        updated = Usd.Stage.Open(str(USD_PATH))
        for variant in ("mujoco", "physx"):
            for stage in (base, updated):
                stage.GetDefaultPrim().GetVariantSet("Physics").SetVariantSelection(variant)
            with self.subTest(variant=variant):
                foot_paths = [str(prim.GetPath()) for prim in base.Traverse()
                              if prim.GetName() in ("R_FOOT", "L_FOOT")]
                for name in ("R_FOOT", "L_FOOT"):
                    foot = next(prim for prim in updated.Traverse() if prim.GetName() == name)
                    shapes = [prim for prim in Usd.PrimRange(foot, Usd.TraverseInstanceProxies())
                              if prim.HasAPI(UsdPhysics.CollisionAPI)]
                    enabled = [prim for prim in shapes if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()]
                    self.assertEqual(len(enabled), 7)
                    self.assertTrue(all(prim.IsA(UsdGeom.Capsule) for prim in enabled))
                    self.assertEqual(sum(prim.IsA(UsdGeom.Mesh) for prim in shapes), 1)
                    for prim in enabled:
                        capsule = UsdGeom.Capsule(prim)
                        self.assertAlmostEqual(capsule.GetRadiusAttr().Get(), 0.01)
                        matrix = UsdGeom.Xformable(prim).GetLocalTransformation()
                        axis = matrix.TransformDir(Gf.Vec3d(0, 0, 1))
                        np.testing.assert_allclose(np.abs(axis), [1, 0, 0], atol=1e-6)
                # Compare original properties in the composed stages, including
                # every mass/inertia, joint, visual mesh and non-foot collider.
                for prim in Usd.PrimRange(base.GetDefaultPrim(), Usd.TraverseInstanceProxies()):
                    other = updated.GetPrimAtPath(prim.GetPath())
                    self.assertTrue(other.IsValid(), str(prim.GetPath()))
                    self.assertEqual(prim.GetTypeName(), other.GetTypeName())
                    foot_mesh = prim.IsA(UsdGeom.Mesh) and any(
                        str(prim.GetPath()).startswith(path + "/") for path in foot_paths
                    ) and prim.HasAPI(UsdPhysics.CollisionAPI)
                    for attr in prim.GetAttributes():
                        if foot_mesh and attr.GetName() == "physics:collisionEnabled":
                            continue
                        self.assertEqual(attr.Get(), other.GetAttribute(attr.GetName()).Get(),
                                         f"{variant}: {attr.GetPath()}")
                    for rel in prim.GetRelationships():
                        self.assertEqual(rel.GetTargets(), other.GetRelationship(rel.GetName()).GetTargets())


if __name__ == "__main__":
    unittest.main()
