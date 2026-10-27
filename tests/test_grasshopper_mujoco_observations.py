"""Protect COM/link velocity conventions in the standalone walking controller."""

import importlib.util
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("walk_native", SCRIPTS / "deploy_grasshopper_walk_mujoco.py")
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


class TestNativeWalkingObservations(unittest.TestCase):
    def test_rotating_offset_com_and_observation_order(self):
        children = "".join(f'<body name="link{i}"><joint name="j{i}"/><geom type="sphere" size=".01" mass=".01" contype="0" conaffinity="0"/></body>' for i in range(14))
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body name="PELVIS"><freejoint/>'
            '<inertial pos=".1 0 0" mass="2" diaginertia=".02 .03 .04"/>' + children + '</body></worldbody></mujoco>')
        data = mujoco.MjData(model)
        data.qpos[3:7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        data.qvel[:6] = [1, 2, 3, 0, 0, 2]
        mujoco.mj_forward(model, data)
        action = np.linspace(-.1, .1, 14)
        command = np.array([.5, 0, .25])
        obs, gravity, world_link, body_link, yaw = native.observe(
            model, data, 1, np.arange(7, 21), np.arange(6, 20), np.zeros(14), action, command)
        # World COM velocity includes omega x R*r = (-.2, 0, 0).
        np.testing.assert_allclose(obs[:3], [2, -.8, 3], atol=1e-6)
        np.testing.assert_allclose(obs[3:6], [0, 0, 2], atol=1e-6)
        np.testing.assert_allclose(world_link, [1, 2, 3], atol=1e-6)
        np.testing.assert_allclose(body_link, [2, -1, 3], atol=1e-6)
        np.testing.assert_allclose(gravity, [0, 0, -1], atol=1e-6)
        np.testing.assert_allclose(obs[37:51], action, atol=1e-6)
        np.testing.assert_allclose(obs[51:54], command, atol=1e-6)
        self.assertEqual(obs.shape, (54,))
        self.assertAlmostEqual(yaw, 2.)


if __name__ == "__main__":
    unittest.main()
