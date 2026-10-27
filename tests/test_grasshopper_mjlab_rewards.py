"""Check sole kinematics, command gating, and landing state for walking rewards."""

import math
from types import SimpleNamespace
import unittest

import torch

from isaaclab.managers import RewardTermCfg, SceneEntityCfg

from nest.assets.robots.grasshopper import POLICY_JOINT_NAMES
from nest.tasks.velocity.config.grasshopper.env_cfg import FOOT_SOLE_OFFSETS, GrasshopperWalkEnvCfg
from nest.tasks.velocity.mdp import rewards
from nest.tasks.velocity.mdp.actions import BiasedJointPositionAction


def proxy(value):
    return SimpleNamespace(torch=value)


class TestGrasshopperMJLabRewards(unittest.TestCase):
    def setUp(self):
        self.cfg = GrasshopperWalkEnvCfg()
        self.command = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.positions = torch.zeros(2, 2, 3)
        self.positions[:, :, 2] = 0.0773
        self.rotations = torch.zeros(2, 2, 4)
        self.rotations[:, :, 3] = 1.0  # Isaac Lab uses XYZW quaternions.
        self.velocities = torch.zeros(2, 2, 3)
        self.angular_velocities = torch.zeros(2, 2, 3)
        self.forces = torch.zeros(2, 2, 3)
        self.friction = torch.zeros_like(self.forces)
        self.landings = torch.zeros(2, 2)
        data = SimpleNamespace(
            body_link_pos_w=proxy(self.positions), body_link_quat_w=proxy(self.rotations),
            body_link_lin_vel_w=proxy(self.velocities), body_link_ang_vel_w=proxy(self.angular_velocities),
        )
        sensor = SimpleNamespace(
            data=SimpleNamespace(net_normal_forces_w=proxy(self.forces), net_friction_forces_w=proxy(self.friction)),
            compute_first_contact=lambda dt: proxy(self.landings),
        )
        self.env = SimpleNamespace(
            scene={"robot": SimpleNamespace(data=data), "feet_contact": sensor},
            command_manager=SimpleNamespace(get_command=lambda _: self.command),
            num_envs=2, device="cpu", step_dt=1 / 60,
        )
        self.asset = SceneEntityCfg("robot", body_ids=[0, 1])
        self.sensor = SceneEntityCfg("feet_contact", body_ids=[0, 1])

    def params(self, name):
        params = getattr(self.cfg.rewards, name).params.copy()
        if "asset_cfg" in params:
            params["asset_cfg"] = self.asset
        if "sensor_cfg" in params:
            params["sensor_cfg"] = self.sensor
        return params

    def test_configuration_replaces_standing_and_air_time_rewards(self):
        terms = vars(self.cfg.rewards)
        self.assertEqual(len(terms), 12)
        self.assertFalse({"alive", "height", "joint_deviation", "stand_still", "feet_air_time"} & terms.keys())
        self.assertTrue(self.cfg.scene.feet_contact.track_friction_forces)
        for name in ("foot_clearance", "foot_swing_height", "foot_slip"):
            self.assertEqual(terms[name].params["asset_cfg"].body_names, ["R_FOOT", "L_FOOT"])
        pose = terms["pose"].params
        self.assertEqual(pose["asset_cfg"].joint_names, list(POLICY_JOINT_NAMES))
        for regime in ("std_standing", "std_walking", "std_running"):
            std = rewards.VariablePosture._resolve_std(pose[regime], list(POLICY_JOINT_NAMES), "cpu")
            self.assertEqual(std.shape, (14,))
            self.assertTrue((std > 0.0).all())

    def test_sole_height_and_velocity_follow_foot_rotation(self):
        torch.testing.assert_close(rewards._foot_clearances(self.env, self.asset, FOOT_SOLE_OFFSETS), torch.zeros(2, 2))
        # A 90-degree yaw rotates the site's X offset into Y. With 2 rad/s yaw,
        # its tangential X speed must have opposite signs on the mirrored feet.
        self.rotations[:, :, 2] = math.sqrt(0.5)
        self.rotations[:, :, 3] = math.sqrt(0.5)
        self.angular_velocities[:, :, 2] = 2.0
        velocity = rewards._foot_site_velocity_xy(self.env, self.asset, FOOT_SOLE_OFFSETS)
        torch.testing.assert_close(velocity[:, :, 0], torch.tensor([[-0.1078, 0.1078]] * 2))
        torch.testing.assert_close(velocity[:, :, 1], torch.zeros(2, 2), atol=1e-7, rtol=0)
        # Clearance must also follow ankle pitch, rather than a fixed world offset.
        self.rotations.zero_()
        self.rotations[:, :, 1] = math.sqrt(0.5)
        self.rotations[:, :, 3] = math.sqrt(0.5)
        heights = rewards._foot_clearances(self.env, self.asset, FOOT_SOLE_OFFSETS)
        torch.testing.assert_close(heights, torch.tensor([[0.0234, 0.1312]] * 2))

    def test_clearance_uses_sole_speed_and_standing_gate(self):
        params = self.params("foot_clearance")
        self.velocities[:, :, 0] = 2.0
        torch.testing.assert_close(rewards.feet_clearance(self.env, **params), torch.tensor([0.4, 0.0]))
        self.positions[:, :, 2] += 0.1
        torch.testing.assert_close(rewards.feet_clearance(self.env, **params), torch.zeros(2))

    def test_g1_ray_caster_sites_remain_the_default(self):
        for side, height in (("left", 0.12), ("right", 0.2)):
            positions = torch.tensor([[0.0, 0.0, height]] * 2)
            self.env.scene[f"{side}_foot_height"] = SimpleNamespace(
                data=SimpleNamespace(pos_w=proxy(positions), ray_hits_w=proxy(torch.zeros(2, 7, 3))),
                cfg=SimpleNamespace(max_distance=1.0),
            )
        torch.testing.assert_close(
            rewards._foot_clearances(self.env, self.asset), torch.tensor([[0.12, 0.2]] * 2),
        )
        sites = rewards._foot_site_positions(self.env, self.asset)
        torch.testing.assert_close(sites[:, :, 2], torch.tensor([[0.12, 0.2]] * 2))

    def test_slip_requires_contact_and_supports_turning_commands(self):
        self.velocities[:, :, 0] = torch.tensor([2.0, 10.0])
        self.forces[:, 0, 2] = 100.0  # The fast swinging foot must not contribute.
        torch.testing.assert_close(rewards.feet_slip(self.env, **self.params("foot_slip")), torch.tensor([4.0, 0.0]))
        self.command[1, 2] = 0.2
        torch.testing.assert_close(rewards.feet_slip(self.env, **self.params("foot_slip")), torch.tensor([4.0, 4.0]))

    def test_swing_cost_is_paid_once_at_landing_and_resets_per_env(self):
        term = rewards.FeetSwingHeight(self.cfg.rewards.foot_swing_height, self.env)
        params = self.params("foot_swing_height")
        self.positions[:, :, 2] += torch.tensor([0.2, 0.1])
        torch.testing.assert_close(term(self.env, **params), torch.zeros(2))
        self.positions[:, :, 2] = 0.0773
        self.forces[:, :, 2] = 100.0
        self.landings.fill_(1.0)
        torch.testing.assert_close(term(self.env, **params), torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(term.peak_heights, torch.zeros(2, 2))
        self.landings.zero_()
        torch.testing.assert_close(term(self.env, **params), torch.zeros(2))
        term.peak_heights.fill_(0.2)
        term.reset([0])
        torch.testing.assert_close(term.peak_heights, torch.tensor([[0.0, 0.0], [0.2, 0.2]]))
        term.reset()
        torch.testing.assert_close(term.peak_heights, torch.zeros(2, 2))

    def test_soft_landing_includes_friction_only_on_first_contact(self):
        self.forces[:, :, 2] = 4.0
        self.friction[:, :, 0] = 3.0
        self.landings[:, 0] = 1.0
        params = self.params("soft_landing")
        torch.testing.assert_close(rewards.soft_landing(self.env, **params), torch.tensor([5.0, 0.0]))
        self.env.scene["feet_contact"].data.net_friction_forces_w = None
        torch.testing.assert_close(rewards.soft_landing(self.env, **params), torch.tensor([4.0, 0.0]))

    def test_explicit_reset_after_inference_mode_step(self):
        term = rewards.FeetSwingHeight(self.cfg.rewards.foot_swing_height, self.env)
        self.positions[:, :, 2] += 0.2
        with torch.inference_mode():
            term(self.env, **self.params("foot_swing_height"))
        self.assertFalse(term.peak_heights.is_inference())
        term.reset([0])
        torch.testing.assert_close(term.peak_heights, torch.tensor([[0.0, 0.0], [0.2, 0.2]]))
        term.reset()
        torch.testing.assert_close(term.peak_heights, torch.zeros(2, 2))

    def test_posture_relaxes_at_command_speed_boundaries(self):
        params = self.params("pose")
        params["asset_cfg"] = SceneEntityCfg("robot", joint_ids=list(range(14)))
        self.env.scene["robot"].joint_names = list(POLICY_JOINT_NAMES)
        self.env.scene["robot"].data.joint_pos = proxy(torch.full((2, 14), 0.05))
        self.env.scene["robot"].data.default_joint_pos = proxy(torch.zeros(2, 14))
        term = rewards.VariablePosture(RewardTermCfg(func=rewards.VariablePosture, weight=1.0, params=params), self.env)
        self.command[:, 0] = torch.tensor([0.049, 0.05])
        values = term(self.env, **params)
        self.assertAlmostEqual(values[0].item(), math.exp(-1), places=6)
        self.assertGreater(values[1].item(), values[0].item())
        self.command[:, 0] = torch.tensor([1.499, 1.5])
        values = term(self.env, **params)
        self.assertGreater(values[1].item(), values[0].item())

    def test_angular_momentum_rotates_link_inertia_into_world(self):
        # Rotate a non-isotropic body's local X inertia (2) onto world Y.
        # World omega=Y must use that inertia, giving |L|^2 = 4.
        data = self.env.scene["robot"].data
        data.body_mass = proxy(torch.ones(2, 2))
        data.body_com_pos_w = proxy(torch.zeros(2, 2, 3))
        data.body_com_lin_vel_w = proxy(torch.zeros(2, 2, 3))
        omega = torch.zeros(2, 2, 3)
        omega[:, 0, 1] = 1.0
        data.body_com_ang_vel_w = proxy(omega)
        data.body_inertia = proxy(torch.diag(torch.tensor([2., 3., 4.])).flatten().repeat(2, 2, 1))
        self.rotations[:, :, 2:] = math.sqrt(0.5)
        torch.testing.assert_close(rewards.angular_momentum_penalty(self.env), torch.full((2,), 4.0))

    def test_encoder_bias_changes_targets_without_changing_raw_observations(self):
        requested = torch.tensor([[0.2, -0.2]])
        bias = torch.tensor([[0.015, -0.01]])
        writes = []
        action = SimpleNamespace(
            processed_actions=requested,
            _env=SimpleNamespace(_velocity_encoder_bias=bias),
            _asset=SimpleNamespace(set_joint_position_target_index=lambda **kw: writes.append(kw)),
            _joint_ids=[2, 4],
        )
        BiasedJointPositionAction.apply_actions(action)
        torch.testing.assert_close(writes[0]["target"], torch.tensor([[0.185, -0.19]]))
        torch.testing.assert_close(requested, torch.tensor([[0.2, -0.2]]))
        self.assertEqual(writes[0]["joint_ids"], [2, 4])


if __name__ == "__main__":
    unittest.main()
