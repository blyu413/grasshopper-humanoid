"""Check physical density scaling, delayed targets and reset isolation."""

from types import SimpleNamespace
import unittest

import torch
from isaaclab.managers import EventTermCfg, SceneEntityCfg

from nest.tasks.velocity.config.grasshopper.env_cfg import GrasshopperWalkEnvCfg
from nest.tasks.velocity.mdp.actions import PhysicsStepDelay
from nest.tasks.velocity.mdp.events import BodyDensity, joint_friction


class TestGrasshopperDomainRandomization(unittest.TestCase):
    def test_delay_is_in_physics_ticks_and_shared_across_joints(self):
        delay = PhysicsStepDelay(torch.tensor([[2., 3.], [4., 5.]]), 0, 4, 100)
        delay.lags.copy_(torch.tensor([0, 2]))
        delay.phases.fill_(10)
        for tick in range(5):
            result = delay.compute(torch.full((2, 2), float(tick + 10)))
            torch.testing.assert_close(result[0], torch.full((2,), float(tick + 10)))
            expected = torch.tensor([4., 5.]) if tick < 2 else torch.full((2,), float(tick + 8))
            torch.testing.assert_close(result[1], expected)

    def test_partial_reset_clears_only_selected_history_and_survives_inference_mode(self):
        neutral = torch.tensor([[2., 3.], [4., 5.]])
        delay = PhysicsStepDelay(neutral, 1, 1, 100)
        with torch.inference_mode():
            delay.compute(torch.full((2, 2), 50.))
            delay.compute(torch.full((2, 2), 60.))
        delay.reset(neutral, [0])
        delay.phases.fill_(10)
        result = delay.compute(torch.full((2, 2), 70.))
        torch.testing.assert_close(result[0], neutral[0])
        torch.testing.assert_close(result[1], torch.full((2,), 60.))
        self.assertFalse(delay.values.is_inference())

    def test_density_preserves_inertia_per_unit_mass_and_does_not_accumulate(self):
        mass = torch.tensor([[0., 2., 20.]]).repeat(4, 1)
        inertia = torch.diag(torch.tensor([2., 3., 4.])).flatten().repeat(4, 3, 1)
        inertia[:, 0] = 0
        original_mass, original_inertia = mass.clone(), inertia.clone()
        writes = {}
        def set_mass(*, masses, body_ids, env_ids):
            mass[env_ids[:, None], body_ids] = masses
            writes["mass_ids"] = env_ids.clone()
        def set_inertia(*, inertias, body_ids, env_ids):
            inertia[env_ids[:, None], body_ids] = inertias
        robot = SimpleNamespace(
            data=SimpleNamespace(body_mass=SimpleNamespace(torch=mass),
                                 body_inertia=SimpleNamespace(torch=inertia)),
            num_bodies=3, set_masses_index=set_mass, set_inertias_index=set_inertia)
        env = SimpleNamespace(scene={"robot": robot}, device="cpu", num_envs=4)
        asset_cfg = SceneEntityCfg("robot")
        cfg = EventTermCfg(func=BodyDensity, mode="startup", params={"asset_cfg": asset_cfg})
        term = BodyDensity(cfg, env)
        torch.manual_seed(42)
        term(env, None, (0.9, 1.1), asset_cfg)
        first_mass, first_inertia = mass.clone(), inertia.clone()
        ratios = mass[:, 1:] / original_mass[:, 1:]
        self.assertTrue(((ratios >= .9) & (ratios <= 1.1)).all())
        torch.testing.assert_close(inertia[:, 1:], original_inertia[:, 1:] * ratios.unsqueeze(-1))
        torch.testing.assert_close(mass[:, 0], torch.zeros(4))
        torch.manual_seed(42)
        term(env, None, (0.9, 1.1), asset_cfg)
        torch.testing.assert_close(mass, first_mass)
        torch.testing.assert_close(inertia, first_inertia)
        term(env, torch.tensor([1, 3]), (1., 1.), asset_cfg)
        torch.testing.assert_close(mass[[1, 3]], original_mass[[1, 3]])
        torch.testing.assert_close(mass[[0, 2]], first_mass[[0, 2]])

    def test_dry_friction_does_not_write_viscous_or_pd_damping(self):
        writes = []
        robot = SimpleNamespace(write_joint_friction_coefficient_to_sim_index=lambda **kw: writes.append(kw))
        env = SimpleNamespace(scene={"robot": robot}, num_envs=3, device="cpu")
        cfg = SceneEntityCfg("robot", joint_ids=[2, 5])
        joint_friction(env, torch.tensor([0, 2]), (0., .5), cfg)
        values = writes[0]["joint_friction_coeff"]
        self.assertEqual(tuple(values.shape), (2, 2))
        self.assertTrue(((values >= 0) & (values <= .5)).all())
        self.assertEqual(writes[0]["joint_ids"], [2, 5])

    def test_play_disables_hardware_dr_and_keeps_checkpoint_layout(self):
        cfg = GrasshopperWalkEnvCfg()
        self.assertEqual(cfg.actions.joint_pos.delay_max_lag, 4)
        self.assertEqual(cfg.actions.joint_pos.delay_update_period, 50)
        self.assertEqual(cfg.events.body_density.params["scale_range"], (.9, 1.1))
        self.assertEqual(cfg.events.actuator_gains.params["operation"], "scale")
        rewards = cfg.rewards.to_dict()
        cfg.play_mode()
        for name in ("body_density", "actuator_gains", "joint_friction", "encoder_bias", "foot_friction", "push_robot"):
            self.assertIsNone(getattr(cfg.events, name))
        self.assertEqual(cfg.actions.joint_pos.delay_max_lag, 0)
        self.assertEqual(cfg.rewards.to_dict(), rewards)


if __name__ == "__main__":
    unittest.main()
