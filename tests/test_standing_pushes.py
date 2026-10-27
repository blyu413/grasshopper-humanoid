"""Verify pulse lifetime, selective reset, and the force curriculum."""

import unittest
from types import SimpleNamespace

import torch

from nest.tasks.stand.mdp import RandomTorsoPulses


class WrenchBuffer:
    def __init__(self, count):
        self.force = torch.zeros(count, 1, 3)

    def reset(self, env_ids=None):
        self.force[env_ids if env_ids is not None else slice(None)] = 0

    def set_forces_and_torques_index(self, forces, torques, body_ids, is_global):
        assert body_ids == [0] and is_global
        assert torch.count_nonzero(torques) == 0
        self.force.copy_(forces)


def fixture(count=16):
    buffer = WrenchBuffer(count)
    env = SimpleNamespace(num_envs=count, device="cpu", step_dt=1 / 60,
                          common_step_counter=0, extras={},
                          scene={"robot": SimpleNamespace(permanent_wrench_composer=buffer)})
    params = dict(asset_cfg=SimpleNamespace(name="robot", body_ids=[0]),
                  force_range=(20.0, 150.0), initial_max_force=60.0, ramp_steps=18_000,
                  duration_s=(0.2, 0.2), interval_s=(1.5, 3.0), push_probability=1.0)
    event = RandomTorsoPulses(SimpleNamespace(params=params), env)
    event.wait[:] = 0
    return env, event, params, buffer


class TestStandingPushes(unittest.TestCase):
    def test_force_ends_after_exact_pulse_duration(self):
        env, event, params, buffer = fixture()
        for _ in range(12):
            event(env, None, **params)
            self.assertTrue((buffer.force.norm(dim=-1) > 0).all())
        event(env, None, **params)
        self.assertEqual(torch.count_nonzero(buffer.force), 0)

    def test_reset_clears_only_selected_environment(self):
        env, event, params, buffer = fixture()
        event(env, None, **params)
        force_before = buffer.force[1:].clone()
        event.reset(torch.tensor([0], dtype=torch.int32))
        self.assertEqual(torch.count_nonzero(buffer.force[0]), 0)
        torch.testing.assert_close(force_before, buffer.force[1:])
        event(env, None, **params)
        self.assertEqual(torch.count_nonzero(buffer.force[0]), 0)
        self.assertTrue((buffer.force[1:].norm(dim=-1) > 0).all())

    def test_random_horizontal_forces_follow_curriculum(self):
        torch.manual_seed(42)
        env, event, params, buffer = fixture(1024)
        event(env, None, **params)
        norms = buffer.force.norm(dim=-1)
        self.assertTrue(((norms >= 20 - 1e-4) & (norms <= 60 + 1e-4)).all())
        env.common_step_counter = 18_000
        event.wait[:] = 0
        event(env, None, **params)
        norms = buffer.force.norm(dim=-1)
        self.assertTrue((norms <= 150 + 1e-4).all())
        self.assertGreater(float(norms.max()), 140)
        self.assertEqual(torch.count_nonzero(buffer.force[..., 2]), 0)
        self.assertTrue((buffer.force[..., :2] > 0).any())
        self.assertTrue((buffer.force[..., :2] < 0).any())


if __name__ == "__main__":
    unittest.main()
