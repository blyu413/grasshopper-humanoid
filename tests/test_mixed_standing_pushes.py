"""Check sustained lifetimes, true rest periods, superposition, and clean episodes."""

import copy
import unittest
from types import SimpleNamespace

import torch

from nest.tasks.stand.mdp import MixedTorsoForces


class WrenchBuffer:
    def __init__(self, count):
        self.force = torch.zeros(count, 1, 3)

    def reset(self, env_ids=None):
        self.force[env_ids if env_ids is not None else slice(None)] = 0

    def set_forces_and_torques_index(self, forces, torques, body_ids, is_global):
        assert body_ids == [0] and is_global and torch.count_nonzero(torques) == 0
        self.force.copy_(forces)


def fixture(count=4, clean_probability=0.0):
    buffer = WrenchBuffer(count)
    env = SimpleNamespace(num_envs=count, device="cpu", step_dt=0.1,
                          common_step_counter=0, extras={},
                          scene={"robot": SimpleNamespace(permanent_wrench_composer=buffer)})
    stage = {"step": 0,
             "sustained": {"enabled": True, "force_range": (20.0, 20.0), "duration_s": (5.0, 5.0), "idle_s": (0.3, 0.3)},
             "pulse": {"enabled": False, "force_range": (50.0, 50.0), "duration_s": (0.2, 0.2), "idle_s": (0.3, 0.3)}}
    params = dict(asset_cfg=SimpleNamespace(name="robot", body_ids=[0]), stages=[stage],
                  clean_episode_probability=clean_probability, max_combined_force=220.0,
                  settle_s=1.0, start_step=0)
    event = MixedTorsoForces(SimpleNamespace(params=params), env)
    event.idle[0] = 0
    return env, event, params, buffer


class TestMixedStandingPushes(unittest.TestCase):
    def test_sustained_force_holds_5_seconds_without_early_resampling(self):
        env, event, params, buffer = fixture()
        event(env, None, **params)
        initial = buffer.force.clone()
        for _ in range(49):
            event(env, None, **params)
            torch.testing.assert_close(buffer.force, initial)
        for _ in range(3):
            event(env, None, **params)
            self.assertEqual(torch.count_nonzero(buffer.force), 0)
        event(env, None, **params)
        self.assertTrue((buffer.force.norm(dim=-1) > 0).all())

    def test_channels_superimpose_and_pulse_end_preserves_sustained_force(self):
        env, event, params, buffer = fixture()
        params["stages"][0]["pulse"]["enabled"] = True
        event.idle[:] = 0
        event(env, None, **params)
        torch.testing.assert_close(buffer.force, event.force.sum(dim=0))
        sustained = event.force[0].clone()
        event(env, None, **params)
        event(env, None, **params)
        torch.testing.assert_close(buffer.force, sustained)
        self.assertTrue((event.remaining[0] > 0).all())

    def test_combined_force_cap_and_zero_vertical_force(self):
        env, event, params, buffer = fixture()
        event.remaining[:] = 10
        event.force[:, :, 0, 0] = 200
        event(env, None, **params)
        torch.testing.assert_close(buffer.force[:, 0, 0], torch.full((4,), 220.0))
        self.assertEqual(torch.count_nonzero(buffer.force[..., 2]), 0)

    def test_clean_episode_remains_force_free_across_cycles(self):
        env, event, params, buffer = fixture(clean_probability=1.0)
        params["stages"][0]["pulse"]["enabled"] = True
        event.idle[:] = 0
        for _ in range(150):
            event(env, None, **params)
            self.assertEqual(torch.count_nonzero(buffer.force), 0)

    def test_reset_clears_both_channels_only_for_selected_environment(self):
        env, event, params, buffer = fixture()
        params["stages"][0]["pulse"]["enabled"] = True
        event.idle[:] = 0
        event(env, None, **params)
        other_forces = buffer.force[1:].clone()
        other_remaining = event.remaining[:, 1:].clone()
        event.reset([0])
        self.assertEqual(torch.count_nonzero(buffer.force[0]), 0)
        self.assertEqual(torch.count_nonzero(event.remaining[:, 0]), 0)
        torch.testing.assert_close(buffer.force[1:], other_forces)
        torch.testing.assert_close(event.remaining[:, 1:], other_remaining)

    def test_curriculum_transition_does_not_interrupt_active_force(self):
        env, event, params, buffer = fixture()
        next_stage = copy.deepcopy(params["stages"][0])
        next_stage["step"] = 2
        next_stage["sustained"]["force_range"] = (100.0, 100.0)
        next_stage["pulse"]["enabled"] = True
        params["stages"].append(next_stage)
        event(env, None, **params)
        original = event.force[0].clone()
        event.idle[1] = 10
        env.common_step_counter = 2
        event(env, None, **params)
        torch.testing.assert_close(buffer.force, original)
        self.assertEqual(env.extras["log"]["Push/stage"], 1)

    def test_initial_settle_window_is_retained_on_reset(self):
        env, event, params, buffer = fixture()
        event.reset()
        for _ in range(12):
            event(env, None, **params)
            self.assertEqual(torch.count_nonzero(buffer.force), 0)
        event(env, None, **params)
        self.assertTrue((buffer.force.norm(dim=-1) > 0).all())


if __name__ == "__main__":
    unittest.main()
