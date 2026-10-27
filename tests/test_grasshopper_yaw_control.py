"""Check yaw command coverage, curriculum migration and stable policy transfer."""

import math
from types import SimpleNamespace
import unittest

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from nest.tasks.velocity.config.grasshopper.env_cfg import COMMAND_DESIGN, GrasshopperWalkEnvCfg
from nest.tasks.velocity.mdp.commands import MJLabVelocityCommand
from nest.tasks.velocity.mdp.curriculums import command_velocity_stages, velocity_curriculum_start_step
from nest.tasks.velocity.mdp.rewards import track_angular_velocity
from nest.tasks.velocity.warm_start import initialize_yaw_command


def make_command(num_envs):
    cfg = GrasshopperWalkEnvCfg().commands.base_velocity
    env = SimpleNamespace(
        num_envs=num_envs, device="cpu",
        sim=SimpleNamespace(vis_marker_registry=SimpleNamespace(clear_debug_vis_callback=lambda _: None)),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(
            heading_w=SimpleNamespace(torch=torch.zeros(num_envs))))},
    )
    return MJLabVelocityCommand(cfg, env)


class TestGrasshopperYawControl(unittest.TestCase):
    def test_sampling_covers_both_turns_and_preserves_straight_and_standing(self):
        torch.manual_seed(42)
        term = make_command(4096)
        term._resample_command(torch.arange(4096))
        command = term.command
        for mask, expected in ((term.is_forward_env, .2), (term.is_heading_env, .3),
                               (term.is_standing_env, .1)):
            self.assertAlmostEqual(mask.float().mean().item(), expected, delta=.03)
        torch.testing.assert_close(command[term.is_standing_env], torch.zeros_like(command[term.is_standing_env]))
        forward = term.is_forward_env & ~term.is_standing_env
        self.assertTrue((command[forward, 0] >= .3).all())
        self.assertTrue((command[forward, 1:] == 0).all())
        direct = ~term.is_heading_env & ~term.is_standing_env & ~term.is_forward_env
        self.assertGreater((command[direct, 2] > .1).sum().item(), 300)
        self.assertGreater((command[direct, 2] < -.1).sum().item(), 300)
        self.assertLessEqual(command[:, 2].abs().max().item(), .5)
        untouched = command[1:].clone()
        term._resample_command(torch.tensor([0]))
        torch.testing.assert_close(command[1:], untouched)

    def test_heading_wrap_clamp_and_forward_precedence_on_every_tick(self):
        term = make_command(4)
        term.is_heading_env[:] = torch.tensor([True, True, True, False])
        term.is_forward_env[2] = True
        term.heading_target[:] = torch.tensor([-math.pi + .1, 2., 2., 0.])
        term.robot.data.heading_w.torch[0] = math.pi - .1
        term.vel_command_b[3, 2] = -.3
        for _ in range(3):
            term._update_command()
            torch.testing.assert_close(term.command[:, 2], torch.tensor([.1, .5, 0., -.3]))

    def test_curriculum_starts_on_transfer_and_persists_across_later_resumes(self):
        anchor = velocity_curriculum_start_step({}, 5000, 24, COMMAND_DESIGN)
        self.assertEqual(anchor, 120000)
        saved = {"command_design": COMMAND_DESIGN, "velocity_curriculum_start_step": anchor}
        self.assertEqual(velocity_curriculum_start_step(saved, 5500, 24, COMMAND_DESIGN), anchor)
        self.assertEqual(velocity_curriculum_start_step(None, 0, 24, COMMAND_DESIGN), 0)
        cfg = GrasshopperWalkEnvCfg()
        env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(
            cfg=cfg.commands.base_velocity)))
        stages = cfg.curriculum.command_vel.params["velocity_stages"]
        for step, limit in ((120000, .5), (239999, .5), (240000, .7)):
            env.common_step_counter = step
            command_velocity_stages(env, [], "base_velocity", stages, start_step=anchor)
            self.assertEqual(cfg.commands.base_velocity.ranges.ang_vel_z, (-limit, limit))

    def test_new_yaw_input_preserves_old_output_and_receives_gradients(self):
        for obs_size, column in ((54, 53), (55, 54), (66, 53)):
            with self.subTest(obs_size=obs_size):
                model = MLPModel(TensorDict({"policy": torch.zeros(8, obs_size)}, batch_size=[8]),
                                 {"actor": ["policy"]}, "actor", 14, hidden_dims=[32],
                                 activation="elu", obs_normalization=True)
                optimizer = torch.optim.Adam(model.parameters())
                obs = torch.randn(8, obs_size)
                obs[:, column] = 0.
                data = TensorDict({"policy": obs}, batch_size=[8])
                model(data).square().mean().backward()
                optimizer.step()
                optimizer.zero_grad()
                model.obs_normalizer._var[:, column] = 0.
                model.obs_normalizer._std[:, column] = 0.
                model.obs_normalizer.count.fill_(491520000)
                before = model(data).detach().clone()
                weights = model.mlp[0].weight.detach().clone()
                initialize_yaw_command(model, column, optimizer)
                torch.testing.assert_close(model(data), before)
                self.assertEqual(model.obs_normalizer.count.item(), 491520000)
                self.assertGreater(model.obs_normalizer._std[0, column].item(), .25)
                keep = torch.arange(obs_size) != column
                torch.testing.assert_close(model.mlp[0].weight[:, keep], weights[:, keep])
                for value in optimizer.state[model.mlp[0].weight].values():
                    if isinstance(value, torch.Tensor) and value.shape == model.mlp[0].weight.shape:
                        self.assertTrue((value[:, column] == 0).all())
                obs[:, column] = .5
                torch.testing.assert_close(model(data), before)
                model(data).square().mean().backward()
                self.assertGreater(model.mlp[0].weight.grad[:, column].abs().sum().item(), 0.)

    def test_yaw_tracking_distinguishes_matching_from_opposite_turns(self):
        command = torch.tensor([[0., 0., .5], [0., 0., -.5]])
        env = SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(
            root_link_ang_vel_b=SimpleNamespace(torch=torch.tensor([[0., 0., .5], [0., 0., .5]]))))},
            command_manager=SimpleNamespace(get_command=lambda _: command), extras={})
        value = track_angular_velocity(env, math.sqrt(.5), "base_velocity")
        torch.testing.assert_close(value, torch.tensor([1., math.exp(-2)]))
        self.assertEqual(env.extras["log"]["Metrics/velocity_error_yaw_rad_s"].item(), .5)

    def test_playback_keeps_explicit_zero_yaw_without_heading_overrides(self):
        cfg = GrasshopperWalkEnvCfg()
        cfg.play_mode()
        self.assertFalse(cfg.commands.base_velocity.heading_command)
        self.assertIsNone(cfg.commands.base_velocity.ranges.heading)
        self.assertEqual(cfg.commands.base_velocity.rel_forward_envs, 0.)
        self.assertEqual(cfg.commands.base_velocity.ranges.ang_vel_z, (0., 0.))


if __name__ == "__main__":
    unittest.main()
