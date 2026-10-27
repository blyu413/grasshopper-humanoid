"""Guard the speed cap, standing transfer, and walking curriculum boundaries."""

from types import SimpleNamespace
import re
import unittest

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from nest.tasks.stand.config.grasshopper.env_cfg import GrasshopperStandEnvCfg
from nest.tasks.velocity.config.grasshopper.agents.rsl_rl_ppo_cfg import GrasshopperWalkPPORunnerCfg
from nest.tasks.velocity.config.grasshopper.env_cfg import GrasshopperWalkEnvCfg, WALK_ACTION_SCALE
from nest.assets.robots.grasshopper import JOINT_EFFORT_LIMITS
from nest.tasks.velocity.mdp.curriculums import command_velocity_stages
from nest.tasks.velocity.warm_start import initialize_from_standing


def make_actor(size):
    return MLPModel(
        TensorDict({"policy": torch.zeros(8, size)}, batch_size=[8]),
        {"actor": ["policy"]}, "actor", 14, hidden_dims=[256, 128, 128], activation="elu",
        distribution_cfg={"class_name": "rsl_rl.modules:GaussianDistribution", "init_std": 0.3},
    )


class TestGrasshopperWalk(unittest.TestCase):
    def test_sensor_pattern_matches_the_actual_nested_usd_feet(self):
        from pxr import Usd, UsdPhysics

        cfg = GrasshopperWalkEnvCfg()
        stage = Usd.Stage.Open(cfg.scene.robot.spawn.usd_path)
        root = stage.GetDefaultPrim()
        root.GetVariantSet("Physics").SetVariantSelection("mujoco")
        pattern = cfg.scene.feet_contact.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        matches = []
        for prim in stage.Traverse():
            path = str(prim.GetPath()).replace(str(root.GetPath()), "/World/envs/env_0/Robot", 1)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and re.fullmatch(pattern, path):
                matches.append(prim.GetName())
        self.assertEqual(sorted(matches), ["L_FOOT", "R_FOOT"])

    def test_transfer_preserves_mean_for_every_command_and_is_trainable(self):
        standing, walking = make_actor(52), make_actor(55)
        obs = torch.randn(8, 52)
        initialize_from_standing(walking, standing.state_dict())
        walk_obs = TensorDict({"policy": torch.cat((obs, torch.randn(8, 3)), dim=-1)}, batch_size=[8])
        with torch.no_grad():
            torch.testing.assert_close(standing(TensorDict({"policy": obs}, batch_size=[8])), walking(walk_obs))
        walking(walk_obs).square().mean().backward()
        self.assertGreater(walking.mlp[0].weight.grad[:, 52:].abs().sum().item(), 0.0)
        with self.assertRaises(ValueError):
            initialize_from_standing(walking, walking.state_dict())

    def test_curriculum_cap_survives_all_boundaries_and_late_resume(self):
        cfg = GrasshopperWalkEnvCfg()
        command = SimpleNamespace(cfg=cfg.commands.base_velocity)
        env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command), common_step_counter=0)
        stages = cfg.curriculum.command_vel.params["velocity_stages"]
        for step in (0, 11999, 12000, 71999, 72000, 119999, 120000, 10**9):
            env.common_step_counter = step
            command_velocity_stages(env, [], "base_velocity", stages)
            self.assertEqual(command.cfg.ranges.lin_vel_x, (0.0, 1.0))
            self.assertEqual(command.cfg.ranges.lin_vel_y, (0.0, 0.0))
            yaw_limit = 0.5 if step < 120000 else 0.7
            self.assertEqual(command.cfg.ranges.ang_vel_z, (-yaw_limit, yaw_limit))
        self.assertEqual(GrasshopperWalkPPORunnerCfg().num_steps_per_env, 24)

    def test_standing_prefix_actions_and_native_task_remain_compatible(self):
        stand, walk = GrasshopperStandEnvCfg(), GrasshopperWalkEnvCfg()
        walk.use_legacy_policy_observations()
        stand_terms = [name for name, value in vars(stand.observations.policy).items() if hasattr(value, "func")]
        walk_terms = [name for name, value in vars(walk.observations.policy).items() if hasattr(value, "func")]
        self.assertEqual(walk_terms, stand_terms + ["velocity_commands"])
        self.assertEqual(walk.actions.joint_pos.scale, stand.actions.joint_pos.scale)
        self.assertEqual(walk.actions.joint_pos.joint_names, stand.actions.joint_pos.joint_names)
        self.assertEqual(walk.scene.robot.actuators, stand.scene.robot.actuators)
        self.assertFalse(stand.scene.robot.spawn.activate_contact_sensors)
        self.assertTrue(walk.scene.robot.spawn.activate_contact_sensors)
        self.assertFalse(hasattr(stand.rewards, "track_linear_velocity"))
        self.assertEqual(walk.scene.ground.init_state.pos[2], 0.0)

    def test_mjlab_actor_critic_noise_physics_and_exploration(self):
        cfg, agent = GrasshopperWalkEnvCfg(), GrasshopperWalkPPORunnerCfg()
        policy = cfg.observations.policy
        self.assertFalse(hasattr(policy, "base_height"))
        self.assertTrue(policy.enable_corruption)
        self.assertFalse(cfg.observations.critic.enable_corruption)
        self.assertEqual(policy.base_lin_vel.noise.n_max, 0.5)
        self.assertEqual(policy.joint_vel.noise.n_max, 1.5)
        self.assertAlmostEqual(cfg.sim.dt * cfg.decimation, 0.02)
        self.assertEqual(cfg.sim.physics.newton_mjwarp.num_substeps, 1)
        self.assertAlmostEqual(cfg.terminations.bad_orientation.params["limit_angle"], 70 * torch.pi / 180)
        self.assertIsNone(cfg.terminations.root_height)
        self.assertEqual(agent.actor.hidden_dims, [512, 256, 128])
        self.assertTrue(agent.actor.obs_normalization)
        self.assertTrue(agent.critic.obs_normalization)
        self.assertEqual(agent.obs_groups["critic"], ["critic"])
        self.assertEqual(agent.actor.distribution_cfg.init_std, 1.0)
        self.assertEqual(agent.algorithm.entropy_coef, 0.01)
        self.assertEqual(agent.algorithm.learning_rate, 1.0e-3)
        self.assertEqual(cfg.commands.base_velocity.resampling_time_range, (3.0, 8.0))
        for name, scale in WALK_ACTION_SCALE.items():
            kp = cfg.scene.robot.actuators["driven_joints"].stiffness[name]
            self.assertAlmostEqual(scale * kp, 0.25 * JOINT_EFFORT_LIMITS[name])
        self.assertIsNotNone(cfg.events.push_robot)
        cfg.play_mode()
        self.assertFalse(cfg.observations.policy.enable_corruption)
        self.assertIsNone(cfg.events.push_robot)


if __name__ == "__main__":
    unittest.main()
