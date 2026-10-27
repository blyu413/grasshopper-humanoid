"""MJLab startup randomization, adapted to Isaac Lab's backend bindings."""

import torch
import warp as wp

from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, SceneEntityCfg


class BodyDensity(ManagerTermBase):
    """Scale each physical body's mass and inertia together; leave COM and shape fixed."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self._default_mass = None

    def __call__(self, env, env_ids, scale_range: tuple[float, float], asset_cfg: SceneEntityCfg):
        if not 0 < scale_range[0] <= scale_range[1]:
            raise ValueError("Body density scales must be positive and ordered")
        if self._default_mass is None:
            self._default_mass = self.asset.data.body_mass.torch.clone()
            self._default_inertia = self.asset.data.body_inertia.torch.clone()
            candidates = torch.arange(self.asset.num_bodies, device=env.device)[asset_cfg.body_ids]
            self._body_ids = candidates[self._default_mass[0, candidates] > 0]
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
        bodies = self._body_ids
        scales = torch.empty((len(ids), len(bodies)), device=env.device).uniform_(*scale_range)
        mass = self._default_mass[ids[:, None], bodies] * scales
        inertia = self._default_inertia[ids[:, None], bodies] * scales.unsqueeze(-1)
        self.asset.set_masses_index(masses=mass, body_ids=bodies, env_ids=ids)
        self.asset.set_inertias_index(inertias=inertia, body_ids=bodies, env_ids=ids)


def joint_friction(env, env_ids, friction_range: tuple[float, float], asset_cfg: SceneEntityCfg):
    """Absolute dry joint friction [N m], as MJLab's dof_frictionloss randomization."""
    if not 0 <= friction_range[0] <= friction_range[1]:
        raise ValueError("Joint dry friction must be nonnegative and ordered")
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    robot = env.scene[asset_cfg.name]
    values = torch.empty((len(ids), len(asset_cfg.joint_ids)), device=env.device).uniform_(*friction_range)
    robot.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=values, joint_ids=asset_cfg.joint_ids, env_ids=ids)


class SharedFootFriction(mdp.randomize_rigid_body_material):
    """One sampled friction per robot, shared by both feet and all sole shapes."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._foot_shapes = None

    def _resolve_foot_shapes(self):
        import mujoco
        from isaaclab_newton.physics import NewtonManager
        solver = NewtonManager._solver
        model = solver.mj_model
        # A body's shapes are not contiguous in Newton's global shape
        # array after USD import. Resolve the actual solver mapping rather
        # than summing per-body shape counts into an assumed offset.
        geoms = [
            i for i in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])) or "").endswith(("R_FOOT", "L_FOOT"))
            and (model.geom_contype[i] or model.geom_conaffinity[i])
        ]
        if len(geoms) != 14:
            raise ValueError(f"Expected fourteen Grasshopper foot colliders, got {len(geoms)}")
        self._foot_shapes = wp.to_torch(solver.mjc_geom_to_newton_shape)[:, geoms].long()
        self._mu = wp.to_torch(NewtonManager.get_model().shape_material_mu)
        self._restitution = wp.to_torch(NewtonManager.get_model().shape_material_restitution)

    def __call__(self, env, env_ids, static_friction_range, dynamic_friction_range,
                 restitution_range, num_buckets, asset_cfg, make_consistent=False):
        impl = self._impl
        if hasattr(impl, "_friction_binding"):
            # Manager terms resolve on PHYSICS_READY before the solver exists.
            # Startup events run later, so defer the mapping until this call.
            if self._foot_shapes is None:
                self._resolve_foot_shapes()
            ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids.to(env.device)
            shapes = self._foot_shapes[ids]
            samples = torch.empty((len(ids), 1), device=env.device).uniform_(*static_friction_range)
            self._mu[shapes] = samples
            self._restitution[shapes] = restitution_range[0]
            impl._newton_manager.add_model_change(impl._notify_shape_properties)
        elif hasattr(impl, "num_shapes_per_body"):
            # PhysX has a finite material table: choose one bucket per robot.
            ids = torch.arange(env.num_envs) if env_ids is None else env_ids.cpu().long()
            materials = wp.to_torch(self.asset.root_view.get_material_properties())
            samples = impl.material_buckets[torch.randint(num_buckets, (len(ids),))]
            for body in impl._backend_body_ids:
                first = sum(impl.num_shapes_per_body[:body])
                count = impl.num_shapes_per_body[body]
                materials[ids, first:first + count] = samples[:, None]
            self.asset.root_view.set_material_properties(
                wp.from_torch(materials), wp.from_torch(ids.to(torch.int32)))
        else:
            raise NotImplementedError("Shared foot friction supports Newton and Isaac Sim PhysX")


def encoder_bias(env, env_ids, bias_range: tuple[float, float], asset_cfg: SceneEntityCfg):
    """MJLab biases position targets, while its default joint observations stay unbiased."""
    if not hasattr(env, "_velocity_encoder_bias"):
        env._velocity_encoder_bias = torch.zeros(
            (env.num_envs, len(asset_cfg.joint_ids)), device=env.device)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    env._velocity_encoder_bias[ids] = torch.empty(
        (len(ids), len(asset_cfg.joint_ids)), device=env.device).uniform_(*bias_range)
