"""NEST's native velocity MDP expressed over the V3 predicted physical state.

Simulator labels are captured after physics/reward evaluation and BEFORE reset,
command resampling and interval pushes. No simulator gradients are required.
"""
import torch
from torch import nn

from .kinematics import Kinematics, find_named, rotate
from .model import Layout
from .reference.mdp import invariant_features


def tensor(value):
    return value.torch if hasattr(value, 'torch') else value


class Context(nn.Module):
    """Only startup-varying dynamics parameters; no privileged actor inputs."""
    def __init__(self, env, model, robot, action_ids):
        super().__init__()
        import warp as wp
        from isaaclab_newton.physics import NewtonManager
        d, dev = robot.data, env.device
        bids = [find_named(model, 'body', n) for n in robot.body_names]
        nominal = torch.tensor(model.body_mass[bids], device=dev, dtype=torch.float32)
        self.register_buffer('nominal_mass', nominal)
        ids = (nominal > 0).nonzero().flatten()
        self.register_buffer('density_ids', ids)
        density = tensor(d.body_mass)[:, ids] / nominal[ids]
        full_density = torch.ones_like(tensor(d.body_mass))
        full_density[:, ids] = density
        self.register_buffer('nominal_inertia', tensor(d.body_inertia)[0].reshape(-1, 3, 3).clone() / full_density[0, :, None, None])
        self.register_buffer('com', tensor(d.body_com_pos_b)[0].clone())
        self.fields = {}
        parts = []
        def add(name, x):
            start = sum(part.shape[-1] for part in parts)
            self.fields[name] = (start, start + x.shape[-1])
            parts.append(x)
        add('density', density)
        add('kp', tensor(d.joint_stiffness)[:, action_ids] / 500.)
        add('kd', tensor(d.joint_damping)[:, action_ids] / 15.)
        add('joint_friction', tensor(d.joint_friction_coeff)[:, action_ids] / .5)
        foot_bids = [find_named(model, 'body', n) for n in ('R_FOOT', 'L_FOOT')]
        gids = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in foot_bids and model.geom_contype[i] != 0]
        if not gids:
            raise ValueError('Missing foot collision geometry')
        solver = NewtonManager._solver
        if solver.mjw_model is not None:
            mu = wp.to_torch(solver.mjw_model.geom_friction)[:, gids[0], :1]
            mu = mu.expand(env.num_envs, -1)
        else:
            mu = torch.full((env.num_envs, 1), float(model.geom_friction[gids[0], 0]), device=dev)
        add('foot_friction', (mu - .75) / .45)
        bias = getattr(env, '_velocity_encoder_bias', torch.zeros((env.num_envs, len(action_ids)), device=dev))
        add('encoder_bias', -bias / .01)
        # Deliberately not a buffer: this per-environment table is recreated on
        # resume from the new environments' actual startup randomization.
        self.table = torch.cat(parts, -1).clone()
        self.width = self.table.shape[-1]

    def capture(self):
        return self.table

    def masses_inertias(self, c):
        lo, hi = self.fields['density']
        density = torch.ones((*c.shape[:-1], len(self.nominal_mass)), device=c.device, dtype=c.dtype)
        density[..., self.density_ids] = c[..., lo:hi]
        return self.nominal_mass * density, self.nominal_inertia * density[..., None, None]


class VelocityMDP(nn.Module):
    def __init__(self, env, model, layout, context, action_ids):
        super().__init__()
        robot = env.scene['robot']
        self.layout, self.context, self.dt = layout, context, env.step_dt
        self.kin = Kinematics(model, robot.joint_names, robot.body_names).to(env.device)
        self.torso = robot.body_names.index('TORSO')
        self.feet = [robot.body_names.index(n) for n in ('R_FOOT', 'L_FOOT')]
        self.register_buffer('action_ids', torch.tensor(action_ids, device=env.device))
        self.register_buffer('default_q', tensor(robot.data.default_joint_pos)[0].clone())
        self.register_buffer('default_qd', tensor(robot.data.default_joint_vel)[0].clone())
        self.register_buffer('limits', tensor(robot.data.soft_joint_pos_limits)[0, action_ids].clone())
        from nest.tasks.velocity.config.grasshopper.env_cfg import FOOT_SOLE_OFFSETS
        self.register_buffer('foot_offsets', torch.tensor(FOOT_SOLE_OFFSETS, device=env.device))
        self.register_buffer('gravity', torch.tensor([0., 0., -1.], device=env.device))
        posture = env.reward_manager.get_term_cfg('pose').func
        for regime in ('standing', 'walking', 'running'):
            self.register_buffer('std_' + regime, getattr(posture, '_std_' + regime).clone())
        self.names = tuple(env.reward_manager.active_terms)
        self.weights = {n: env.reward_manager.get_term_cfg(n).weight for n in self.names}
        self.params = {n: env.reward_manager.get_term_cfg(n).params for n in self.names}
        self.limit_angle = env.cfg.terminations.bad_orientation.params['limit_angle']
        self.noise = bool(env.cfg.observations.policy.enable_corruption)

    def obs(self, s, ap, c=None, g=None):
        L = self.layout
        _, R, v, w, q, qd = L.unpack(s)
        # NEST's base_lin_vel is root COM velocity expressed in root link axes;
        # the velocity tracking reward separately uses the link-origin velocity.
        off = rotate(R, self.context.com[self.kin.root_index])
        com_v = v + torch.cross(w, off, dim=-1)
        Rt = R.transpose(-1, -2)
        return torch.cat((rotate(Rt, com_v), rotate(Rt, w), rotate(Rt, self.gravity),
                          (q-self.default_q)[..., self.action_ids],
                          (qd-self.default_qd)[..., self.action_ids], ap), -1)

    def add_noise(self, obs):
        if not self.noise:
            return obs
        n = self.layout.actions
        amp = obs.new_tensor([.5]*3 + [.2]*3 + [.05]*3 + [.01]*n + [1.5]*n + [0.]*n)
        return obs + (2*torch.rand_like(obs)-1)*amp

    def advance_goal(self, s, g):
        R = self.layout.unpack(s)[1]
        yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
        error = g[..., 4] - yaw
        error = torch.atan2(error.sin(), error.cos())
        wz = (error*g[..., 7]).clamp(g[..., 5], g[..., 6])
        cmd = torch.cat((g[..., :2], torch.where(g[..., 3] > .5, wz, g[..., 2]).unsqueeze(-1)), -1)
        return torch.cat((cmd, g[..., 3:]), -1)

    def critic_input(self, s, c, g):
        R = self.layout.unpack(s)[1]
        error = g[..., 4] - torch.atan2(R[..., 1, 0], R[..., 0, 0])
        flag = g[..., 3:4]
        goal = torch.cat((g[..., :4], flag*error.sin().unsqueeze(-1), flag*error.cos().unsqueeze(-1), flag*g[..., 5:]), -1)
        # Delay is physical actuator memory, so the value function must see it.
        delay = s[..., self.layout.delay_start:]
        return torch.cat((invariant_features(self.layout.unpack(s)), c, delay, goal), -1)

    def reward_terms(self, s, a, ap, previous_peak, c, g, landing):
        L = self.layout
        S = L.unpack(s)
        P, R, V, W = self.kin(S)
        root_R = S[1]
        vb, wb = rotate(root_R.transpose(-1, -2), S[2]), rotate(root_R.transpose(-1, -2), S[3])
        command = g[..., :3]
        speed = command[..., :2].norm(dim=-1) + command[..., 2].abs()
        std = torch.where((speed < .05).unsqueeze(-1), self.std_standing,
                          torch.where((speed < 1.5).unsqueeze(-1), self.std_walking, self.std_running))
        torso_g = rotate(R[..., self.torso, :, :].transpose(-1, -2), self.gravity)
        offsets = rotate(R[..., self.feet, :, :], self.foot_offsets)
        feet, fvel = P[..., self.feet, :] + offsets, V[..., self.feet, :] + torch.cross(W[..., self.feet, :], offsets, dim=-1)
        mass, inertia = self.context.masses_inertias(c)
        off = rotate(R, self.context.com)
        com, cv = P + off, V + torch.cross(W, off, dim=-1)
        total = mass.sum(-1, keepdim=True).clamp_min(1e-6)
        center = (mass.unsqueeze(-1)*com).sum(-2)/total
        center_v = (mass.unsqueeze(-1)*cv).sum(-2)/total
        spin = rotate(R, rotate(inertia, rotate(R.transpose(-1, -2), W)))
        orbital = torch.cross(com-center.unsqueeze(-2), mass.unsqueeze(-1)*(cv-center_v.unsqueeze(-2)), dim=-1)
        momentum = (spin+orbital).sum(-2)
        height = feet[..., 2].clamp_min(0.)
        active = (speed > .05).to(s.dtype)
        peak = previous_peak + (1-s[..., L.contact])*torch.relu(height-previous_peak)
        q = S[4][..., self.action_ids]
        values = {
            'track_linear_velocity': torch.exp(-((command[..., :2]-vb[..., :2]).square().sum(-1)+vb[..., 2].square())/self.params['track_linear_velocity']['std']**2),
            'track_angular_velocity': torch.exp(-((command[..., 2]-wb[..., 2]).square()+wb[..., :2].square().sum(-1))/self.params['track_angular_velocity']['std']**2),
            'upright': torch.exp(-torso_g[..., :2].square().sum(-1)/self.params['upright']['std']**2),
            'pose': torch.exp(-((q-self.default_q[self.action_ids]).square()/std.square()).mean(-1)),
            'body_ang_vel': W[..., self.torso, :2].square().sum(-1),
            'angular_momentum': momentum.square().sum(-1),
            'joint_limits': (torch.relu(self.limits[:, 0]-q)+torch.relu(q-self.limits[:, 1])).sum(-1),
            'action_rate': (a-ap).square().sum(-1),
            'foot_clearance': ((height-self.params['foot_clearance']['target_height']).abs()*fvel[..., :2].norm(dim=-1)).sum(-1)*active,
            'foot_swing_height': ((peak/.1-1).square()*landing).sum(-1)*active,
            'foot_slip': (fvel[..., :2].square().sum(-1)*s[..., L.contact]).sum(-1)*active,
            'soft_landing': (s[..., L.force]*landing).sum(-1)*active,
        }
        return torch.stack([values[n]*self.weights[n]*self.dt for n in self.names], -1)

    def reward_and_done(self, s, a, ap, previous_peak, c, g, landing_logits):
        reward = self.reward_terms(s, a, ap, previous_peak, c, g, landing_logits.sigmoid()).sum(-1)
        upright = self.layout.unpack(s)[1][..., 2, 2]
        done = torch.sigmoid((s.new_tensor(self.limit_angle).cos()-upright)/.02)
        return reward, done


class Adapter:
    def __init__(self, env):
        from isaaclab_newton.physics import NewtonManager
        self.env = env.unwrapped
        env = self.env
        self.robot = env.scene['robot']
        self.action = env.action_manager.get_term('joint_pos')
        self.action_ids = list(self.action._joint_ids)
        acfg = self.action.cfg
        self.layout = Layout(len(self.robot.joint_names), len(self.action_ids), acfg.delay_max_lag,
                             env.cfg.decimation, acfg.delay_update_period, acfg.delay_min_lag)
        self.model = NewtonManager._solver.mj_model
        self.context = Context(env, self.model, self.robot, self.action_ids).to(env.device)
        self.mdp = VelocityMDP(env, self.model, self.layout, self.context, self.action_ids).to(env.device)
        self.has_contact = 'feet_contact' in env.scene.sensors
        if self.has_contact:
            self.sensor = env.scene['feet_contact']
            cfg = env.reward_manager.get_term_cfg('foot_slip').params['sensor_cfg']
            self.sensor_ids = cfg.body_ids
        self.default = tensor(self.robot.data.default_joint_pos)[0, self.action_ids].clone()
        scale = self.action._scale
        self.scale = torch.as_tensor(scale, device=env.device).expand(env.num_envs, self.layout.actions)[0].clone()
        self.mg = float(tensor(self.robot.data.body_mass).sum(-1).mean()) * 9.81
        self.target = self.reward_values = self.landing = self.reward_goal = self.previous_peak = None
        self.lag_trace = []
        self._original_apply = self.action.apply_actions
        def apply():
            self._original_apply()
            delay = self.action._delay_buffer
            self.lag_trace.append(delay.lags.clone().float() if delay is not None else torch.zeros(env.num_envs, device=env.device))
        self.action.apply_actions = apply
        self._original_reward = env.reward_manager.compute
        def capture(dt):
            self.previous_peak = self.state()[:, self.layout.peak].clone()
            self.reward_goal = self.goal().clone()
            self.landing = self.contacts()[2].clone()
            value = self._original_reward(dt)
            self.target = self.state().clone()
            self.reward_values = env.reward_manager._step_reward.clone()*dt
            return value
        env.reward_manager.compute = capture

    def close(self):
        self.action.apply_actions = self._original_apply
        self.env.reward_manager.compute = self._original_reward

    def contacts(self):
        if not self.has_contact:
            z = torch.zeros((self.env.num_envs, 2), device=self.env.device)
            return z, z, z, z
        data = self.sensor.data
        normal = tensor(data.net_normal_forces_w)[:, self.sensor_ids]
        total = normal
        if data.net_friction_forces_w is not None:
            total = total + tensor(data.net_friction_forces_w)[:, self.sensor_ids]
        contact = (normal.norm(dim=-1) > 0).float()
        age = tensor(data.current_contact_time)[:, self.sensor_ids]
        landing = tensor(self.sensor.compute_first_contact(dt=self.env.step_dt))[:, self.sensor_ids]
        return contact, age, landing, total.norm(dim=-1)

    def delay_state(self):
        delay = self.action._delay_buffer
        if delay is None:
            # Evaluation with delay disabled still uses the checkpoint's actor;
            # the training model itself is not needed for deployment.
            target = self.action.processed_actions
            return torch.cat((target, target.new_zeros((len(target), 2))), -1)
        rows = (torch.arange(self.layout.max_lag+1, device=self.env.device)+delay.cursor).remainder(self.layout.max_lag+1)
        queue = delay.values[rows].permute(1, 0, 2).flatten(1)
        return torch.cat((queue, delay.lags[:, None].float(), ((delay.ticks+delay.phases)%delay.update_period)[:, None].float()), -1)

    def state(self):
        d = self.robot.data
        contact, age, _, force = self.contacts()
        peak = (self.env.reward_manager.get_term_cfg('foot_swing_height').func.peak_heights
                if 'foot_swing_height' in self.mdp.names else torch.zeros_like(contact))
        quat = tensor(d.root_link_quat_w)[:, [3, 0, 1, 2]]  # Isaac Lab xyzw -> reference wxyz
        return torch.cat((tensor(d.root_link_pos_w)-self.env.scene.env_origins, quat,
                          tensor(d.root_link_lin_vel_w), tensor(d.root_link_ang_vel_w),
                          tensor(d.joint_pos), tensor(d.joint_vel), contact, age, force, peak, self.delay_state()), -1)

    def goal(self):
        cmd = self.env.command_manager.get_term('base_velocity')
        # Straight and standing masks override heading in the native task.
        heading = cmd.is_heading_env & ~cmd.is_standing_env & ~cmd.is_forward_env
        col = lambda x: torch.full_like(cmd.heading_target[:, None], x)
        return torch.cat((cmd.command, heading[:, None].float(), cmd.heading_target[:, None],
                          col(cmd.cfg.ranges.ang_vel_z[0]), col(cmd.cfg.ranges.ang_vel_z[1]),
                          col(cmd.cfg.heading_control_stiffness)), -1)

    def begin_step(self):
        self.lag_trace.clear()

    def executed_lags(self):
        if len(self.lag_trace) != self.layout.decimation:
            raise RuntimeError('Expected one action application per physics substep')
        return torch.stack(self.lag_trace, -1)
