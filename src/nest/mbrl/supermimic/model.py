"""V3 networks with robot-sized state/action interfaces and exact motor delay.

The pinned reference MLPs, SO(3) operations, normalizers and initialization are
retained. MoMo is absent from NEST; the auxiliary branch predicts foot sensors.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .reference import networks as ref
from .reference.mdp import invariant_features
from .reference.rotations import quat_to_mat, mat_to_quat, yaw_mat, so3_exp

HISTORY = 10
VARIANT = "nest_grasshopper_reference_v3"
SMALL_VARIANT = "nest_grasshopper_small_mlp_bptt"


def policy_variant(profile):
    return SMALL_VARIANT if profile == 'small_mlp' else VARIANT


def make_policy(obs_dim, action_dim, profile='reference_v3'):
    if profile == 'small_mlp':
        return SmallPolicy(obs_dim, action_dim)
    if profile == 'reference_v3':
        return Policy(obs_dim, action_dim)
    raise ValueError(f'Unknown policy profile: {profile}')


@dataclass(frozen=True)
class Layout:
    joints: int
    actions: int
    max_lag: int = 4
    decimation: int = 4
    delay_period: int = 50
    min_lag: int = 0

    @property
    def physical(self): return 13 + 2 * self.joints
    @property
    def contact(self): return slice(self.physical, self.physical + 2)
    @property
    def age(self): return slice(self.physical + 2, self.physical + 4)
    @property
    def force(self): return slice(self.physical + 4, self.physical + 6)
    @property
    def peak(self): return slice(self.physical + 6, self.physical + 8)
    @property
    def delay_start(self): return self.physical + 8
    @property
    def delay_size(self): return (self.max_lag + 1) * self.actions + 2
    @property
    def size(self): return self.delay_start + self.delay_size
    @property
    def obs_dim(self): return 9 + 3 * self.actions

    def unpack(self, s):
        n = self.joints
        return s[..., :3], quat_to_mat(s[..., 3:7]), s[..., 7:10], s[..., 10:13], s[..., 13:13+n], s[..., 13+n:13+2*n]


def pack(S):
    p, R, v, w, q, qd = S
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


class Policy(ref.Policy):
    def __init__(self, obs_dim, action_dim, history=HISTORY):
        # Same layer construction/order and initialization as pinned Policy.
        nn.Module.__init__(self)
        self.hist = history
        self.enc = ref.mlp(3, [2048, 1024, 512, 256], 32)
        self.dec = ref.mlp(32 + obs_dim * history, [2048, 1024, 512, 256], action_dim)
        with torch.no_grad():
            self.dec[-1].weight.mul_(.01)
            self.dec[-1].bias.zero_()
        self.obs_norm, self.goal_norm = ref.RunningNorm(obs_dim), ref.RunningNorm(3)


class SmallPolicy(nn.Module):
    """V2-sized ELU actor; NEST history and current velocity goal interface.

    The small profile uses five complete 51-value NEST frames and the current
    goal. Unlike ADAPT's term-wise V2 packing, actions remain in each frame and
    goals are not stacked. This preserves NEST's existing observation semantics.
    """
    def __init__(self, obs_dim, action_dim, history=5):
        super().__init__()
        self.hist = history
        self.mlp = ref.mlp(obs_dim * history + 3, [512, 256, 128], action_dim)
        self.obs_norm, self.goal_norm = ref.RunningNorm(obs_dim), ref.RunningNorm(3)

    def forward(self, obs_hist, goal):
        x = torch.cat((self.obs_norm(obs_hist).flatten(-2), self.goal_norm(goal)), -1)
        return ref.run_mlp(self.mlp, x)


class MotorDelay(nn.Module):
    """Oldest-to-newest physical target queue, lag and resampling phase.

    Training supplies the recorded per-substep lags; imagination samples the
    same exogenous lag process. Encoder calibration is applied AFTER the queue,
    exactly as BiasedJointPositionAction.apply_actions does.
    """
    def __init__(self, layout, default, scale):
        super().__init__()
        self.layout = layout
        self.register_buffer('default', default.clone())
        self.register_buffer('scale', scale.clone())

    def forward(self, state, action, lags=None):
        L = self.layout
        queue = state[..., :-2].reshape(*state.shape[:-1], L.max_lag + 1, L.actions)
        lag, phase = state[..., -2], state[..., -1]
        target = self.default + action * self.scale
        executed, actual_lags = [], []
        for tick in range(L.decimation):
            if lags is None:
                draw = torch.randint(L.min_lag, L.max_lag + 1, lag.shape, device=action.device)
                lag = torch.where(phase == 0, draw.to(lag.dtype), lag)
            else:
                lag = lags[..., tick]
            queue = torch.cat((queue[..., 1:, :], target.unsqueeze(-2)), -2)
            ids = (L.max_lag - lag.long())[..., None, None].expand(*lag.shape, 1, L.actions)
            executed.append(queue.gather(-2, ids).squeeze(-2))
            actual_lags.append(lag)
            phase = (phase + 1).remainder(L.delay_period)
        new_state = torch.cat((queue.flatten(-2), lag.unsqueeze(-1), phase.unsqueeze(-1)), -1)
        # Physical context uses normalized PD targets. Static encoder bias is
        # represented independently in c, preserving the upstream action units.
        executed = (torch.stack(executed, -2) - self.default) / self.scale
        return new_state, executed.flatten(-2), torch.stack(actual_lags, -1)


class PhysicalWorldModel(ref.WorldModel):
    def __init__(self, layout, context_dim, widths=(2048, 2048, 2048)):
        nn.Module.__init__(self)
        self.layout = layout
        self.in_dim = layout.physical + layout.actions + context_dim
        out_dim = layout.physical - 1
        self.net = ref.mlp(self.in_dim, list(widths), out_dim, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = ref.RunningNorm(self.in_dim)
        self.out_norm = ref.RunningNorm(out_dim, eps=1e-4)
        self.register_buffer('error_floor', torch.tensor(
            [.003]*3 + [.005]*3 + [.02]*3 + [.05]*3 + [.003]*layout.joints + [.05]*layout.joints))

    def step(self, S, a, c):
        p, R, v, w, q, qd = S
        y = ref.run_mlp(self.net, self.in_norm(self.inputs(S, a, c)))
        d = y * self.out_norm.std() + self.out_norm.mean
        dp, dr, dv, dw, dq, dqd = d.split((3, 3, 3, 3, self.layout.joints, self.layout.joints), -1)
        rotate = lambda rot, x: (rot @ x.unsqueeze(-1)).squeeze(-1)
        return p + rotate(yaw_mat(R), dp), R @ so3_exp(dr), v + rotate(R, dv), w + rotate(R, dw), q + dq, qd + dqd

    def error(self, prediction, target):
        return self.state_error(self.layout.unpack(prediction), self.layout.unpack(target)) / torch.maximum(
            self.out_norm.std(), self.error_floor)


class ContactAuxiliary(nn.Module):
    def __init__(self, layout, context_dim, mg, dt, widths=(512, 256, 128)):
        super().__init__()
        self.layout, self.mg, self.dt = layout, mg, dt
        dim = 2 * layout.physical + 8 + layout.actions + context_dim
        self.net = ref.mlp(dim, list(widths), 10)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = ref.RunningNorm(dim)

    def inputs(self, s, physical, a, c):
        L = self.layout
        return torch.cat((invariant_features(L.unpack(s)), s[..., L.physical:L.delay_start],
                          invariant_features(L.unpack(physical)), a, c), -1)

    def forward(self, s, physical, a, c):
        y = ref.run_mlp(self.net, self.in_norm(self.inputs(s, physical, a, c)))
        contact = y[..., :2].sigmoid()
        age = contact * torch.relu(s[..., self.layout.age] + self.dt + self.dt * y[..., 2:4].tanh())
        force, peak = F.softplus(y[..., 4:6]) * self.mg, F.softplus(y[..., 6:8]) * .1
        return torch.cat((contact, age, force, peak), -1), y[..., :2], y[..., 8:10]


class WorldModel(nn.Module):
    def __init__(self, layout, context_dim, default, scale, mg, dt, cfg):
        super().__init__()
        self.layout = layout
        self.delay = MotorDelay(layout, default, scale)
        n = context_dim + layout.decimation * layout.actions
        self.physical = PhysicalWorldModel(layout, n, cfg.model_widths)
        self.auxiliary = ContactAuxiliary(layout, n, mg, dt, cfg.auxiliary_widths)

    def forward(self, s, a, c, detach_physics_for_auxiliary=False, lags=None):
        L = self.layout
        delay, executed, _ = self.delay(s[..., L.delay_start:], a, lags)
        context = torch.cat((c, executed), -1)
        physical = pack(self.physical.step(L.unpack(s), a, context))
        aux_s = torch.cat((s[..., :L.physical].detach(), s[..., L.physical:]), -1) if detach_physics_for_auxiliary else s
        aux, contact, landing = self.auxiliary(aux_s, physical.detach() if detach_physics_for_auxiliary else physical, a, context)
        return torch.cat((physical, aux, delay), -1), contact, landing

    @torch.no_grad()
    def update_normalization(self, s, target, a, c, lags):
        L = self.layout
        _, executed, _ = self.delay(s[..., L.delay_start:], a, lags)
        context = torch.cat((c, executed), -1)
        self.physical.in_norm.update(self.physical.inputs(L.unpack(s), a, context))
        self.physical.out_norm.update(self.physical.targets(L.unpack(s), L.unpack(target)))
        self.auxiliary.in_norm.update(self.auxiliary.inputs(s, target, a, context))


def set_precision(amp):
    if amp not in ('bf16', 'none'):
        raise ValueError(amp)
    ref.AMP_DTYPE = torch.bfloat16 if amp == 'bf16' else None
