"""Differentiable FK from the compiled Newton/MuJoCo robot, including mimic DOFs."""
import numpy as np
import torch
from torch import nn
from .reference.rotations import quat_to_mat, axis_angle_to_mat


def find_named(model, kind, name):
    count = model.nbody if kind == 'body' else model.njnt
    matches = [i for i in range(count) if getattr(model, kind)(i).name == name
               or getattr(model, kind)(i).name.endswith('_' + name)
               or getattr(model, kind)(i).name.endswith('/' + name)]
    if len(matches) != 1:
        raise ValueError(f'Expected unique {kind} {name!r}, got {matches}')
    return matches[0]


def rotate(R, v):
    return (R @ v.unsqueeze(-1)).squeeze(-1)


class Kinematics(nn.Module):
    def __init__(self, model, joint_names, body_names, root_name='PELVIS'):
        super().__init__()
        root = find_named(model, 'body', root_name)
        bids = [find_named(model, 'body', name) for name in body_names]
        if len(set(bids)) != len(bids):
            raise ValueError('Duplicate body mapping')
        # Newton's asset body order is not necessarily topological.
        order = sorted(range(len(bids)), key=lambda i: bids[i])
        self.body_order = order
        self.body_ids = [bids[i] for i in order]
        self.output_order = [order.index(i) for i in range(len(order))]
        local = {b: i for i, b in enumerate(self.body_ids)}
        jids = [find_named(model, 'joint', name) for name in joint_names]
        jl = {j: i for i, j in enumerate(jids)}
        self.parents, self.joints = [], []
        axes, anchors, qzero = [], [], []
        for b in self.body_ids:
            parent = -1 if b == root else local.get(int(model.body_parentid[b]), -2)
            if parent == -2:
                raise ValueError('Missing fixed ancestor in compiled articulation')
            self.parents.append(parent)
            n, j = int(model.body_jntnum[b]), int(model.body_jntadr[b])
            if b == root:
                if n != 1 or model.jnt_type[j] != 0:
                    raise ValueError('Expected free root joint')
                index = -1
            elif n == 0:
                index = -1
            elif n == 1 and model.jnt_type[j] == 3 and j in jl:
                index = jl[j]
            else:
                raise ValueError('Expected fixed bodies or one named hinge per body')
            self.joints.append(index)
            axes.append(model.jnt_axis[j].copy() if index >= 0 else [0., 0., 1.])
            anchors.append(model.jnt_pos[j].copy() if index >= 0 else [0., 0., 0.])
            qzero.append(model.qpos0[model.jnt_qposadr[j]] if index >= 0 else 0.)
        self.root_index = bids.index(root)
        for name, value in dict(offset=model.body_pos[self.body_ids], rotation=model.body_quat[self.body_ids],
                                axes=axes, anchors=anchors, qzero=qzero).items():
            self.register_buffer(name, torch.tensor(np.asarray(value), dtype=torch.float32))
        self.rotation = quat_to_mat(self.rotation)

    def forward(self, S):
        p, R, v, w, q, qd = S
        positions, rotations, velocities, angular = [], [], [], []
        for i, parent in enumerate(self.parents):
            if parent < 0:
                pi, Ri, vi, wi = p, R, v, w
            else:
                Rp, pp, vp, wp = rotations[parent], positions[parent], velocities[parent], angular[parent]
                off = rotate(Rp, self.offset[i])
                pi, Ri, vi, wi = pp + off, Rp @ self.rotation[i], vp + torch.cross(wp, off, dim=-1), wp
                j = self.joints[i]
                if j >= 0:
                    # A nonzero hinge anchor moves the link origin as it rotates.
                    axis = rotate(Ri, self.axes[i])
                    anchor = rotate(Ri, self.anchors[i])
                    pivot = pi + anchor
                    pivot_v = vi + torch.cross(wi, anchor, dim=-1)
                    Ri = Ri @ axis_angle_to_mat(self.axes[i], q[..., j] - self.qzero[i])
                    wi = wi + axis * qd[..., j:j+1]
                    anchor = rotate(Ri, self.anchors[i])
                    pi = pivot - anchor
                    vi = pivot_v - torch.cross(wi, anchor, dim=-1)
            positions.append(pi); rotations.append(Ri); velocities.append(vi); angular.append(wi)
        return tuple(torch.stack(values, -3 if values is rotations else -2)[..., self.output_order, :, :]
                     if values is rotations else torch.stack(values, -2)[..., self.output_order, :]
                     for values in (positions, rotations, velocities, angular))
