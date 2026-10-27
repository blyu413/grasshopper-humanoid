"""Physical-state definitions extracted from pinned upstream mdp.py.

Task rewards/motion-library code is supplied by the native MoMo adapter.
"""
import torch
from .rotations import mat_to_6d, yaw_mat

NUM_J = 29
XI_DIM = 33
INV_DIM = 71

def invariant_features(S):
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat((p[..., 2:3], mat_to_6d(H.transpose(-1, -2) @ R),
                      (Rt @ v.unsqueeze(-1)).squeeze(-1),
                      (Rt @ w.unsqueeze(-1)).squeeze(-1), q, qd), -1)
