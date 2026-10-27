"""History ownership for V3 policy exports in the existing NEST replay flow."""
import torch
from torch import nn


class HistoryPolicy(nn.Module):
    def __init__(self, actor, history=10, frame_dim=51):
        super().__init__()
        self.actor, self.history_length, self.frame_dim = actor, history, frame_dim
        self.register_buffer('history', torch.empty(0), persistent=False)

    def reset(self):
        self.history = self.history.new_empty(0)

    def forward(self, observation):
        if observation.shape[-1] != self.frame_dim + 3:
            raise ValueError('Expected the native 54-value Grasshopper observation')
        frame = observation[..., :self.frame_dim]
        if self.history.numel() == 0:
            self.history = frame[:,None].repeat(1,self.history_length,1)
        else:
            self.history = torch.cat((self.history[:,1:],frame[:,None]),1)
        return self.actor(self.history,observation[...,self.frame_dim:])
