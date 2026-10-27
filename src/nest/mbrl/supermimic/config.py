"""V3 training defaults copied from Motion Force Adapter (2026-10-27)."""
from dataclasses import dataclass
import math

@dataclass
class Config:
    policy_profile: str = 'reference_v3'
    window: int = 8
    horizon: int = 32
    model_widths: tuple[int, ...] = (2048, 2048, 2048)
    model_batches: int = 10
    model_batch_size: int = 2048
    imagination_batch_size: int = 4096
    learning_rate: float = 2e-4
    gamma: float = 0.99
    lam: float = 0.95
    critic_epochs: int = 5
    critic_minibatches: int = 4
    action_noise: float = 0.1
    replay_steps: int = 1000
    update_to_data_ratio: float = 0.4
    decay_iterations: int = 250_000
    final_lr_fraction: float = 0.03
    warmup_steps: int = 64  # Not reported in the paper.
    replay_device: str = 'cuda:0'  # Use cuda:0 when replay + BPTT fit GPU memory.

    model_only_iterations: int = 500
    auxiliary_widths: tuple[int, ...] = (512, 256, 128)
    amp: str = 'bf16'
    compile: bool = True

    def __post_init__(self):
        if self.policy_profile not in ('reference_v3', 'small_mlp'):
            raise ValueError('Unknown policy profile')
        for name in ('window', 'horizon', 'model_batches', 'model_batch_size',
                     'imagination_batch_size', 'critic_epochs', 'critic_minibatches',
                     'replay_steps', 'decay_iterations'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        if self.replay_steps < self.window or self.warmup_steps < self.window:
            raise ValueError('Replay and warm-up must fit one model window')
        if self.model_only_iterations < 0:
            raise ValueError('model_only_iterations must be nonnegative')
        if not 0 < self.update_to_data_ratio <= 1:
            raise ValueError('This collector supports 0 < update-to-data ratio <= 1')

    def lr(self, iteration: int) -> float:
        fraction = min(iteration / self.decay_iterations, 1.0)
        return self.learning_rate * (self.final_lr_fraction +
            (1 - self.final_lr_fraction) * 0.5 * (1 + math.cos(math.pi * fraction)))

    def collection_steps(self, iteration: int) -> int:
        # 0.4 policy/model updates per vectorized simulator control step: 2,3,2,3...
        return math.floor((iteration + 1) / self.update_to_data_ratio) - math.floor(
            iteration / self.update_to_data_ratio)


