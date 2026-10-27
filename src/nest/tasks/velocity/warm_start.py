"""Initialize standing-to-walking and forward-to-turning policy transfers."""

import torch


@torch.no_grad()
def initialize_yaw_command(model: torch.nn.Module, column: int, optimizer=None,
                           command_std: float = 0.5 / 3**0.5) -> None:
    """Activate an unseen yaw input while preserving the zero-yaw network output.

    Forward-only training leaves this feature at zero variance and its first-layer
    weights untrained. Seed its normalization scale and zero just that column;
    retain all other learned features, statistics and optimizer state.
    """
    weight = model.mlp[0].weight
    weight[:, column] = 0.0
    normalizer = model.obs_normalizer
    if hasattr(normalizer, "_std"):
        normalizer._mean[:, column] = 0.0
        normalizer._var[:, column] = command_std**2
        normalizer._std[:, column] = command_std
    if optimizer is not None:
        for state in optimizer.state.get(weight, {}).values():
            if isinstance(state, torch.Tensor) and state.shape == weight.shape:
                state[:, column] = 0.0


def initialize_from_standing(actor: torch.nn.Module, source: dict[str, torch.Tensor]) -> None:
    """Preserve the mean action for every command; retain fresh exploration noise.

    The critic and optimizer deliberately remain fresh because walking changes
    both the observation space and the return being predicted.
    """
    target = actor.state_dict()
    if set(target) != set(source):
        raise ValueError("Standing checkpoint does not match the walking actor architecture")
    for name, value in target.items():
        old = source[name].to(device=value.device, dtype=value.dtype)
        if name == "mlp.0.weight":
            if old.shape != (value.shape[0], 52) or value.shape[1] != 55:
                raise ValueError("Expected standing/walking actor inputs of 52/55")
            value.zero_()
            value[:, :52].copy_(old)
        elif name == "distribution.std_param":
            if old.shape != value.shape:
                raise ValueError("Standing checkpoint has the wrong action dimension")
            # Use the new task's configured exploration, not near-deterministic standing noise.
        elif old.shape != value.shape:
            raise ValueError(f"Incompatible actor tensor: {name}")
        else:
            value.copy_(old)
    actor.load_state_dict(target, strict=True)
