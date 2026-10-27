"""Unchanged V3 gradient freezing and lambda-return helpers."""
from contextlib import contextmanager
import torch

@contextmanager
def frozen(*modules):
    """Freeze parameter gradients, preserving input/action Jacobians."""
    parameters = [p for module in modules for p in module.parameters()]
    flags = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


def lambda_returns(rewards, dones, values, gamma, lam):
    """Eq. 3: values includes V_H; terminals differ from time-limit truncations."""
    result, target = [], values[-1]
    for t in reversed(range(len(rewards))):
        target = rewards[t] + gamma * (1 - dones[t]) * (
            (1 - lam) * values[t + 1] + lam * target)
        result.append(target)
    return torch.stack(result[::-1])



def ensure_compile_headers():
    """Same CPython header fallback used by Motion Force Adapter's V3 entry."""
    import os
    import sys
    import sysconfig
    from pathlib import Path
    include = Path(sysconfig.get_path('include'))
    if (include/'Python.h').exists():
        return str(include)
    version = f'{sys.version_info.major}.{sys.version_info.minor}'
    matches = sorted((Path.home()/'.local/share/uv/python').glob(f'cpython-{version}*/include/python{version}/Python.h'))
    if not matches:
        raise RuntimeError(f'torch.compile requires Python {version} development headers; use --no-compile for eager preflight')
    include = matches[-1].parent
    os.environ['CPATH'] = str(include)+(':'+os.environ['CPATH'] if os.environ.get('CPATH') else '')
    return str(include)
