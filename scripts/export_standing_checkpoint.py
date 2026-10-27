#!/usr/bin/env python3
"""Export a standing checkpoint using the verified TorchScript network architecture."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "logs/rsl_rl/grasshopper_stand/2026-09-24_18-54-51/exported/policy.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    policy = torch.jit.load(str(args.reference), map_location="cpu").eval()
    # The actor's Gaussian distribution controls training exploration. Its parameters
    # are intentionally absent from the deterministic inference export.
    keys = set(policy.state_dict())
    actor = checkpoint["actor_state_dict"]
    if set(actor) - keys != {"distribution.std_param"} or not keys <= set(actor):
        raise ValueError("Checkpoint architecture does not match the standing reference export")
    policy.load_state_dict({key: actor[key] for key in keys}, strict=True)
    if not all(torch.equal(value, actor[key]) for key, value in policy.state_dict().items()):
        raise AssertionError("Exported actor weights differ from checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    policy.save(str(args.output_dir / "policy.pt"))
    input_dim = actor["mlp.0.weight"].shape[1]
    with torch.inference_mode():
        probe = torch.zeros(1, input_dim)
        if not torch.isfinite(policy(probe)).all():
            raise AssertionError("Export produced nonfinite actions")
        torch.onnx.export(policy, probe, str(args.output_dir / "policy.onnx"), dynamo=False,
                          input_names=["obs"], output_names=["actions"],
                          dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}}, opset_version=17)
    (args.output_dir / "export.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()), "iteration": checkpoint["iter"],
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "reference_architecture": str(args.reference.resolve()), "actor_weights_verified": True,
        "observation_dim": input_dim, "action_dim": actor["mlp.6.weight"].shape[0],
    }, indent=2) + "\n")
    print((args.output_dir / "policy.pt").resolve())


if __name__ == "__main__":
    main()
