#!/usr/bin/env python3
"""Evaluate all four fine-tuning exports as they become available, then package them."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_manifest(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--package-dir", type=Path, default=ROOT / "outputs/grasshopper_stand_push_policies")
    parser.add_argument("--stages", type=int, nargs="+", default=[500, 1000, 1500, 2000])
    parser.add_argument("--after-training", action="store_true", help="Wait for the entire training run to finish.")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.after_training:
        print("WAITING_FOR_TRAINING_COMPLETION", flush=True)
        while True:
            manifest = read_manifest(run_dir / "run.json")
            if manifest and manifest["status"] == "failed":
                raise RuntimeError(f"Training failed: {manifest.get('error')}")
            if manifest and manifest["status"] == "complete":
                break
            time.sleep(10)
    evaluations, exports = {}, {}
    for stage in args.stages:
        while True:
            manifest = read_manifest(run_dir / "run.json")
            if manifest and manifest["status"] == "failed":
                raise RuntimeError(f"Training failed: {manifest.get('error')}")
            export = next((e for e in (manifest or {}).get("exports", [])
                           if e["additional_iterations"] == stage), None)
            if export:
                break
            if manifest and manifest["status"] == "complete":
                raise RuntimeError(f"Training completed without the {stage}-iteration export")
            time.sleep(10)
        destination = ROOT / f"outputs/grasshopper_push_test/gpu_repeated_after_{stage:04d}"
        destination.mkdir(parents=True, exist_ok=True)
        existing = read_manifest(destination / "results.json")
        expected_cases = {(0.0, "+X"), *[(f, d) for f in (100.0, 150.0, 200.0) for d in ("+X", "-X", "+Y", "-Y")]}
        export_hash = hashlib.sha256(Path(export["policy"]).read_bytes()).hexdigest()
        reusable = (existing is not None and len(existing.get("trials", [])) == 208
                    and existing["metadata"]["policy_sha256"] == export_hash
                    and existing["metadata"]["solver"] == "Newton/MJWarp"
                    and existing["metadata"]["push_times_s"] == [2.0, 4.0, 6.0]
                    and existing["metadata"]["push_duration_s"] == 0.2
                    and existing["metadata"]["samples_per_case"] == 16
                    and existing["metadata"]["seed"] == 42
                    and {(r["force_N"], r["direction"]) for r in existing["summary"]} == expected_cases)
        print(f"STAGE_EVALUATION_{'REUSED' if reusable else 'STARTED'} additional_iterations={stage}", flush=True)
        command = [sys.executable, "-u", str(ROOT / "scripts/test_grasshopper_push.py"),
                   "--device", "cuda:0", "--visualizer", "none", "--num-envs", "16",
                   "--policy", export["policy"], "--forces", "0", "100", "150", "200",
                   "--repeat-pushes", "3", "--output-dir", str(destination)]
        if not reusable:
            with (destination / "evaluation.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        results = read_manifest(destination / "results.json")
        if len(results["trials"]) != 208:
            raise AssertionError(f"Incomplete evaluation at stage {stage}")
        if results["metadata"]["policy_sha256"] != export_hash:
            raise AssertionError("Evaluation used the wrong policy")
        evaluations[stage], exports[stage] = results, export
        print(json.dumps({"stage": stage, "cases": results["summary"]}), flush=True)
        print(f"STAGE_EVALUATION_COMPLETE additional_iterations={stage}", flush=True)

    def score(stage):
        cases = evaluations[stage]["summary"]
        mild = sum(r["recovered"] for r in cases if r["force_N"] <= 150)
        stress = [r for r in cases if r["force_N"] == 200]
        recovered = sum(r["recovered"] for r in stress)
        recovery = sum((r["mean_recovery_s"] or 0) * r["recovered"] for r in stress) / max(1, recovered)
        return mild, recovered, -recovery, stage

    recommended = max(args.stages, key=score)
    package = args.package_dir.resolve()
    package.mkdir(parents=True, exist_ok=False)
    import torch

    packaged = []
    for stage in args.stages:
        export = exports[stage]
        destination = package / f"after_{stage:04d}"
        shutil.copytree(Path(export["policy"]).parent, destination)
        checkpoint = Path(export["checkpoint"])
        shutil.copy2(checkpoint, destination / checkpoint.name)
        shutil.copytree(ROOT / f"outputs/grasshopper_push_test/gpu_repeated_after_{stage:04d}",
                        destination / "evaluation")
        policy = torch.jit.load(str(destination / "policy.pt"), map_location="cpu")
        state = torch.load(destination / checkpoint.name, weights_only=True, map_location="cpu")
        if not all(torch.equal(value, state["actor_state_dict"][key]) for key, value in policy.state_dict().items()):
            raise AssertionError(f"Export/checkpoint weight mismatch at {stage}")
        packaged.append({"additional_iterations": stage, "checkpoint_iteration": state["iter"],
                         "policy": f"after_{stage:04d}/policy.pt",
                         "onnx": f"after_{stage:04d}/policy.onnx",
                         "checkpoint": f"after_{stage:04d}/{checkpoint.name}",
                         "policy_sha256": hashlib.sha256((destination / "policy.pt").read_bytes()).hexdigest(),
                         "actor_weights_verified": True,
                         "evaluation": f"after_{stage:04d}/evaluation/results.json"})
    baseline = ROOT / "outputs/grasshopper_push_test/gpu_repeated_before"
    shutil.copytree(baseline, package / "baseline")
    shutil.copytree(run_dir / "params", package / "training_params")
    shutil.copy2(run_dir / "run.json", package / "training_run.json")
    (package / "manifest.json").write_text(json.dumps({
        "training_run": str(run_dir), "stages": packaged,
        "recommended_stage": recommended, "recommended_policy": f"after_{recommended:04d}/policy.pt",
        "selection_rule": "Maximize recovery for 0/100/150 N, then 200 N; prefer faster 200 N recovery; latest breaks ties",
        "evaluation_solver": "Newton/MJWarp", "pulse_times_s": [2, 4, 6], "pulse_duration_s": 0.2,
        "samples_per_case": 16, "observation_dim": 52, "action_dim": 14,
    }, indent=2) + "\n")
    command = [sys.executable, str(ROOT / "scripts/compare_grasshopper_push.py"),
               str(baseline / "results.json"),
               *[str(package / f"after_{s:04d}/evaluation/results.json") for s in args.stages],
               "--labels", "Original", *[f"after_{s}" for s in args.stages],
               "--output-dir", str(package / "comparison")]
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"ALL_POLICIES_PACKAGED recommended_stage={recommended} directory={package}", flush=True)


if __name__ == "__main__":
    main()
