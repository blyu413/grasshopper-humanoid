#!/usr/bin/env python3
"""After training, compare four mixed-force policies against their source checkpoint."""

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
PROFILES = {
    "short": {"forces": [0, 100, 150, 200], "args": ["--repeat-pushes", "3"], "cases": 13},
    "sustained_5s": {"forces": [0, 40, 80, 100, 150, 200],
                     "args": ["--push-duration", "5", "--seconds", "12"], "cases": 21},
    "combined": {"forces": [40, 80, 100],
                 "args": ["--push-duration", "5", "--seconds", "12", "--extra-pulse-force", "120", "--extra-pulse-at", "4"],
                 "cases": 12},
}


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def evaluate(policy, destination):
    results = {}
    policy_hash = hashlib.sha256(policy.read_bytes()).hexdigest()
    for name, profile in PROFILES.items():
        output = destination / name
        output.mkdir(parents=True, exist_ok=True)
        expected_count = 16 * profile["cases"]
        report = read_json(output / "results.json")
        if not (report and report["metadata"]["policy_sha256"] == policy_hash
                and len(report["trials"]) == expected_count):
            command = [sys.executable, "-u", str(ROOT / "scripts/test_grasshopper_push.py"),
                       "--device", "cuda:0", "--visualizer", "none", "--num-envs", "16",
                       "--policy", str(policy), "--forces", *map(str, profile["forces"]),
                       *profile["args"], "--output-dir", str(output)]
            with (output / "evaluation.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = read_json(output / "results.json")
        if not report or len(report["trials"]) != expected_count or report["metadata"]["policy_sha256"] != policy_hash:
            raise AssertionError(f"Incomplete or mismatched {name} evaluation")
        results[name] = report
        print(json.dumps({"policy": str(policy), "profile": name,
                          "survived": sum(r["survived"] for r in report["trials"]),
                          "recovered": sum(r["recovered_at_end"] for r in report["trials"]),
                          "trials": expected_count}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--package-dir", type=Path, default=ROOT / "outputs/grasshopper_stand_mixed_policies")
    parser.add_argument("--stages", type=int, nargs="+", default=[500, 1000, 1500, 2000])
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    print("WAITING_FOR_TRAINING_COMPLETION", flush=True)
    while True:
        manifest = read_json(run_dir / "run.json")
        if manifest and manifest["status"] == "failed":
            raise RuntimeError(f"Training failed: {manifest.get('error')}")
        if manifest and manifest["status"] == "complete":
            break
        time.sleep(10)
    exports = {entry["additional_iterations"]: entry for entry in manifest["exports"]}
    if not all(stage in exports for stage in args.stages):
        raise RuntimeError("Training did not produce all requested stage policies")

    package = args.package_dir.resolve()
    identity = {"training_run": str(run_dir), "source_sha256": manifest["source_sha256"], "profiles": PROFILES}
    if package.exists() and any(package.iterdir()) and read_json(package / "evaluation_job.json") != identity:
        raise RuntimeError("Package directory belongs to another run or evaluation protocol")
    package.mkdir(parents=True, exist_ok=True)
    (package / "evaluation_job.json").write_text(json.dumps(identity, indent=2) + "\n")
    if not (package / "baseline/policy.pt").is_file():
        with (package / "baseline_export.log").open("w") as log:
            subprocess.run([sys.executable, str(ROOT / "scripts/export_standing_checkpoint.py"),
                            manifest["source_checkpoint"], str(package / "baseline")],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    reports = {0: evaluate(package / "baseline/policy.pt", package / "baseline/evaluation")}
    import torch

    entries = []
    for stage in args.stages:
        export = exports[stage]
        target = package / f"after_{stage:04d}"
        shutil.copytree(Path(export["policy"]).parent, target, dirs_exist_ok=True)
        checkpoint = Path(export["checkpoint"])
        shutil.copy2(checkpoint, target / checkpoint.name)
        policy = torch.jit.load(str(target / "policy.pt"), map_location="cpu")
        state = torch.load(target / checkpoint.name, weights_only=True, map_location="cpu")
        if not all(torch.equal(value, state["actor_state_dict"][key]) for key, value in policy.state_dict().items()):
            raise AssertionError(f"Checkpoint/export mismatch at {stage}")
        reports[stage] = evaluate(target / "policy.pt", target / "evaluation")
        entries.append({"additional_iterations": stage, "checkpoint_iteration": state["iter"],
                        "policy": f"after_{stage:04d}/policy.pt", "onnx": f"after_{stage:04d}/policy.onnx",
                        "checkpoint": f"after_{stage:04d}/{checkpoint.name}",
                        "actor_weights_verified": True,
                        "policy_sha256": hashlib.sha256((target / "policy.pt").read_bytes()).hexdigest()})

    def score(stage):
        profiles = reports[stage]
        clean = sum(row["recovered"] for profile in profiles.values() for row in profile["summary"] if row["force_N"] == 0)
        sustained_and_combined = sum(row["recovered"] for name in ("sustained_5s", "combined")
                                    for row in profiles[name]["summary"] if 0 < row["force_N"] <= 100)
        short = sum(row["recovered"] for row in profiles["short"]["summary"] if row["force_N"] > 0)
        stress = sum(row["recovered"] for row in profiles["sustained_5s"]["summary"] if row["force_N"] > 100)
        return clean, sustained_and_combined, short, stress, stage

    recommended = max(args.stages, key=score)
    for name in PROFILES:
        paths = [package / "baseline/evaluation" / name / "results.json",
                 *[package / f"after_{stage:04d}/evaluation" / name / "results.json" for stage in args.stages]]
        subprocess.run([sys.executable, str(ROOT / "scripts/compare_grasshopper_push.py"),
                        *map(str, paths), "--labels", "Source", *[f"after_{s}" for s in args.stages],
                        "--output-dir", str(package / "comparison" / name)], cwd=ROOT, check=True)
    shutil.copytree(run_dir / "params", package / "training_params", dirs_exist_ok=True)
    shutil.copytree(run_dir / "source", package / "training_source", dirs_exist_ok=True)
    shutil.copy2(run_dir / "run.json", package / "training_run.json")
    (package / "manifest.json").write_text(json.dumps({
        "training_run": str(run_dir), "source_checkpoint": manifest["source_checkpoint"],
        "stages": entries, "recommended_stage": recommended,
        "recommended_policy": f"after_{recommended:04d}/policy.pt",
        "selection_rule": "Clean standing, then sustained/combined <=100 N recovery, short-force recovery, sustained stress, latest",
        "baseline_score": score(0), "stage_scores": {s: score(s) for s in args.stages},
        "evaluation_profiles": PROFILES, "samples_per_case": 16,
        "evaluation_solver": "Newton/MJWarp", "observation_dim": 52, "action_dim": 14,
    }, indent=2) + "\n")
    print(f"ALL_MIXED_POLICIES_PACKAGED recommended_stage={recommended} directory={package}", flush=True)


if __name__ == "__main__":
    main()
