#!/usr/bin/env python3
"""Compare policies evaluated with the same Grasshopper push-test protocol."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nest-push-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.results]
    labels = args.labels or [path.parent.name for path in args.results]
    if len(labels) != len(reports):
        parser.error("Provide one label per result file")
    for report in reports[1:]:
        for key in ("solver", "seed", "samples_per_case", "body", "force_frame", "control_dt_s",
                    "push_duration_s", "push_times_s", "trial_duration_s", "fall_thresholds", "recovery_thresholds"):
            if report["metadata"][key] != reports[0]["metadata"][key]:
                raise ValueError(f"Evaluation protocol differs: {key}")
        first_pulse = reports[0]["metadata"].get("extra_pulse", {})
        other_pulse = report["metadata"].get("extra_pulse", {})
        if first_pulse.get("force_N", 0) or other_pulse.get("force_N", 0):
            if first_pulse != other_pulse:
                raise ValueError("Evaluation extra-pulse protocol differs")
    common = set.intersection(*[{(r["force_N"], r["direction"]) for r in report["summary"]} for report in reports])
    directions = ["+X", "-X", "+Y", "-Y"]
    cases = sorted(common, key=lambda item: (item[0], directions.index(item[1])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), layout="constrained", sharex=True)
    table = []
    x = np.arange(len(cases))
    width = 0.8 / len(reports)
    for index, (label, report) in enumerate(zip(labels, reports)):
        lookup = {(r["force_N"], r["direction"]): r for r in report["summary"]}
        for ax, field in zip(axes, ["survived", "recovered"]):
            ax.bar(x - 0.4 + width / 2 + index * width,
                   [lookup[c][field] / lookup[c]["trials"] * 100 for c in cases], width, label=label)
        for case in cases:
            table.append({"policy": label, **lookup[case]})
    for ax, title in zip(axes, ["No fall during trial", "Stable at trial end"]):
        ax.set_title(title)
        ax.set_ylabel("Trials (%)")
        ax.set_ylim(0, 108)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper right")
    axes[1].set_xticks(x, ["No force" if force == 0 else f"{force:g} N\n{direction}" for force, direction in cases])
    meta = reports[0]["metadata"]
    fig.suptitle(f"Grasshopper standing: before/after push fine tuning\n{meta['solver']} | "
                 f"{meta['samples_per_case']} paired samples per case | duration: {meta['push_duration_s']:g} s | "
                 f"push times: {meta['push_times_s']} s" +
                 (f" | extra pulse: {meta['extra_pulse']['force_N']:g} N at {meta['extra_pulse']['start_s']:g} s"
                  if meta.get('extra_pulse', {}).get('force_N', 0) else ''), fontsize=13)
    fig.savefig(args.output_dir / "comparison.png", dpi=160)
    with (args.output_dir / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print((args.output_dir / "comparison.png").resolve())


if __name__ == "__main__":
    main()
