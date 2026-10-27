#!/usr/bin/env python3
"""Plot survival and recovery rates from test_grasshopper_push.py results."""

import argparse
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
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    report = json.loads(args.results.read_text())
    meta, rows = report["metadata"], report["summary"]
    forces = sorted({r["force_N"] for r in rows if r["force_N"] > 0})
    directions = ["+X", "-X", "+Y", "-Y"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, field, title in zip(axes, ["survived", "recovered"], ["No fall during trial", "Stable at trial end"]):
        rates = np.full((len(forces), len(directions)), np.nan)
        for i, force in enumerate(forces):
            for j, direction in enumerate(directions):
                row = next((r for r in rows if r["force_N"] == force and r["direction"] == direction), None)
                if row:
                    rates[i, j] = row[field] / row["trials"]
        im = ax.imshow(rates, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(directions)), directions)
        ax.set_yticks(range(len(forces)), [f"{f:g} N" for f in forces])
        ax.set_xlabel("World force direction")
        ax.set_ylabel("Torso force (0.2 s pulse)")
        ax.set_title(title)
        for i, force in enumerate(forces):
            for j, direction in enumerate(directions):
                row = next((r for r in rows if r["force_N"] == force and r["direction"] == direction), None)
                if row:
                    ax.text(j, i, f"{row[field]}/{row['trials']}", ha="center", va="center", fontsize=13)
    baseline = next(r for r in rows if r["force_N"] == 0)
    fig.suptitle(f"Grasshopper standing policy: torso push test\n"
                 f"{meta['solver']} | pulse at {meta['push_start_s']:g} s | {meta['trial_duration_s']:g} s trials | "
                 f"no-force baseline: {baseline['survived']}/{baseline['trials']} standing", fontsize=13)
    fig.colorbar(im, ax=axes, label="Fraction of trials", shrink=0.8)
    target = args.results.with_name("push_test_summary.png")
    fig.savefig(target, dpi=180)
    print(target.resolve())


if __name__ == "__main__":
    main()
