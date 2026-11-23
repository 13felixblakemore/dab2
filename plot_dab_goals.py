"""Plot five fixed-goal DAB maps from a saved first-hitting run."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Path to first_hitting_results.npz")
    args = parser.parse_args()

    with np.load(args.results) as data:
        x, y, valid = data["X"], data["Y"], data["valid_mask"]
        hits = data["first_hit_clipped"]
        goals, groups = np.unique(data["v"], axis=0, return_inverse=True)
    if len(goals) < 5:
        parser.error("The results must contain at least five distinct goals")

    # Choose saved goals near the four room centers and the lower middle doorway.
    anchors = [(0.25, 0.75), (0.75, 0.75), (0.25, 0.25), (0.75, 0.25), (0.50, 0.25)]
    available = np.ones(len(goals), dtype=bool)
    examples = []
    for anchor in anchors:
        distances = np.sum((goals - anchor) ** 2, axis=1)
        idx = np.argmin(np.where(available, distances, np.inf))
        available[idx] = False
        selected = hits[groups == idx][:, valid]
        counts = np.isfinite(selected).sum(axis=0)
        scores = np.full(valid.shape, np.nan)
        scores[valid] = np.divide(np.nansum(selected, axis=0), counts,
                                  out=np.full(counts.shape, np.nan), where=counts > 0)
        examples.append((goals[idx], len(selected), scores.reshape(x.shape)))

    vmax = max(np.nanmax(scores) for _, _, scores in examples)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.3), layout="constrained")
    for ax, (goal, count, scores) in zip(axes, examples):
        ax.set_facecolor("black")
        mesh = ax.pcolormesh(x, y, scores, shading="nearest", vmin=0, vmax=vmax)
        ax.scatter(*goal, color="white", edgecolor="black", s=55)
        ax.set(aspect="equal", xlim=(x.min(), x.max()), ylim=(y.min(), y.max()),
               title=f"Goal ({goal[0]:.2f}, {goal[1]:.2f})\n{count} starts")
    fig.colorbar(mesh, ax=axes, label="Mean clipped first-hit DAB")
    output = args.results.with_name("first_hit_five_goals.png")
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
