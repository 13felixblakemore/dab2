import argparse
from pathlib import Path
import dab_first_hitting
import numpy as np
import matplotlib.pyplot as plt
import fb_navigation.environments.adjustable_four_rooms
from fb_navigation.environments.continuous_four_rooms import visualize_environment

def choose_bottlenecks(mean_dab):
    print(mean_dab.shape)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dab_path", required=True, help="Path to a dab np array")
    parser.add_argument("--doorway_scale", required=True, type=float, default=1.0, help="Doorway width")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", default=0)
    return parser.parse_args()

def find_local_maxima(dab, valid_mask, k=1):
    """Return (row, column) indices strictly above all adjacent valid values."""
    valid = np.asarray(valid_mask, dtype=bool).reshape(dab.shape) & np.isfinite(dab)
    rows, cols = dab.shape
    maxima = []
    for row, col in np.argwhere(valid):
        row_start = max(0, row - 1)
        col_start = max(0, col - 1)
        neighborhood = np.s_[row_start:min(rows, row + (1+k)), col_start:min(cols, col + (1+k))]
        neighbors = valid[neighborhood].copy()
        neighbors[row - row_start, col - col_start] = False
        if np.all(dab[row, col] > dab[neighborhood][neighbors]):
            maxima.append((row, col))
    return maxima

def plot_maxima(dab, maxima, env, output_path, X, Y):
    fig, ax = plt.subplots(figsize=(7, 6))
    mesh = ax.pcolormesh(X, Y, dab, shading="nearest", cmap="viridis")
    visualize_environment(
        env, ax, draw_grid=False,
        draw_initial_position=False, draw_goal=False
    )
    ax.set(xlim=(0, env.size), ylim=(0, env.size), xlabel="x", ylabel="y")
    ax.set_aspect("equal")
    fig.colorbar(mesh, ax=ax)
    fig.tight_layout()
    for row, col in maxima:
        ax.scatter(
            X[row, col], Y[row, col],
            s=80, facecolors="none", edgecolors="red", linewidths=2,
            zorder=10,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved", output_path)


def main():
    args = arguments()
    with np.load(args.dab_path) as data:
        X, Y, valid_mask = data["X"], data["Y"], data["valid_mask"]
        dab = dab_first_hitting.fill_grid(data["mean_clipped"], valid_mask).reshape(X.shape)
    maxima = find_local_maxima(dab, valid_mask, k=10)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = fb_navigation.environments.adjustable_four_rooms.AdjustableFourRooms(
        doorway_scale=args.doorway_scale,
        include_stay_action=False,
        seed=args.seed + 3,
    )

    plot_maxima(dab, maxima, env, output_dir / "local_maxima.png", X, Y)

if __name__ == "__main__":
    main()
