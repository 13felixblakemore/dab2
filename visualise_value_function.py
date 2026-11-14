"""Plot max_a F(s, a, B(goal))^T B(goal) over continuous four-rooms states.

Run with no arguments for two example goals, or use e.g.
  python visualise_value_function.py --goal 0.8 0.8 --resolution 200
"""

import argparse
import ast
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fb_navigation.agents.forward_backward_jax import FBLearner
from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms
from fb_navigation.environments.continuous_four_rooms import visualize_environment


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", type=float, nargs=2, action="append",
                        metavar=("X", "Y"), help="Goal coordinates; repeat for multiple panels")
    parser.add_argument("--resolution", type=int, default=200)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=root / "value_function.png")
    args = parser.parse_args()
    if args.resolution < 2:
        parser.error("--resolution must be at least 2")
    checkpoint = args.checkpoint
    if checkpoint is None:
        tree = ast.parse((root / "dab.py").read_text())
        config = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "config" for t in node.targets)
        )
        checkpoint = root / config["path"]
    env = AdjustableFourRooms(doorway_scale=2.0, include_stay_action=False, seed=0)
    goals = np.asarray(args.goal or [[0.8, 0.8], [0.2, 0.2]], dtype=np.float32)
    for goal in goals:
        if not np.isfinite(goal).all() or np.any(goal <= 0) or np.any(goal >= env.size):
            parser.error(f"Goal {goal} must lie inside the environment")
        if any(b.xmin <= goal[0] <= b.xmax and b.ymin <= goal[1] <= b.ymax
               for b in env.wall_blocks):
            parser.error(f"Goal {goal} lies in a wall")
    learner = FBLearner.from_checkpoint(str(checkpoint))
    x = np.linspace(0, env.size, args.resolution)
    X, Y = np.meshgrid(x, x)
    states = jnp.asarray(np.column_stack([X.ravel(), Y.ravel()]))
    wall = np.zeros(X.shape, dtype=bool)
    for b in env.wall_blocks:
        wall |= (X >= b.xmin) & (X <= b.xmax) & (Y >= b.ymin) & (Y <= b.ymax)
    maps = []
    for goal in goals:
        z = learner.embed_B(jnp.asarray(goal)[None, :])[0]
        # Batch evaluation bounds memory use for dense continuous-state grids.
        chunks = []
        for offset in range(0, len(states), 4096):
            batch = states[offset:offset + 4096]
            f = learner.embed_F(batch, jnp.broadcast_to(z, (len(batch), len(z))))
            q = jnp.einsum("nda,d->na", f, z)
            chunks.append(np.asarray(jnp.max(q, axis=-1)))
        maps.append(np.ma.masked_where(wall, np.concatenate(chunks).reshape(X.shape)))
    vmin = min(float(values.min()) for values in maps)
    vmax = max(float(values.max()) for values in maps)
    fig, axes = plt.subplots(1, len(goals), figsize=(6 * len(goals), 5.5),
                             squeeze=False, layout="constrained")
    for ax, goal, values in zip(axes.flat, goals, maps):
        heatmap = ax.pcolormesh(X, Y, values, shading="nearest", cmap="magma",
                                vmin=vmin, vmax=vmax, rasterized=True)
        visualize_environment(env, ax, draw_grid=False,
                              draw_initial_position=False, draw_goal=False)
        ax.scatter(*goal, marker="*", s=180, c="#ff9c35", edgecolors="white",
                   linewidths=0.7, zorder=6)
        ax.set(xlim=(0, env.size), ylim=(0, env.size), xlabel="x", ylabel="y",
               title=f"Goal ({goal[0]:.2f}, {goal[1]:.2f})")
    fig.colorbar(heatmap, ax=list(axes.flat), label=r"$\max_a F(s,a,B(g))^\top B(g)$",
                 shrink=0.85)
    fig.suptitle(f"Learned value function (embedding dimension {learner.embed_B(jnp.asarray(goals[:1])).shape[-1]})")
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
