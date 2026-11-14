"""Generate five successor-density heatmaps using the checkpoint in dab.py."""

import ast
from pathlib import Path

import jax
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
    # Read the configuration without executing dab.py's expensive DAB sweep.
    tree = ast.parse((root / "dab.py").read_text())
    config = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "config" for t in node.targets)
    )
    env = AdjustableFourRooms(
        doorway_scale=2.0, include_stay_action=False, seed=1
    )
    learner = FBLearner.from_checkpoint(str(root / config["path"]))
    x = jnp.linspace(0, 1, 50)
    X, Y = jnp.meshgrid(x, x)
    states = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    u, v = env.sample_points(5), env.sample_points(5)
    z = learner.embed_B(v)
    f = learner.embed_F(u, z)
    temperature = learner.temperature
    q = jnp.einsum("nda,nd->na", f, z)
    probs = jax.nn.softmax(q / temperature, axis=-1)
    densities = np.asarray(
        jnp.einsum("nda,na->nd", f, probs) @ learner.embed_B(states).T
    )
    wall = np.zeros(X.shape, dtype=bool)
    for b in env.wall_blocks:
        wall |= (
            (np.asarray(X) >= b.xmin) & (np.asarray(X) <= b.xmax)
            & (np.asarray(Y) >= b.ymin) & (np.asarray(Y) <= b.ymax)
        )
    valid = densities[:, ~wall.ravel()]
    limits = float(valid.min()), float(valid.max())
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout="constrained")
    for i, ax in enumerate(axes.flat[:5]):
        values = np.ma.masked_where(wall, densities[i].reshape(X.shape))
        mesh = ax.pcolormesh(
            X, Y, values, shading="nearest", cmap="viridis",
            vmin=limits[0], vmax=limits[1],
        )
        visualize_environment(
            env, ax, draw_grid=False, draw_initial_position=False, draw_goal=False
        )
        ax.scatter(*np.asarray(u[i]), marker="o", s=85, c="white",
                   edgecolors="black", zorder=5, label="Start u")
        ax.scatter(*np.asarray(v[i]), marker="*", s=170, c="red",
                   edgecolors="black", zorder=6, label="Goal v")
        ax.set(xlim=(0, env.size), ylim=(0, env.size),
               title=f"Example {i + 1}", xlabel="x", ylabel="y")
    axes.flat[5].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[5].legend(handles, labels, loc="center", fontsize=16, frameon=False)
    fig.colorbar(mesh, ax=list(axes.flat[:5]),
                 label="Learned successor density", shrink=0.8)
    fig.suptitle("Successor measure: five start-goal pairs", fontsize=20)
    output = root / "successor_measure_5_examples.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
