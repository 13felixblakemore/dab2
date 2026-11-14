import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from fb_navigation.agents.forward_backward_jax import FBLearner
from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms
from fb_navigation.environments.continuous_four_rooms_jax import ContinuousFourRooms
from fb_navigation.environments.continuous_four_rooms import visualize_environment


def visualise_dab(dab, X, Y, env, *, title=None,
                  colorbar_label="Successor-density ratio"):
    """Plot flat/grid DAB values or their mean over a leading pair axis."""
    X, Y = np.asarray(X), np.asarray(Y)
    if X.ndim != 2 or X.shape != Y.shape:
        raise ValueError("X and Y must be matching 2D coordinate grids")

    values = np.ma.masked_invalid(np.ma.asarray(dab))
    averaged = False
    if values.shape == X.shape:
        pass  # Already a spatial grid; do not average a coordinate axis.
    elif values.shape == (X.size,):
        values = values.reshape(X.shape)
    elif values.ndim == 2 and values.shape[1] == X.size:
        values = values.mean(axis=0).reshape(X.shape)
        averaged = True
    elif values.ndim == 3 and values.shape[1:] == X.shape:
        values = values.mean(axis=0)
        averaged = True
    else:
        raise ValueError(
            "DAB must have shape (grid_size,), X.shape, "
            "(n_pairs, grid_size), or (n_pairs, *X.shape)"
        )

    # Wall interiors have no valid state density and must not set color limits.
    wall_mask = np.zeros(X.shape, dtype=bool)
    for block in env.wall_blocks:
        wall_mask |= (
            (X >= block.xmin) & (X <= block.xmax)
            & (Y >= block.ymin) & (Y <= block.ymax)
        )
    values = np.ma.masked_where(wall_mask, values)

    fig, ax = plt.subplots(figsize=(7, 6))
    heatmap = ax.pcolormesh(X, Y, values, shading="nearest", cmap="viridis", zorder=0)
    visualize_environment(env, ax, draw_grid=False, draw_initial_position=False, draw_goal=False)
    ax.set_xlim(0, env.size)
    ax.set_ylim(0, env.size)
    ax.set_title(title if title is not None else (
        "Mean DAB over sampled pairs" if averaged else "DAB"
    ))
    fig.colorbar(heatmap, ax=ax, label=colorbar_label)
    fig.tight_layout()
    return fig, ax


def visualise_successor_measure(density, X, Y, env, *, start, goal, title=None):
    """Plot F_pi(start, B(goal)) dot B(s) over destination states s.

    This is the learned successor density used in the DAB numerator,
    without division by the diagonal successor density.
    """
    fig, ax = visualise_dab(
        density, X, Y, env,
        title=title or "Learned successor measure",
        colorbar_label="Learned successor density",
    )
    start, goal = np.asarray(start), np.asarray(goal)
    ax.scatter(*start, marker="o", s=90, c="white", edgecolors="black",
               label="Start u", zorder=5)
    ax.scatter(*goal, marker="*", s=180, c="red", edgecolors="black",
               label="Goal v", zorder=6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, ax


def main():
    config= {
        "seed": 0,
        "path": "experiments/fourrooms_doorways/runs/width2_replay_continuing_orthonormal_seed0_20261109_084240/epoch_0800.pkl",
    }

    # Instantiate environment and FB
    env = AdjustableFourRooms(
        doorway_scale=2.0,
        include_stay_action=False,
        seed=int(config["seed"]+3),
    )
    learner = FBLearner.from_checkpoint(config["path"])

    # mesh / linspace of states s (50x50)
    x = jnp.linspace(0, 1, 50)
    y = jnp.linspace(0, 1, 50)

    # Destination-state grid for the successor measure and DAB.

    X, Y = jnp.meshgrid(x, y)
    states = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

    # embed states
    embedded_states = learner.embed_B(states)

    # sample n (u,v) pairs

    u = env.sample_points(1000)
    v = env.sample_points(1000)

    z = learner.embed_B(v)
    f_num = learner.embed_F(u, z)

    f_denom = jax.vmap(
        lambda task: learner.embed_F(
            states, jnp.broadcast_to(task, (len(states), task.shape[0]))
        )
    )(z)
    # shape: (1000, 2500, 50, 4)

    q = jnp.einsum("nda,nd->na", f_num, z) # 1000, 4
    q_denom = jnp.einsum("nsda,nd->nsa", f_denom, z) # 1000, 2500, 4

    a_num = jnp.argmax(q, axis=-1)
    a_denom = jnp.argmax(q_denom, axis=-1)

    f_pi = jnp.einsum(
        "nda,na->nd",
        f_num,
        jax.nn.one_hot(a_num, f_num.shape[-1])
    )

    f_pi_denom = jnp.einsum(
        "nsda,nsa->nsd",
        f_denom,
        jax.nn.one_hot(a_denom, f_denom.shape[-1])
    )

    densities_num = f_pi @ embedded_states.T

    densities_denom = jnp.einsum(
        "nsd,sd->ns", f_pi_denom, embedded_states
    )                                                     # (1000, 2500)

    eps = 1e-4  # Diagnostic threshold; tune to density scale

    valid = (
        jnp.isfinite(densities_num)
        & jnp.isfinite(densities_denom)
        & (densities_num >= 0)
        & (densities_denom > eps)
    )

    safe_den = jnp.where(valid, densities_denom, 1.0)

    dab = jnp.where(
        valid,
        densities_num / safe_den,
        jnp.nan
    )

    mean_dab = jnp.nanmean(dab, axis=0)
    valid_counts = jnp.sum(valid, axis=0)
    s_idx = int(jnp.nanargmax(mean_dab))

    r = dab[:, s_idx]
    num = densities_num[:, s_idx]
    den = densities_denom[:, s_idx]

    print("Location:", states[s_idx])
    print("Mean DAB:", mean_dab[s_idx])

    print("Ratio percentiles:",
          jnp.percentile(r, jnp.array([0, 50, 90, 99, 100])))

    print("Denominator percentiles:",
          jnp.percentile(den, jnp.array([0, 1, 10, 50, 100])))

    # Which start-goal pair contributes most?
    pair_idx = int(jnp.argmax(jnp.abs(r)))

    print("Pair:", pair_idx)
    print("u:", u[pair_idx])
    print("v:", v[pair_idx])
    print("Numerator:", num[pair_idx])
    print("Denominator:", den[pair_idx])
    print("Ratio:", r[pair_idx])

    successor_fig, _ = visualise_successor_measure(
        densities_num[pair_idx], X, Y, env,
        start=u[pair_idx], goal=v[pair_idx],
        title=f"Successor measure for diagnostic pair {pair_idx}",
    )
    successor_fig.savefig("successor_measure.png", dpi=180, bbox_inches="tight")
    visualise_dab(mean_dab, X, Y, env, title="Mean DAB over sampled pairs")
    plt.show()

if __name__ == "__main__":
    main()
