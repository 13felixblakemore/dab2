"""Post-hoc goal-before-bottleneck metric from an existing FB checkpoint.

For each independently sampled pair (u, v), and each candidate state s,
compute a discounted first-hit-before-goal estimate using a single policy pi_v:

    H(u, s; v) = (P_us - P_uv * P_vs) / (1 - P_sv * P_vs)

where P_xy = m^{pi_v}(x, y) / m^{pi_v}(y, y).

WARNING: This hitting-probability identity is exact for a discrete Markov
chain with a suitable successor-measure convention. Its use with pointwise
continuous-state FB densities is an approximation that requires validation.

Run from the project root, e.g.:
    python dab_first_hitting.py --checkpoint experiments/.../epoch_0800.pkl \
        --doorway-scale 2.0 --pairs 1000 --batch-size 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a trained FB checkpoint")
    parser.add_argument("--doorway-scale", type=float, default=2.0,
                        help="Must match the geometry used in training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pairs", type=int, default=1000)
    parser.add_argument("--grid-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of goals per batch; reduce for limited VRAM")
    parser.add_argument("--policy", choices=("greedy", "softmax"), default="greedy",
                        help="Must match the FB TD target's action-selection policy")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Used only with --policy softmax")
    parser.add_argument("--min-self-density", type=float, default=1e-5,
                        help="Diagnostic floor for m(s,s) and m(v,v)")
    parser.add_argument("--min-gap", type=float, default=1e-3,
                        help="Reject 1-P_sv*P_vs <= this value")
    parser.add_argument("--probability-tolerance", type=float, default=0.05,
                        help="Tolerance around [0,1] for diagnostic plausibility filtering")
    parser.add_argument("--output-dir", default="results/first_hitting")
    return parser.parse_args()


def build_valid_grid(env, grid_size):
    coords = np.linspace(0.0, float(env.size), grid_size)
    X, Y = np.meshgrid(coords, coords)
    wall_mask = np.zeros(X.shape, dtype=bool)
    for block in env.wall_blocks:
        wall_mask |= (
            (X >= block.xmin) & (X <= block.xmax)
            & (Y >= block.ymin) & (Y <= block.ymax)
        )
    valid_mask = ~wall_mask.ravel()
    all_states = np.stack([X.ravel(), Y.ravel()], axis=-1)
    return X, Y, valid_mask, jnp.asarray(all_states[valid_mask])


def compute_density_terms(learner, states, b_states, u, v, z, policy, temperature, batch_size):
    """Return all six density terms for the same policy pi_{z_i}.

    u, v: (G, 2); z: (G, D); states: (S, 2); b_states: (S, D)
    Returns: m_us (G,S), m_vs (G,S), m_uv (G,), m_vv (G,),
             m_ss (G,S), m_sv (G,S).
    """
    count_states, latent_dim = b_states.shape

    def chosen_forward(obs, tasks):
        f = learner.embed_F(obs, tasks)  # (N, D, A)
        q = jnp.einsum("nda,nd->na", f, tasks)  # (N, A)
        if policy == "greedy":
            actions = jnp.argmax(q, axis=-1)
            return jnp.take_along_axis(
                f, actions[:, None, None], axis=-1
            ).squeeze(-1)  # (N, D)
        weights = jax.nn.softmax(q / temperature, axis=-1)
        return jnp.einsum("nda,na->nd", f, weights)

    @jax.jit
    def compute_chunk(u_chunk, v_chunk, z_chunk):
        # Starting at u and v, both following policy pi_v.
        f_u = chosen_forward(u_chunk, z_chunk)  # (G, D)
        f_v = chosen_forward(v_chunk, z_chunk)  # (G, D)

        # Starting at every candidate s for each of the G goal policies.
        f_s = jax.vmap(
            lambda zi: chosen_forward(
                states, jnp.broadcast_to(zi, (count_states, latent_dim))
            )
        )(z_chunk)  # (G, S, D)

        # Each *destination* B(y) is evaluated under the same pi_v.
        m_us = f_u @ b_states.T  # (G, S)
        m_vs = f_v @ b_states.T  # (G, S)
        m_uv = jnp.einsum("gd,gd->g", f_u, z_chunk)  # (G,)
        m_vv = jnp.einsum("gd,gd->g", f_v, z_chunk)  # (G,)
        m_ss = jnp.einsum("gsd,sd->gs", f_s, b_states)  # (G, S)
        m_sv = jnp.einsum("gsd,gd->gs", f_s, z_chunk)  # (G, S)
        return m_us, m_vs, m_uv, m_vv, m_ss, m_sv

    chunks = [[] for _ in range(6)]
    for start in range(0, len(u), batch_size):
        end = min(start + batch_size, len(u))
        values = compute_chunk(u[start:end], v[start:end], z[start:end])
        for outputs, value in zip(chunks, values):
            # Float64 is useful for the cancellation in the post-hoc formula.
            outputs.append(np.asarray(value, dtype=np.float64))
        if (start // batch_size + 1) % 20 == 0 or end == len(u):
            print(f"Computed {end}/{len(u)} goal policies", flush=True)
    return tuple(np.concatenate(arrays, axis=0) for arrays in chunks)


def evaluate_first_hitting(densities, min_self_density, min_gap, tolerance):
    m_us, m_vs, m_uv, m_vv, m_ss, m_sv = densities
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        P_us = m_us / m_ss
        P_vs = m_vs / m_ss
        P_uv = (m_uv / m_vv)[:, None]
        P_sv = m_sv / m_vv[:, None]
        gap = 1.0 - P_sv * P_vs
        first_hit = (P_us - P_uv * P_vs) / gap

    # Only reject numeric singularities here; do not change sign or clip values.
    stable = (
        np.isfinite(first_hit)
        & np.isfinite(P_us) & np.isfinite(P_uv)
        & np.isfinite(P_vs) & np.isfinite(P_sv)
        & (m_ss > min_self_density)
        & (m_vv[:, None] > min_self_density)
        & (gap > min_gap)
    )
    stable_hit = np.where(stable, first_hit, np.nan)

    # More restrictive diagnostic, NOT an unbiased estimator of beta(s).
    def plausible(p):
        return (p >= -tolerance) & (p <= 1.0 + tolerance)

    plausible_mask = (
        stable & plausible(P_us) & plausible(P_uv)
        & plausible(P_vs) & plausible(P_sv) & plausible(first_hit)
    )
    # Clip only small tolerated deviations, and only for this diagnostic plot.
    plausible_hit = np.where(plausible_mask, np.clip(first_hit, 0.0, 1.0), np.nan)

    return {
        "raw": first_hit,
        "stable": stable_hit,
        "plausible": plausible_hit,
        "stable_mask": stable,
        "plausible_mask": plausible_mask,
        "gap": gap,
        "P_us": P_us, "P_uv": P_uv, "P_vs": P_vs, "P_sv": P_sv,
    }


def nanmean_no_warning(values):
    counts = np.sum(np.isfinite(values), axis=0)
    means = np.nansum(values, axis=0) / np.maximum(counts, 1)
    return np.where(counts > 0, means, np.nan)


def save_heatmap(values, X, Y, valid_mask, env, path, title, colorbar_label):
    from fb_navigation.environments.continuous_four_rooms import visualize_environment

    full_grid = np.full(X.size, np.nan)
    full_grid[valid_mask] = np.asarray(values)
    masked_grid = np.ma.masked_invalid(full_grid.reshape(X.shape))
    fig, ax = plt.subplots(figsize=(7, 6))
    mesh = ax.pcolormesh(X, Y, masked_grid, shading="nearest", cmap="viridis")
    visualize_environment(
        env, ax, draw_grid=False,
        draw_initial_position=False, draw_goal=False
    )
    ax.set(xlim=(0, env.size), ylim=(0, env.size), xlabel="x", ylabel="y", title=title)
    ax.set_aspect("equal")
    fig.colorbar(mesh, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved", path)


def summarise(values, name):
    finite = values[np.isfinite(values)]
    if not finite.size:
        print(f"{name}: no finite values")
        return
    print(f"{name} percentiles [0,1,10,50,90,99,100]:",
          np.percentile(finite, [0, 1, 10, 50, 90, 99, 100]))


def main():
    args = arguments()
    if args.batch_size < 1 or args.pairs < 1 or args.grid_size < 2:
        raise ValueError("batch-size and pairs must be >=1; grid-size must be >=2")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")

    from fb_navigation.agents.forward_backward_jax import FBLearner
    from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = AdjustableFourRooms(
        doorway_scale=args.doorway_scale,
        include_stay_action=False,
        seed=args.seed + 3,
    )
    learner = FBLearner.from_checkpoint(args.checkpoint)

    X, Y, valid_mask, states = build_valid_grid(env, args.grid_size)
    b_states = learner.embed_B(states)
    u = jnp.asarray(env.sample_points(args.pairs))
    v = jnp.asarray(env.sample_points(args.pairs))
    z = learner.embed_B(v)  # Must use same z preprocessing as training.

    print("Grid states:", states.shape, "goals:", z.shape)
    print("Policy:", args.policy, "temperature:", args.temperature)
    densities = compute_density_terms(
        learner, states, b_states, u, v, z,
        args.policy, args.temperature, args.batch_size
    )
    results = evaluate_first_hitting(
        densities, args.min_self_density,
        args.min_gap, args.probability_tolerance
    )

    m_us, m_vs, m_uv, m_vv, m_ss, m_sv = densities
    print("Negative m_us fraction:", np.mean(m_us < 0))
    print("Negative m_ss fraction:", np.mean(m_ss < 0))
    print("Negative m_vv fraction:", np.mean(m_vv < 0))
    summarise(m_ss, "m_ss")
    summarise(m_vv, "m_vv")
    summarise(results["gap"], "1 - P_sv*P_vs")
    summarise(results["raw"], "Raw first-hit estimate")
    print("Stable ratio fraction:", np.mean(results["stable_mask"]))
    print("Plausible ratio fraction:", np.mean(results["plausible_mask"]))

    stable_mean = nanmean_no_warning(results["stable"])
    plausible_mean = nanmean_no_warning(results["plausible"])
    coverage = np.mean(results["plausible_mask"], axis=0)
    for scores, filename, title, cb_label in [
        (stable_mean, "first_hit_stable_mean.png", "Post-hoc first-hit: numerically stable mean", "Discounted first-hit estimate"),
        (plausible_mean, "first_hit_plausible_mean.png", "Post-hoc first-hit: plausible-only diagnostic", "Filtered first-hit estimate"),
        (coverage, "first_hit_coverage.png", "Fraction of pairs passing plausibility checks", "Valid fraction"),
        (nanmean_no_warning(m_us), "mean_successor_density.png", "Mean FB successor density from u", "Mean successor density"),
    ]:
        save_heatmap(scores, X, Y, valid_mask, env, output_dir / filename, title, cb_label)

    np.savez_compressed(
        output_dir / "first_hitting_results.npz",
        X=X, Y=Y, valid_mask=valid_mask,
        states=np.asarray(states), u=np.asarray(u), v=np.asarray(v),
        stable_mean=stable_mean, plausible_mean=plausible_mean,
        coverage=coverage,
        # Pairwise arrays facilitate diagnostic analysis and convergence tests.
        first_hit_stable=results["stable"],
        first_hit_plausible=results["plausible"],
        P_us=results["P_us"], P_uv=results["P_uv"],
        P_vs=results["P_vs"], P_sv=results["P_sv"],
    )
    print("Saved", output_dir / "first_hitting_results.npz")


if __name__ == "__main__":
    main()
