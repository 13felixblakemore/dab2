"""Plot successor densities, goal-conditioned values, or DAB from a Four Rooms checkpoint."""
import argparse
import json
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dab_first_hitting import build_valid_grid, compute_density_terms
from fb_navigation.agents.forward_backward_jax import FBLearner
from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms
from fb_navigation.environments.continuous_four_rooms import visualize_environment
from fb_navigation.environments.continuous_four_rooms_jax import WallBlock


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


def environment(checkpoint, seed, doorway_scale):
    run = checkpoint.parent
    config = run / "config.json"
    options = json.loads(config.read_text()).get("environment", {}) if config.is_file() else {}
    options.setdefault("include_stay_action", False)
    if doorway_scale is not None:
        options["doorway_scale"] = doorway_scale
    env = AdjustableFourRooms(seed=seed, **options)
    metadata = run / "metadata.json"
    walls = json.loads(metadata.read_text()).get("wall_blocks") if metadata.is_file() else None
    if walls is not None:
        env._wall_blocks = [WallBlock(*wall) for wall in walls]
    return env


def cross_room_pairs(env):
    # Use the recorded dividers so archived, asymmetric layouts are sampled correctly.
    vertical, horizontal = env.wall_blocks[0], env.wall_blocks[3]
    starts, goals = np.asarray(env.sample_points(1000)), np.asarray(env.sample_points(1000))

    def rooms(points):
        x, y = points.T
        inside = ((x < vertical.xmin) | (x > vertical.xmax)) & (
            (y < horizontal.ymin) | (y > horizontal.ymax))
        return np.where(inside, (x > vertical.xmax) + 2 * (y > horizontal.ymax), -1)

    first, second = rooms(starts), rooms(goals)
    difference = first ^ second
    groups = [np.flatnonzero((first >= 0) & (second >= 0) & (difference == kind))[:2]
              for kind in (1, 2, 3)]
    if any(len(group) < 2 for group in groups):
        raise ValueError("Could not sample two pairs per room category. Try another --seed.")
    indices = np.concatenate(groups)
    return starts[indices], goals[indices]


def successor_maps(learner, states, starts, goals, policy):
    tasks = learner.embed_B(goals)
    forward = learner.embed_F(starts, tasks)
    scores = jnp.einsum("nda,nd->na", forward, tasks)
    if policy == "greedy":
        weights = jax.nn.one_hot(jnp.argmax(scores, axis=-1), scores.shape[-1])
    else:
        weights = jax.nn.softmax(scores / learner.temperature, axis=-1)
    return np.asarray(jnp.einsum("nda,na->nd", forward, weights) @ learner.embed_B(states).T)


def value_maps(learner, states, goals):
    for task in learner.embed_B(goals):
        chunks = []
        for offset in range(0, len(states), 4096):
            batch = states[offset:offset + 4096]
            forward = learner.embed_F(batch, jnp.broadcast_to(task, (len(batch), len(task))))
            chunks.append(np.asarray(jnp.max(jnp.einsum("nda,d->na", forward, task), axis=-1)))
        yield np.concatenate(chunks)


def save_heatmaps(path, env, grid, values, goals, starts=None):
    x, y, valid, _ = grid
    columns = min(3, len(goals))
    rows = (len(goals) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.5 * rows),
                             squeeze=False, layout="constrained")
    for index, (ax, goal, data) in enumerate(zip(axes.flat, goals, values)):
        panel = np.full(x.size, np.nan)
        panel[valid] = data
        mesh = ax.pcolormesh(x, y, panel.reshape(x.shape), shading="nearest",
                             cmap="viridis" if starts is not None else "magma",
                             vmin=values.min(), vmax=values.max())
        visualize_environment(env, ax, draw_grid=False, draw_initial_position=False, draw_goal=False)
        ax.scatter(*goal, marker="*", s=150, color="red", edgecolors="white", zorder=6)
        if starts is not None:
            ax.scatter(*starts[index], s=65, color="white", edgecolors="black", zorder=5)
        ax.set(xlabel="x", ylabel="y", title=f"Goal ({goal[0]:.2f}, {goal[1]:.2f})",
               xlim=(0, env.size), ylim=(0, env.size))
    for ax in list(axes.flat)[len(goals):]:
        ax.set_visible(False)
    label = "Successor density" if starts is not None else "Greedy goal value"
    fig.colorbar(mesh, ax=list(axes.flat)[:len(goals)], label=label)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_checkpoint_dab(args):
    run = args.checkpoint.parent
    config = json.loads((run/'config.json').read_text())
    recorded = json.loads((run/'metadata.json').read_text())
    env = AdjustableFourRooms(**config['environment'], seed=args.seed)
    block_type = type(env.wall_blocks[0])
    env._wall_blocks = [block_type(*box) for box in recorded['wall_blocks']]
    learner = FBLearner.from_checkpoint(str(args.checkpoint))
    assert learner.temperature == config['learner']['temperature']
    X, Y, mask, states = build_valid_grid(env, 50)
    u, v = env.sample_points(args.pairs), env.sample_points(args.pairs)
    terms = compute_density_terms(learner, states, learner.embed_B(states), u, v,
                                  learner.embed_B(v), 'softmax', learner.temperature, 4)
    m_us, _, _, _, m_ss, _ = terms
    assert m_us.shape == m_ss.shape == (args.pairs, int(mask.sum()))
    with np.errstate(divide='ignore', invalid='ignore'):
        raw = m_us/m_ss
        raw_mean = np.mean(raw, axis=0)
    # Keep the raw arithmetic mean unfiltered; mark numerically undefined states.
    undefined = ~np.isfinite(raw).all(axis=0)
    raw_mean[undefined] = np.nan
    accepted = np.isfinite(raw) & (m_us >= 0) & (m_ss > 50) & (raw <= 2)
    counts = accepted.sum(axis=0)
    filtered_mean = np.divide(np.where(accepted, raw, 0).sum(axis=0), counts,
                              out=np.full(counts.shape, np.nan), where=counts>0)
    # First adjacent-room sample, with rooms inferred from the recorded wall slabs.
    walls = np.array(recorded['wall_blocks'])
    vertical = walls[(walls[:,3]-walls[:,1])>(walls[:,2]-walls[:,0])]
    horizontal = walls[(walls[:,2]-walls[:,0])>(walls[:,3]-walls[:,1])]
    vx = Counter(map(tuple, vertical[:,[0,2]])).most_common(1)[0][0]
    hy = Counter(map(tuple, horizontal[:,[1,3]])).most_common(1)[0][0]
    def rooms(points):
        x, y = np.asarray(points).T
        inside = ((x<vx[0])|(x>vx[1])) & ((y<hy[0])|(y>hy[1]))
        return np.where(inside, (x>vx[1]).astype(int)+2*(y>hy[1]).astype(int), -1)
    room_u, room_v = rooms(u), rooms(v)
    delta = room_u ^ room_v
    examples = np.flatnonzero((room_u>=0)&(room_v>=0)&((delta==1)|(delta==2)))
    if not len(examples):
        raise ValueError('No adjacent-room example; increase --pairs')
    pair = int(examples[0])
    output = Path('results/dab')/run.name/args.checkpoint.stem
    output.mkdir(parents=True, exist_ok=True)
    def grid(values):
        full = np.full(X.size, np.nan)
        full[mask] = values
        return full.reshape(X.shape)
    density_grid = grid(m_us[pair])
    fig, _ = visualise_successor_measure(density_grid, X, Y, env, start=u[pair], goal=v[pair],
                  title=f'Successor density — adjacent-room pair {pair}\nEmbedding {learner.embed_dim}; softmax T={learner.temperature:g}')
    fig.savefig(output/'successor_density_example.png', dpi=170)
    plt.close(fig)
    images = [('dab_raw_mean', raw_mean, 'Original DAB: raw arithmetic mean (no filtering)'),
              ('dab_filtered_mean', filtered_mean, 'Original DAB: filtered mean'),
              ('successor_density_mean', m_us.mean(axis=0), 'Successor density: mean over sampled tasks')]
    for name, values, title in images:
        fig, _ = visualise_dab(grid(values), X, Y, env,
                    title=f'{title}\n{args.pairs} pairs; embedding {learner.embed_dim}; softmax T={learner.temperature:g}',
                    colorbar_label='Successor density' if name.startswith('successor') else 'Mean successor-density ratio')
        fig.savefig(output/(name+'.png'), dpi=170)
        plt.close(fig)
    summary = dict(checkpoint=str(args.checkpoint), wall_blocks=recorded['wall_blocks'],
                   seed=args.seed, n_pairs=args.pairs, grid_size=50, embed_dim=learner.embed_dim,
                   hidden_dim=config['learner']['hidden_dim'], policy='softmax', temperature=float(learner.temperature),
                   formulation='original successor-density ratio m_us/m_ss under pi_v',
                   sampling='independent uniform free-space endpoints, including same-room tasks',
                   example_pair=pair, example_selection='first adjacent-room sampled pair; no density-based selection',
                   nonfinite_values=int((~np.isfinite(raw)).sum()), undefined_raw_mean_states=int(undefined.sum()),
                   filtered_definition='finite ratio AND numerator>=0 AND denominator>50 AND ratio<=2; mean over retained samples',
                   retained_fraction=float(accepted.mean()),
                   raw_mean_range=[float(np.nanmin(raw_mean)),float(np.nanmax(raw_mean))],
                   filtered_mean_range=[float(np.nanmin(filtered_mean)),float(np.nanmax(filtered_mean))])
    (output/'plot_metadata.json').write_text(json.dumps(summary, indent=2))
    np.savez_compressed(output/'plot_data.npz', X=X,Y=Y,valid_mask=mask,starts=np.asarray(u),goals=np.asarray(v),
                        raw_mean=raw_mean,filtered_mean=filtered_mean,retained_counts=counts,
                        mean_successor_density=m_us.mean(axis=0),example_density=m_us[pair],example_pair=pair)
    print(json.dumps(summary, indent=2))
    print(f'Saved four plots to {output.resolve()}')


def plot_dab_goals(results, parser):
    with np.load(results) as data:
        x, y, valid = data["X"], data["Y"], data["valid_mask"]
        if "first_hit_clipped" not in data.files:
            parser.error("The results must contain per-pair first_hit_clipped values")
        hits = data["first_hit_clipped"]
        goals, groups = np.unique(data["v"], axis=0, return_inverse=True)
    if len(goals) < 5:
        parser.error("The results must contain at least five distinct goals")
    if hits.shape[1] == valid.size:
        hits = hits[:, valid]
    elif hits.shape[1] != np.count_nonzero(valid):
        parser.error("The first_hit_clipped values do not match the saved grid")

    # Choose saved goals near the four room centers and the lower middle doorway.
    anchors = [(0.25, 0.75), (0.75, 0.75), (0.25, 0.25), (0.75, 0.25), (0.50, 0.25)]
    available = np.ones(len(goals), dtype=bool)
    examples = []
    for anchor in anchors:
        distances = np.sum((goals - anchor) ** 2, axis=1)
        idx = np.argmin(np.where(available, distances, np.inf))
        available[idx] = False
        selected = hits[groups == idx]
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
    output = results.with_name("first_hit_five_goals.png")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    successor = commands.add_parser("successor", help="Plot six cross-room examples")
    value = commands.add_parser("value", help="Plot value surfaces for fixed goals")
    for command, resolution in ((successor, 80), (value, 200)):
        command.add_argument("--checkpoint", required=True, type=Path)
        command.add_argument("--resolution", type=int, default=resolution)
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--doorway-scale", type=float, help="Fallback when walls were not recorded")
    successor.add_argument("--policy", choices=("greedy", "softmax"), default="greedy")
    successor.add_argument("--output", type=Path, help="Output directory")
    value.add_argument("--goal", nargs=2, type=float, action="append", metavar=("X", "Y"))
    value.add_argument("--output", type=Path, default=Path("value_function.png"), help="Output image")
    dab = commands.add_parser("dab", help="Plot original DAB ratios and successor densities")
    dab.add_argument('--checkpoint', type=Path, default=Path('experiments/fourrooms_doorways/runs/symmetric_width1_replay_continuing_orthonormal_embed150_seed0/epoch_0800.pkl'))
    dab.add_argument('--pairs', type=int, default=1000)
    dab.add_argument('--seed', type=int, default=4)
    dab_goals = commands.add_parser("dab-goals", help="Plot five goals from saved first-hitting results")
    dab_goals.add_argument("results", type=Path, help="Path to first_hitting_results.npz")
    args = parser.parse_args()
    if args.command == "dab-goals":
        plot_dab_goals(args.results, dab_goals)
        return
    if args.command == "dab":
        plot_checkpoint_dab(args)
        return
    if args.resolution < 2:
        parser.error("--resolution must be at least 2")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")

    learner = FBLearner.from_checkpoint(str(args.checkpoint))
    if learner.continuous:
        parser.error("These plots require a discrete-action checkpoint")
    env = environment(args.checkpoint, args.seed, args.doorway_scale)
    grid = build_valid_grid(env, args.resolution)
    states = grid[3]
    if args.command == "successor":
        starts, goals = cross_room_pairs(env)
        values = successor_maps(learner, states, starts, goals, args.policy)
        output = args.output or Path("successor_examples") / args.checkpoint.parent.name
        path = output / "overview.png"
        save_heatmaps(path, env, grid, values, goals, starts)
        for index in range(len(goals)):
            panel = slice(index, index + 1)
            save_heatmaps(output / f"example_{index + 1}.png", env, grid,
                          values[panel], goals[panel], starts[panel])
        np.savez(output / "examples.npz", starts=starts, goals=goals, densities=values, states=states)
        (output / "metadata.json").write_text(json.dumps({
            "checkpoint": str(args.checkpoint), "policy": args.policy, "seed": args.seed,
            "grid_size": args.resolution,
            "wall_blocks": [[b.xmin, b.ymin, b.xmax, b.ymax] for b in env.wall_blocks],
        }, indent=2))
    else:
        goals = np.asarray(args.goal or [[.8, .8], [.2, .2]])
        for x, y in goals:
            if not (0 < x < env.size and 0 < y < env.size) or any(
                    wall.xmin <= x <= wall.xmax and wall.ymin <= y <= wall.ymax for wall in env.wall_blocks):
                parser.error(f"Goal ({x}, {y}) must lie in free space inside the environment")
        path = args.output
        save_heatmaps(path, env, grid, np.stack(list(value_maps(learner, states, goals))), goals)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
