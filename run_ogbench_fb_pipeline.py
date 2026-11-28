"""Train continuous-action FB on offline OGBench PointMaze and find DAB peaks.

Install ogbench alongside the project requirements. Run from the project root:
    python run_ogbench_fb_pipeline.py "@experiments/ogbench/medium_navigate.json"
"""

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.patheffects import withStroke
from matplotlib.ticker import PercentFormatter

from choose_bottlenecks import add_peak_threshold_arguments, find_local_maxima
from dab_first_hitting import compute_density_terms, evaluate_first_hitting, nanmean_no_warning
from fb_navigation.agents.forward_backward_jax import FBLearner, RadialBasisFunction2D


DATASETS = tuple(
    f"pointmaze-{maze}-{data}-v0"
    for maze, data in product(("medium", "large", "giant"), ("navigate", "stitch"))
)


class ConfigArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, line):
        option = line.strip()
        return [] if not option or option.startswith("#") else [option]


def expand_json_configs(parser, argv):
    expanded = []
    for argument in argv:
        if not (argument.startswith("@") and Path(argument[1:]).suffix.lower() == ".json"):
            expanded.append(argument)
            continue

        path = Path(argument[1:])
        try:
            config = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as error:
            parser.error(f"Cannot load {path}: {error}")
        if not isinstance(config, dict):
            parser.error(f"{path} must contain a JSON object")
        for key, value in config.items():
            if value is None and key in ("dataset_path", "checkpoint", "peak_tolerance", "peak_margin"):
                continue
            if type(value) not in (str, int, float):
                parser.error(f"{path}: {key} must be a string or number")
            expanded.append(f"--{key.replace('_', '-')}={value}")
    return expanded


def buffer_capacity(value):
    if value == "dataset":
        return value
    capacity = int(value)
    if capacity < 1:
        raise argparse.ArgumentTypeError("expected 'dataset' or a positive integer")
    return capacity


class PointMazeFB(FBLearner):
    def action_candidates(self, key):
        """OGBench bounds the two control components independently."""
        return jax.random.uniform(
            key, (self.continuous_action_samples, 2),
            minval=-self.action_limit, maxval=self.action_limit,
        )


def arguments(argv=None):
    parser = ConfigArgumentParser(
        description=__doc__, fromfile_prefix_chars="@",
        epilog="Load a preset with @FILE.json. A repeated option after the file overrides its value.",
    )
    parser.add_argument("--dataset", choices=DATASETS, default=DATASETS[0])
    parser.add_argument("--output-dir", required=True, type=Path, help="New or empty directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="JAX device, e.g. cpu or gpu:0")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset-path", type=Path, help="Local OGBench training .npz")
    source.add_argument("--checkpoint", type=Path,
                        help="Analyze a trusted FB checkpoint produced by this script")
    parser.add_argument("--dataset-dir", type=Path, default=Path("~/.ogbench/data"),
                        help="Cache directory for OGBench downloads")
    parser.add_argument("--updates", type=int, default=50_000)
    parser.add_argument("--target-update-interval", type=int, default=20,
                        help="Optimizer steps between target updates")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=buffer_capacity, default="dataset",
                        help="Replay capacity or 'dataset' to keep every transition")
    parser.add_argument("--buffer-device", default="cpu", help="Device holding offline replay")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Additional hidden layers after the first")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--polyak-coef", type=float, default=0.95,
                        help="Weight on the old target parameters")
    parser.add_argument("--reg-coef", type=float, default=1.0)
    parser.add_argument("--b-regularization", choices=("orthonormal", "touati_continuous"),
                        default="orthonormal")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature for both FB training and DAB")
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--z-sampling", choices=("replay", "cauchy"), default="cauchy",
                        help="FB task-vector sampling (default: cauchy)")
    parser.add_argument("--cauchy-scale", type=float, default=0.5,
                        help="Cauchy radius scale before multiplying by sqrt(embed_dim)")
    parser.add_argument("--zero-z-p", type=float, default=0.0, help="Probability of a zero task vector")
    parser.add_argument("--continuous-action-samples", type=int, default=32,
                        help="Uniform Box action candidates per proposal set")
    parser.add_argument("--rbf-grid", type=int, default=0,
                        help="RBF centers per axis (0 uses scaled x/y directly)")
    parser.add_argument("--rbf-sigma-scale", type=float, default=1.0,
                        help="RBF width as a multiple of center spacing along the longest axis")
    parser.add_argument("--grid-size", type=int, default=50,
                        help="DAB grid cells along the longest maze axis")
    parser.add_argument("--pairs", type=int, default=1000,
                        help="Uniform free-grid start/goal pairs")
    parser.add_argument("--dab-batch-size", type=int, default=8)
    parser.add_argument("--dab-proposal-sets", type=int, default=4,
                        help="Independent action proposal sets to average for DAB")
    parser.add_argument("--min-self-density", type=float, default=1e-5)
    parser.add_argument("--min-gap", type=float, default=1e-3)
    parser.add_argument("--probability-tolerance", type=float, default=0.05,
                        help="Tolerance around [0, 1] for diagnostic plausibility")
    parser.add_argument("--min-coverage", type=float, default=0.5,
                        help="Minimum fraction of numerically stable pairs required at a peak")
    parser.add_argument("--k", type=int, default=5,
                        help="Local maximum neighborhood radius in grid cells")
    add_peak_threshold_arguments(parser)
    args = parser.parse_args(expand_json_configs(parser, sys.argv[1:] if argv is None else argv))
    args.dataset_dir = args.dataset_dir.expanduser()
    if min(args.updates, args.target_update_interval, args.batch_size, args.embed_dim, args.hidden_dim,
           args.pairs, args.dab_batch_size, args.dab_proposal_sets, args.k) < 1:
        parser.error("Update, batch, network dimension, pair, proposal set, and neighborhood counts must be positive")
    if args.num_layers < 0 or args.continuous_action_samples < 2:
        parser.error("Layer count must be nonnegative and action candidates must number at least 2")
    if args.grid_size < 2 or args.rbf_grid < 0 or args.rbf_grid == 1:
        parser.error("Grid size must be at least 2 and RBF grid must be 0 or at least 2")
    if not (0 <= args.seed < 2**32 and 0 < args.discount < 1 and 0 <= args.min_coverage <= 1):
        parser.error("Seed must be uint32, discount in (0, 1), and minimum coverage in [0, 1]")
    if not (0 <= args.zero_z_p <= 1 and 0 <= args.polyak_coef <= 1
            and 0 <= args.adam_beta1 < 1 and 0 <= args.adam_beta2 < 1):
        parser.error("Zero-task probability and Polyak coefficient must be in [0, 1], Adam betas in [0, 1)")
    for name in ("temperature", "learning_rate", "adam_eps", "max_grad_norm", "cauchy_scale", "rbf_sigma_scale"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("reg_coef", "min_self_density", "min_gap", "probability_tolerance"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    return args


def maze_geometry(env, grid_size):
    """Get a free-space grid in native OGBench units and the FB input transform."""
    layout = env.maze_map
    cell_size = env.ij_to_xy((0, 1))[0] - env.ij_to_xy((0, 0))[0]
    floor = np.asarray([env.ij_to_xy(ij) for ij in np.argwhere(layout == 0)])
    lower = floor.min(axis=0) - cell_size / 2
    upper = floor.max(axis=0) + cell_size / 2
    origin = (upper + lower) / 2
    scale = float((upper - lower).max() / 2)
    steps = np.maximum(2, np.rint((upper - lower) * grid_size / (2 * scale)).astype(int))
    x_edges, y_edges = [np.linspace(lo, hi, count + 1) for lo, hi, count in zip(lower, upper, steps)]
    x, y = np.meshgrid((x_edges[1:] + x_edges[:-1]) / 2, (y_edges[1:] + y_edges[:-1]) / 2)
    xy = np.stack((x, y), axis=-1)

    # Exclude centers whose point body would overlap a wall, including corners.
    radius = env.model.geom("pointbody").size[0]
    free = np.ones(x.shape, dtype=bool)
    for row, col in np.argwhere(layout == 1):
        wall = env.model.geom(f"block_{row}_{col}")
        delta = np.maximum(np.abs(xy - wall.pos[:2]) - wall.size[:2], 0)
        free &= np.sum(delta**2, axis=-1) > radius**2
    return x, y, free, origin, scale, (lower, upper)


def cached_dataset_path(dataset, directory, base_url):
    path = directory / f"{dataset}.npz"
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        url = f"{base_url}/{path.name}"
        print(f"Downloading {url}", flush=True)
        # Close the download before renaming it into the cache on Windows.
        with TemporaryDirectory(dir=directory) as temporary:
            download = Path(temporary) / path.name
            urlretrieve(url, download)
            download.replace(path)
    return path


def train(args, dataset, action_limit, origin, scale, bounds, metadata):
    obs, next_obs, actions = (dataset[key] for key in ("observations", "next_observations", "actions"))
    if obs.ndim != 2 or obs.shape[1] != 2 or next_obs.shape != obs.shape or actions.shape != obs.shape:
        raise ValueError("Expected aligned PointMaze (x, y) states and two-component actions")
    if not len(obs) or not all(np.isfinite(array).all() for array in (obs, next_obs, actions)):
        raise ValueError("Expected a nonempty dataset of finite transitions")
    if np.any(np.abs(actions) > action_limit + 1e-6):
        raise ValueError("Dataset actions exceed the PointMaze action bounds")

    featuriser = None
    if args.rbf_grid:
        x_bounds, y_bounds = ((np.asarray(bounds) - origin) / scale).T.tolist()
        sigma = args.rbf_sigma_scale * max(np.ptp(x_bounds), np.ptp(y_bounds)) / (args.rbf_grid - 1)
        featuriser = RadialBasisFunction2D(x_bounds, y_bounds, args.rbf_grid, sigma, device=args.device)
    learner = PointMazeFB(
        obs_dim=2, n_actions=None, embed_dim=args.embed_dim, action_space="continuous",
        action_limit=action_limit, continuous_action_samples=args.continuous_action_samples,
        device=args.device, buffer_device=args.buffer_device,
        buffer_size=len(obs) if args.buffer_size == "dataset" else args.buffer_size,
        seed=args.seed, batch_size=args.batch_size, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        learning_rate=args.learning_rate, adam_beta1=args.adam_beta1, adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps, max_grad_norm=args.max_grad_norm, polyak_coef=args.polyak_coef,
        discount_factor=args.discount, temperature=args.temperature,
        reg_coef=args.reg_coef, b_regularization=args.b_regularization,
        z_sampling=args.z_sampling, cauchy_scale=args.cauchy_scale, zero_z_p=args.zero_z_p,
        goal_terminated=False, featuriser=featuriser,
    )
    # OGBench's loader discards the dummy transition at each episode boundary.
    # Use a continuing FB target so every density term follows the same policy.
    learner.buffer.add_batch({
        "obs": (obs - origin) / scale,
        "obs_next": (next_obs - origin) / scale,
        "action": actions,
    })
    print(f"Training FB on {learner.buffer.size:,} {args.dataset} transitions", flush=True)
    losses = np.empty(args.updates, dtype=np.float32)
    for start in range(0, args.updates, args.target_update_interval):
        end = min(start + args.target_update_interval, args.updates)
        batch_losses = np.asarray(learner.update_networks(end - start))
        if not np.isfinite(batch_losses).all():
            raise FloatingPointError(f"Nonfinite FB training loss at updates {start + 1}–{end}")
        losses[start:end] = batch_losses
        learner.soft_update_target_networks()
        if end % 1000 == 0 or end == args.updates:
            print(f"FB updates {end:,}/{args.updates:,}, loss {losses[max(0, end - 1000):end].mean():.5f}",
                  flush=True)
    learner.environment_config = metadata
    learner.save(args.output_dir / "fb_checkpoint.pkl")
    np.save(args.output_dir / "training_losses.npy", losses)
    return learner


def plot_bottlenecks(path, dataset, k, min_coverage, x, y, free, dab, coverage, peaks):
    wall_color, masked_color = "#28313b", "#d6e2ec"
    fig, (score_ax, coverage_ax) = plt.subplots(
        1, 2, figsize=(12, 5.7), sharex=True, sharey=True, layout="constrained",
    )
    for ax in (score_ax, coverage_ax):
        ax.set(aspect="equal", xlabel="OGBench x")
        ax.pcolormesh(x, y, free, cmap=ListedColormap([wall_color, masked_color]),
                      shading="nearest", vmin=0, vmax=1)
    score_ax.set(ylabel="OGBench y", title=f"DAB and ranked peaks (k={k})")
    coverage_ax.set_title("Stable start–goal pairs")

    visible = free & (coverage >= min_coverage)
    score = score_ax.pcolormesh(x, y, np.where(visible, dab, np.nan), cmap="viridis",
                                shading="nearest", vmin=0, vmax=1)
    stable = coverage_ax.pcolormesh(x, y, np.where(free, coverage, np.nan), cmap="cividis",
                                   shading="nearest", vmin=0, vmax=1)
    fig.colorbar(score, ax=score_ax, label="Mean clipped first-hit estimate", shrink=0.83)
    fig.colorbar(stable, ax=coverage_ax, label="Fraction of pairs", shrink=0.83,
                 format=PercentFormatter(xmax=1))
    score_ax.legend(handles=[
        Patch(color=wall_color, label="Wall or clearance"),
        Patch(color=masked_color, label=f"Floor with no DAB shown (coverage < {min_coverage:.0%})"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=1, frameon=False)

    for rank, (row, col) in enumerate(peaks, start=1):
        px, py = x[row, col], y[row, col]
        score_ax.scatter(px, py, s=42, facecolors="none", edgecolors="white", linewidths=2.2, zorder=3)
        score_ax.scatter(px, py, s=42, facecolors="none", edgecolors="#c11f39", linewidths=1.0, zorder=4)
        score_ax.annotate(str(rank), (px, py), xytext=(5, 5), textcoords="offset points",
                          fontsize=9, fontweight="bold", zorder=5,
                          path_effects=[withStroke(linewidth=3, foreground="white")])
    fig.suptitle(dataset)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(args, learner, x, y, free, origin, scale):
    xy = np.column_stack((x[free], y[free]))
    if len(xy) < 2:
        raise ValueError("Fewer than two free grid points. Increase --grid-size.")
    states = jax.device_put(((xy - origin) / scale).astype(np.float32), learner.device)
    rng = np.random.default_rng(args.seed)
    starts = rng.integers(len(xy), size=args.pairs)
    goals = rng.integers(len(xy) - 1, size=args.pairs)
    goals += goals >= starts
    u, v = states[starts], states[goals]
    densities = compute_density_terms(
        learner, states, learner.embed_B(states), u, v, learner.embed_B(v),
        "softmax", learner.temperature, args.dab_batch_size, seed=args.seed,
        proposal_sets=args.dab_proposal_sets,
    )
    hits = evaluate_first_hitting(
        densities, min_self_density=args.min_self_density, min_gap=args.min_gap,
        tolerance=args.probability_tolerance,
    )
    dab = np.full(x.shape, np.nan)
    dab[free] = nanmean_no_warning(np.clip(hits["stable"], 0, 1))
    coverage = np.full(x.shape, np.nan)
    coverage[free] = hits["stable_mask"].mean(axis=0)
    plausible_coverage = np.full(x.shape, np.nan)
    plausible_coverage[free] = hits["plausible_mask"].mean(axis=0)
    eligible = free & (coverage >= args.min_coverage)
    peaks = find_local_maxima(
        dab, eligible, k=args.k, peak_tolerance=args.peak_tolerance, peak_margin=args.peak_margin,
    )
    peaks.sort(key=lambda rc: dab[rc], reverse=True)

    output = args.output_dir
    np.savez_compressed(
        output / "first_hitting_results.npz",
        X=x, Y=y, valid_mask=free.ravel(), mean_clipped=dab[free],
        coverage=coverage, plausible_coverage=plausible_coverage,
        starts=xy[starts], goals=xy[goals], maxima=np.asarray(peaks, dtype=int).reshape(-1, 2),
    )
    with (output / f"bottlenecks_k{args.k}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("rank", "row", "col", "x", "y", "dab", "stable_coverage", "plausible_coverage"))
        for rank, (row, col) in enumerate(peaks, start=1):
            index = row, col
            writer.writerow((rank, row, col, x[index], y[index], dab[index],
                             coverage[index], plausible_coverage[index]))
    plot_bottlenecks(output / f"bottlenecks_k{args.k}.png", args.dataset, args.k, args.min_coverage,
                     x, y, free, dab, coverage, peaks)
    print(f"Found {len(peaks)} peaks at k={args.k}. Mean stable pair coverage: {coverage[free].mean():.1%}",
          flush=True)


def main():
    args = arguments()
    try:
        import ogbench
        from ogbench.utils import DATASET_URL, load_dataset
    except ModuleNotFoundError as error:
        if error.name != "ogbench":
            raise
        raise SystemExit("Install OGBench in the project environment with: python -m pip install ogbench") from error

    env = ogbench.make_env_and_datasets(args.dataset, env_only=True, render_mode=None)
    try:
        x, y, free, origin, scale, bounds = maze_geometry(env.unwrapped, args.grid_size)
        low, high = env.action_space.low, env.action_space.high
        if high.shape != (2,) or not np.all(high == high[0]) or not np.all(low == -high):
            raise ValueError("Expected a symmetric two-dimensional PointMaze action Box")
        action_limit = float(high[0])
    finally:
        env.close()

    metadata = {
        "dataset": args.dataset, "observation_origin": origin.tolist(), "observation_scale": scale,
        "action_proposals": "uniform_box",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise SystemExit(f"Output directory must be empty: {args.output_dir}")
    if args.checkpoint:
        learner = PointMazeFB.from_checkpoint(args.checkpoint, device=args.device)
        if learner.environment_config != metadata or not learner.continuous or learner.action_limit != action_limit:
            raise ValueError("Checkpoint must come from this PointMaze pipeline and match --dataset")
    else:
        path = args.dataset_path or cached_dataset_path(args.dataset, args.dataset_dir, DATASET_URL)
        dataset = load_dataset(str(path))
        learner = train(args, dataset, action_limit, origin, scale, bounds, metadata)
    settings = {
        "arguments": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "environment": metadata, "learner": learner.config,
        "features": "rbf" if isinstance(learner.featuriser, RadialBasisFunction2D) else "scaled_xy",
    }
    (args.output_dir / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    with jax.default_device(learner.device):
        analyze(args, learner, x, y, free, origin, scale)
    print(f"Saved results in {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
