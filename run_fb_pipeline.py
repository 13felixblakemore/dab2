"""Train Four Rooms FB, find DAB peaks, and evaluate an option for each peak."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

from choose_bottlenecks import add_peak_threshold_arguments, find_local_maxima, plot_maxima
from dab_first_hitting import fill_grid
from evaluate_options import add_option_policy_arguments, make_rollout, option_policy_settings
from fb_navigation.agents.forward_backward_jax import FBLearner
from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms
from fb_navigation.environments.continuous_four_rooms_jax import ContinuousWorld


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--doorway_scale", type=float)
    parser.add_argument("--resume", action="store_true", help="Resume training in the same output directory")
    parser.add_argument("--dab_pairs", type=int, default=1000)
    parser.add_argument("--dab_pair_sampling", choices=("uniform", "mesh"), default="uniform")
    parser.add_argument("--dab_pair_mesh_size", type=int, default=21)
    parser.add_argument("--grid_size", type=int, default=50)
    parser.add_argument("--dab_batch_size", type=int, default=8)
    parser.add_argument("--policy", choices=("greedy", "softmax"), default="greedy",
                        help="Policy used to compute DAB")
    parser.add_argument("--k", type=int, default=1, help="Local maximum neighborhood parameter")
    parser.add_argument("--bottleneck_size", type=int, default=1,
                        help="Side length of the square averaged at each candidate")
    add_peak_threshold_arguments(parser)
    parser.add_argument("--num_samples", type=int, default=1000, help="Start states shared by all options")
    parser.add_argument("--step_limit", type=int, default=30)
    parser.add_argument("--trials_per_start", type=int, default=1)
    parser.add_argument("--threshold", type=float, help="Option goal tolerance (default: environment tolerance)")
    parser.add_argument("--epsilon", type=float, default=0.02)
    add_option_policy_arguments(parser)
    args = parser.parse_args()
    if min(args.dab_pairs, args.dab_batch_size, args.num_samples, args.trials_per_start, args.k) < 1:
        parser.error("Pair, batch, sample, trial, and neighborhood counts must be positive")
    if args.dab_pair_sampling == "mesh" and args.dab_pair_mesh_size < 2:
        parser.error("dab_pair_mesh_size must be at least 2")
    if args.grid_size < 2 or not 1 <= args.bottleneck_size <= args.grid_size:
        parser.error("The grid must be at least 2 by 2 and contain the bottleneck square")
    if args.step_limit < 1 or not 0 <= args.epsilon <= 1:
        parser.error("step_limit must be positive and epsilon must be in [0, 1]")
    if args.threshold is not None and (not np.isfinite(args.threshold) or args.threshold <= 0):
        parser.error("threshold must be finite and positive")
    return args


def bottlenecks(dab_path, env, k, size, *, peak_tolerance=None, peak_margin=None):
    with np.load(dab_path) as data:
        X, Y, valid_mask = data["X"], data["Y"], data["valid_mask"]
        dab = fill_grid(data["mean_clipped"], valid_mask).reshape(X.shape)
    dab = np.where(valid_mask.reshape(X.shape), dab, np.nan)
    maxima = find_local_maxima(dab, valid_mask, k=k, bottleneck_size=size,
                               peak_tolerance=peak_tolerance, peak_margin=peak_margin)
    peaks = []
    for row, col in maxima:
        region = np.s_[row:row + size, col:col + size]
        x, y = X[region].mean(), Y[region].mean()
        if any(w.xmin <= x <= w.xmax and w.ymin <= y <= w.ymax for w in env.wall_blocks):
            continue
        peaks.append((dab[region].mean(), row, col, x, y))
    peaks.sort(key=lambda peak: peak[0], reverse=True)
    scores, rows, cols, xs, ys = np.asarray(peaks, dtype=float).reshape(-1, 5).T
    return dab, X, Y, np.column_stack((rows, cols)).astype(int), np.column_stack((xs, ys)), scores


def test_options(args, learner, env, goals, scores, output_dir):
    threshold = env.threshold_distance if args.threshold is None else args.threshold
    policy, temperature = option_policy_settings(args, learner)
    with jax.default_device(learner.device):
        starts = env.sample_points(args.num_samples)
        targets = jnp.asarray(goals, dtype=starts.dtype)
        key = jax.random.PRNGKey(args.seed)
        run = make_rollout(learner, env, args.step_limit, threshold,
                           args.trials_per_start, epsilon=args.epsilon,
                           policy=policy, temperature=temperature)
        successes = np.empty((len(goals), len(starts), args.trials_per_start), dtype=bool)
        for i in range(len(goals)):
            successes[i] = np.asarray(run(learner.state["params"], starts, targets[i:i + 1],
                                          jax.random.fold_in(key, i))[0])

    starts = np.asarray(starts)
    # A start inside the goal tolerance is already successful at step zero.
    outside = np.any(np.abs(starts[None] - np.asarray(targets)[:, None]) > threshold, axis=-1)
    counts = outside.sum(axis=1)
    rates = np.divide((successes & outside[:, :, None]).sum(axis=(1, 2)),
                      counts * args.trials_per_start, out=np.full(len(goals), np.nan), where=counts > 0)
    np.savez_compressed(output_dir / "option_success.npz", starts=starts, goals=goals,
                        successes=successes, outside_goal=outside, success_rate_by_option=rates,
                        epsilon=args.epsilon, threshold=threshold, step_limit=args.step_limit,
                        option_policy=policy, option_temperature=np.nan if temperature is None else temperature)
    with (output_dir / "option_success.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["option", "x", "y", "dab", "starts_outside_goal", "success_rate"])
        for i, ((x, y), score, count, rate) in enumerate(zip(goals, scores, counts, rates), start=1):
            writer.writerow([i, x, y, score, count, rate if count else ""])
            result = f"{rate:.1%}" if count else "no starts outside goal tolerance"
            print(f"Option {i} ({x:.4f}, {y:.4f}): {result}", flush=True)


def main(args):
    root = Path(__file__).resolve().parent
    output_dir = args.output_dir.resolve()
    training_dir = output_dir / "training"
    dab_dir = output_dir / "dab"
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    scale = args.doorway_scale
    if scale is None:
        scale = config["environment"].get("doorway_scale", 1.0)

    print("Training FB", flush=True)
    train_command = [
        sys.executable, "-m", "fb_navigation.training.run_fourrooms_doorways",
        "--config", str(config_path), "--output", str(training_dir),
        "--device", args.device, "--seed", str(args.seed), "--doorway-scale", str(scale),
        "--checkpoint-every", str(config.get("checkpoint_every", 200)),
    ]
    if args.resume:
        train_command.append("--resume")
    subprocess.run(train_command, cwd=root, check=True)
    checkpoint = training_dir / f"epoch_{config['epochs']:04d}.pkl"

    print("Computing DAB", flush=True)
    subprocess.run([
        sys.executable, str(root / "dab_first_hitting.py"),
        "--checkpoint", str(checkpoint), "--output-dir", str(dab_dir),
        "--device", args.device, "--seed", str(args.seed), "--doorway-scale", str(scale),
        "--pairs", str(args.dab_pairs), "--grid-size", str(args.grid_size),
        "--pair-sampling", args.dab_pair_sampling, "--pair-mesh-size", str(args.dab_pair_mesh_size),
        "--batch-size", str(args.dab_batch_size), "--policy", args.policy,
        "--temperature", str(config["learner"].get("temperature", 200.0)),
    ], cwd=root, check=True)

    learner = FBLearner.from_checkpoint(checkpoint, device=args.device)
    if learner.continuous:
        env = ContinuousWorld.from_movement_config(learner.environment_config, seed=args.seed + 4)
    else:
        trained_config = json.loads((training_dir / "config.json").read_text())
        env = AdjustableFourRooms(**trained_config["environment"], seed=args.seed + 4)

    print("Finding bottlenecks", flush=True)
    dab, X, Y, maxima, goals, scores = bottlenecks(
        dab_dir / "first_hitting_results.npz", env, args.k, args.bottleneck_size,
        peak_tolerance=args.peak_tolerance, peak_margin=args.peak_margin,
    )
    np.savez_compressed(output_dir / "bottlenecks.npz", maxima=maxima, goals=goals, dab_scores=scores,
                        bottleneck_size=args.bottleneck_size)
    plot_maxima(dab, maxima, env, output_dir / "local_maxima.png", X, Y,
                bottleneck_size=args.bottleneck_size)
    print(f"Testing {len(goals)} bottleneck options", flush=True)
    test_options(args, learner, env, goals, scores, output_dir)
    print("Saved results in", output_dir)


if __name__ == "__main__":
    main(arguments())
