"""Read-only checkpoint diagnostics on a fixed evaluation suite and held-out data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fb_navigation.environments.continuous_maze import ContinuousMaze
from fb_navigation.agents.forward_backward_jax import FBLearner, _cauchy_ball
from fb_navigation.training.run_continuous_maze import evaluate


def wilson_interval(successes, trials):
    z, p = 1.95996398454, successes/trials
    denominator = 1+z*z/trials
    center = (p+z*z/(2*trials))/denominator
    radius = z*np.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/denominator
    return center-radius, center+radius


def heldout_transitions(seed, count):
    """Uniform random behavior, uniform resets every 30 steps; never read live replay."""
    env, rng = ContinuousMaze(seed), np.random.RandomState(seed+1)
    obs, actions, next_obs = [], [], []
    for i in range(count):
        if i % 30 == 0:
            current = env.reset()
        action = int(rng.randint(5))
        following, _, _, _ = env.step(action)
        obs.append(current)
        actions.append(action)
        next_obs.append(following)
        current = following
    return dict(obs=np.asarray(obs, np.float32), action=np.asarray(actions, np.int32),
                obs_next=np.asarray(next_obs, np.float32))


def bootstrap_residual(learner, batch, other_obs, tasks):
    """Sampled FB bootstrap error; excludes the singular immediate occupancy term."""
    p, target = learner.state['params'], learner.state['target']
    feature = learner.featuriser.transform
    forward, backward = learner.forward_network.apply, learner.backward_network.apply
    next_f = forward(target['f'], feature(batch['obs_next']), tasks)
    pi = jax.nn.softmax(jnp.einsum('bda,bd->ba', next_f, tasks)/learner.temperature, axis=-1)
    expected_f = jnp.einsum('bda,ba->bd', next_f, pi)
    target_value = learner.discount_factor*(expected_f @ backward(target['b'], feature(other_obs)).T)
    all_f = forward(p['f'], feature(batch['obs']), tasks)
    selected_f = jnp.take_along_axis(all_f, batch['action'][:, None, None], axis=-1).squeeze(-1)
    value = selected_f @ backward(p['b'], feature(other_obs)).T
    error = value-target_value
    return dict(residual_mean=jnp.mean(error), residual_mae=jnp.mean(jnp.abs(error)),
                residual_mse=jnp.mean(error**2), target_mse=jnp.mean(target_value**2),
                backup_entropy=-jnp.mean(jnp.sum(jax.scipy.special.xlogy(pi, pi), axis=-1)))


def residual_metrics(learner, probes, seed, batches, batch_size=128):
    rng = np.random.RandomState(seed)
    calculate = jax.jit(lambda b, other, w: bootstrap_residual(learner, b, other, w))
    values = []
    for index in range(batches):
        first = rng.randint(len(probes['obs']), size=batch_size)
        second = rng.randint(len(probes['obs']), size=batch_size)
        batch = {k: jnp.asarray(v[first]) for k, v in probes.items()}
        tasks = _cauchy_ball(jax.random.PRNGKey(seed+index), batch_size, learner.embed_dim)
        values.append({k: float(v) for k, v in calculate(batch, jnp.asarray(probes['obs'][second]), tasks).items()})
    result = {k: float(np.mean([v[k] for v in values])) for k in values[0]}
    result['residual_rmse'] = float(np.sqrt(result.pop('residual_mse')))
    result['relative_residual_rmse'] = result['residual_rmse']/max(np.sqrt(result.pop('target_mse')), 1e-12)
    return result


def trajectory_metrics(learner, paths):
    gaps, relative_gaps = [], []
    states = np.asarray([s for p in paths for s in p['path'][:-1]], np.float32)
    goals = np.asarray([p['goal'] for p in paths for _ in p['path'][:-1]], np.float32)
    for start in range(0, len(states), 256):
        z = learner.embed_B(goals[start:start+256])
        f = learner.embed_F(states[start:start+256], z)
        q = np.asarray(jnp.einsum('bda,bd->ba', f, z))
        top = np.sort(q, axis=-1)
        gap = top[:, -1]-top[:, -2]
        gaps.extend(gap.tolist())
        relative_gaps.extend((gap/np.maximum(np.max(q, axis=1)-np.min(q, axis=1), 1e-12)).tolist())
    steps = sum(len(p['path'])-1 for p in paths)
    return dict(evaluation_actions=steps,
                wall_action_percent=100*sum(p['collisions'] for p in paths)/steps,
                mean_capped_steps=float(np.mean([len(p['path'])-1 for p in paths])),
                mean_action_value_gap=float(np.mean(gaps)),
                median_action_value_gap=float(np.median(gaps)),
                mean_relative_action_value_gap=float(np.mean(relative_gaps)),
                greedy_policy_entropy=0.0)


def plot_results(rows, output):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    epochs = [r['epoch'] for r in rows]
    success = np.array([r['success_rate'] for r in rows])
    bounds = np.array([[r['success_ci_low'], r['success_ci_high']] for r in rows])
    axes.flat[0].errorbar(epochs, success, yerr=np.maximum(0, np.array([success-bounds[:, 0], bounds[:, 1]-success])), fmt='o-')
    axes.flat[0].axhline(.83, color='gray', linestyle=':', label='Figure 9 FB-50 ~.83 (visual estimate)')
    axes.flat[0].set(ylim=(0, 1), ylabel='Success rate (95% Wilson CI)')
    axes.flat[0].legend(fontsize=7)
    for ax, field, label in zip(list(axes.flat)[1:],
            ['mean_successful_path_length', 'wall_action_percent', 'residual_rmse', 'mean_action_value_gap', 'relative_residual_rmse'],
            ['Successful path length', 'Actions rejected by walls (%)', 'Sampled FB bootstrap RMSE', 'Mean top-two Q gap', 'Relative bootstrap RMSE']):
        ax.plot(epochs, [np.nan if r[field] is None else r[field] for r in rows], 'o-')
        ax.set(ylabel=label)
    for ax in axes.flat:
        ax.set(xlabel='Checkpoint epoch')
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output/'checkpoint_diagnostics.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('experiments/continuous_maze/runs/seed0'))
    parser.add_argument('--output', type=Path, default=Path('experiments/continuous_maze/diagnostics/seed0'))
    parser.add_argument('--epochs', nargs='+', type=int, default=[100, 200, 400, 600, 800])
    parser.add_argument('--goals', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=918273)
    parser.add_argument('--residual-batches', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.goals < 1 or args.residual_batches < 1 or any(e < 1 for e in args.epochs):
        parser.error('Goals, residual batches and checkpoint epochs must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(run=str(args.run.resolve()), goals=args.goals, seed=args.seed,
                  residual_batches=args.residual_batches, device=args.device,
                  evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  sources_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                  for p in ['continuous_maze.py', 'forward_backward_jax.py', 'run_continuous_maze.py']})
    manifest = args.output/'config.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError('Diagnostic configuration changed: choose a new output directory')
    manifest.write_text(json.dumps(config, indent=2))
    probes = heldout_transitions(args.seed+10, 4096)
    if not (args.output/'heldout_transitions.npz').exists():
        np.savez_compressed(args.output/'heldout_transitions.npz', **probes)
    rows, missing = [], []
    for epoch in sorted(set(args.epochs)):
        checkpoint = args.run/f'epoch_{epoch:04d}.pkl'
        if not checkpoint.exists():
            missing.append(epoch)
            continue
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        result_file = args.output/f'epoch_{epoch:04d}.json'
        if result_file.exists():
            result = json.loads(result_file.read_text())
            if result['checkpoint_sha256'] != digest:
                raise ValueError('Checkpoint content changed after evaluation')
        else:
            learner = FBLearner.from_checkpoint(checkpoint, device=args.device)
            expected_updates = epoch*25*40
            if int(learner.state['optim'][0]) != expected_updates:
                raise ValueError(f'Unexpected optimizer step count for checkpoint {epoch}')
            score, paths = evaluate(learner, args.goals, args.seed, epsilon=.02, steps=30)
            low, high = wilson_interval(score['successes'], args.goals)
            result = dict(epoch=epoch, checkpoint_sha256=digest, optimizer_updates=expected_updates,
                          success_ci_low=low, success_ci_high=high, **score,
                          **trajectory_metrics(learner, paths),
                          **residual_metrics(learner, probes, args.seed+20, args.residual_batches))
            if any(not np.isfinite(v) for v in result.values() if isinstance(v, (float, int))):
                raise FloatingPointError('Nonfinite diagnostic')
            temporary = result_file.with_suffix('.tmp')
            temporary.write_text(json.dumps(result, indent=2))
            temporary.replace(result_file)
            (args.output/f'epoch_{epoch:04d}_trajectories.json').write_text(json.dumps(paths))
        rows.append(result)
        print(json.dumps(result), flush=True)
    if rows:
        with (args.output/'checkpoints.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plot_results(rows, args.output)
    status = dict(evaluated=[r['epoch'] for r in rows], pending=missing)
    (args.output/'status.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
