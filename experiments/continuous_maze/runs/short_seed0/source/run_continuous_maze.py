"""Isolated, resumable Continuous Maze experiment. See experiments/continuous_maze/README.md."""
import argparse
import csv
import hashlib
import json
import pickle
import platform
import random
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from continuous_maze import ContinuousMaze, plot_maze
from forward_backward_jax import FBLearner, RadialBasisFunction2D, _cauchy_ball

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / 'experiments/continuous_maze/fb50.json'


def make_learner(config, seed, device='cpu'):
    return FBLearner(**config['learner'], seed=seed, device=device, buffer_device='cpu',
                     featuriser=RadialBasisFunction2D(**config['rbf'], device=device))


def evaluate(learner, goals, seed, epsilon=.02, steps=30):
    """Independent, paired diagnostic trials. Check success after each action, as released."""
    trial_rng = np.random.RandomState(seed)
    successes, lengths, collisions, distances, paths = [], [], [], [], []
    for _ in range(goals):
        trial_seed = int(trial_rng.randint(0, 2**31-1))
        env, actions = ContinuousMaze(trial_seed), random.Random(trial_seed)
        obs = env.reset()
        goal = env.goal.copy()
        z = learner.embed_B(goal[None])
        path, hits = [obs.tolist()], 0
        for t in range(1, steps+1):
            action = (actions.randrange(5) if actions.random() < epsilon
                      else int(learner.act(obs[None], z).item()))
            obs, reward, _, info = env.step(action)
            hits += int(info['goes_through_wall'])
            path.append(obs.tolist())
            if reward:
                break
        successes.append(int(reward))
        if reward:
            lengths.append(t)
        collisions.append(hits)
        distances.append(float(np.max(np.abs(obs-goal))))
        paths.append(dict(path=path, goal=goal.tolist(), success=bool(reward), collisions=hits))
    probabilities = np.array([1-epsilon+epsilon/5] + [epsilon/5]*4)
    positive = probabilities[probabilities > 0]
    return dict(success_rate=float(np.mean(successes)), successes=sum(successes), goals=goals,
                mean_successful_path_length=float(np.mean(lengths)) if lengths else None,
                wall_collisions=sum(collisions), mean_final_distance=float(np.mean(distances)),
                behavior_entropy=float(-np.sum(positive*np.log(positive)))), paths


def plot_outputs(out, rows, results, paths, seed):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r['epoch'] for r in rows], [r['success_rate'] for r in rows], label='FB-50, epsilon=.02')
    axes[0].set(xlabel='Epoch', ylabel='Success rate', ylim=(0, 1))
    axes[0].legend()
    axes[1].bar(list(results), [r['success_rate'] for r in results.values()])
    axes[1].set(ylabel='Success rate', ylim=(0, 1), title=f'Seed {seed}; final evaluation')
    fig.tight_layout()
    fig.savefig(out / 'success.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 6))
    rng = np.random.RandomState(seed)
    plot_maze(ax, rng.uniform(0, 1, (1000, 2)), paths[:8])
    fig.tight_layout()
    fig.savefig(out / 'trajectories.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--epochs', type=int, help='Explicit truncated run; does not change updates per epoch')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.epochs is not None:
        config['epochs'] = args.epochs
    out = args.output or ROOT / f'experiments/continuous_maze/runs/seed{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'config.json'
    if manifest.exists() and not args.resume:
        raise FileExistsError(f'{out} already has a run. Use --resume or another output directory.')
    if args.resume and json.loads(manifest.read_text()) != config:
        raise ValueError('Resume configuration must match the original run')
    manifest.write_text(json.dumps(config, indent=2))
    learner = make_learner(config, args.seed, args.device)
    env, action_rng = ContinuousMaze(args.seed), random.Random(args.seed)
    start_epoch, rows = 0, []
    if args.resume:
        with (out / 'resume.pkl').open('rb') as stream:
            saved = pickle.load(stream)
        start_epoch, rows = saved['epoch'], saved['rows']
        learner = FBLearner.from_checkpoint(out / saved['checkpoint'], device=args.device)
        learner.buffer.storage = saved['replay']
        learner.buffer.ptr, learner.buffer.size = saved['ptr'], saved['size']
        learner.buffer._rng.bit_generator.state = saved['replay_rng']
        env.rng.set_state(saved['env_rng'])
        action_rng.setstate(saved['action_rng'])
    sources = ['forward_backward_jax.py', 'continuous_maze.py', 'run_continuous_maze.py']
    metadata = dict(seed=args.seed, python=platform.python_version(), jax=jax.__version__,
                    numpy=np.__version__, devices=[str(d) for d in jax.devices()],
                    schedule=dict(updates=config['epochs']*config['cycles']*config['updates'],
                                  transitions=config['epochs']*config['cycles']*config['episodes']*config['steps']),
                    sha256={s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources})
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    diagnostics = jax.jit(lambda p, target, batch, other, w: learner._loss(
        p, target, batch, other, w, jnp.zeros(config['learner']['batch_size'], bool), return_components=True))
    for epoch in range(start_epoch, config['epochs']):
        started = time.perf_counter()
        losses, collection_collisions = [], 0
        for cycle in range(config['cycles']):
            batch = dict(obs=[], action=[], obs_next=[], next_actions=[])
            for _ in range(config['episodes']):
                obs = env.reset()
                # Released code draws a behavior task even under uniform exploration.
                learner.sample_cauchy_ball(1)
                for _ in range(config['steps']):
                    if config['collection_epsilon'] != 1:
                        raise ValueError('Reference reproduction requires uniform collection')
                    action_rng.random()
                    action = action_rng.randrange(5)
                    obs_next, _, _, info = env.step(action)
                    collection_collisions += int(info['goes_through_wall'])
                    for name, value in zip(batch, (obs, action, obs_next, [True]*5)):
                        batch[name].append(value)
                    obs = obs_next
            learner.buffer.add_batch(batch)
            losses.extend(np.asarray(learner.update_networks(config['updates'])).tolist())
            learner.soft_update_target_networks()
        # Diagnostic sampling never advances learner/replay RNG streams.
        rng = np.random.RandomState(100000+epoch)
        idx = rng.randint(learner.buffer.size, size=learner.batch_size)
        other_idx = rng.randint(learner.buffer.size, size=learner.batch_size)
        batch = {k: jnp.asarray(v[idx]) for k, v in learner.buffer.storage.items()}
        other = dict(obs=jnp.asarray(learner.buffer.obs[other_idx]))
        w = _cauchy_ball(jax.random.PRNGKey(100000+epoch), learner.batch_size, learner.embed_dim)
        components = {k: float(v) for k, v in diagnostics(learner.state['params'], learner.state['target'], batch, other, w).items()}
        if not np.all(np.isfinite(losses)) or not all(np.isfinite(v) for v in components.values()):
            raise FloatingPointError('Nonfinite training diagnostics')
        score, _ = evaluate(learner, config['evaluation_goals'], 200000+args.seed*1000+epoch,
                            config['evaluation_epsilon'], config['steps'])
        row = dict(epoch=epoch+1, updates=(epoch+1)*config['cycles']*config['updates'],
                   transitions=(epoch+1)*config['cycles']*config['episodes']*config['steps'],
                   seconds=time.perf_counter()-started, loss_mean=float(np.mean(losses)),
                   collection_wall_collisions=collection_collisions, **score, **components)
        rows.append(row)
        with (out / 'training.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        checkpoint = f'epoch_{epoch+1:04d}.pkl'
        learner.save(out / checkpoint)
        saved = dict(epoch=epoch+1, rows=rows, checkpoint=checkpoint,
                     replay=learner.buffer.storage, ptr=learner.buffer.ptr, size=learner.buffer.size,
                     replay_rng=learner.buffer._rng.bit_generator.state,
                     env_rng=env.rng.get_state(), action_rng=action_rng.getstate())
        with (out / 'resume.tmp').open('wb') as stream:
            pickle.dump(saved, stream, protocol=pickle.HIGHEST_PROTOCOL)
        (out / 'resume.tmp').replace(out / 'resume.pkl')
        print(json.dumps(row), flush=True)
    results, all_paths = {}, {}
    for name, epsilon in [('paper_epsilon_002', config['evaluation_epsilon']), ('greedy_diagnostic', 0), ('random_diagnostic', 1)]:
        results[name], all_paths[name] = evaluate(learner, config['final_goals'], 300000+args.seed,
                                                epsilon, config['steps'])
    (out / 'final.json').write_text(json.dumps(results, indent=2))
    (out / 'trajectories.json').write_text(json.dumps(all_paths))
    plot_outputs(out, rows, results, all_paths['paper_epsilon_002'], args.seed)
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
