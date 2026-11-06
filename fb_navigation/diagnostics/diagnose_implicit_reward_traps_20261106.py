"""Frozen-checkpoint continuing returns for learned rewards at recorded wall traps."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import argparse
import csv
import hashlib
import json
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms
from fb_navigation.environments.continuous_four_rooms_jax import Point, _move
from fb_navigation.agents.forward_backward_jax import FBLearner
from fb_navigation.diagnostics.investigate_doorway_decline import nominal_transitions

ACTIONS = ('LEFT', 'RIGHT', 'UP', 'DOWN')
EXPERIMENT = ROOT / 'experiments/fourrooms_doorways'


def reward_bounds(learner, tasks):
    """Conservative global interval bounds: normalized RBF inputs form a simplex."""
    layers = jax.device_get(learner.state['params']['b'])
    lower = layers[0]['weight'].min(0) + layers[0]['bias']
    upper = layers[0]['weight'].max(0) + layers[0]['bias']
    lower, upper = np.maximum(lower, 0), np.maximum(upper, 0)
    for layer in layers[1:-1]:
        w, b = layer['weight'].astype(np.float64), layer['bias']
        pos, neg = np.maximum(w, 0), np.minimum(w, 0)
        lower, upper = np.maximum(lower @ pos + upper @ neg + b, 0), np.maximum(upper @ pos + lower @ neg + b, 0)
    w = layers[-1]['weight'].astype(np.float64) @ tasks.T
    b = layers[-1]['bias'].astype(np.float64) @ tasks.T
    pos, neg = np.maximum(w, 0), np.minimum(w, 0)
    return np.stack([lower @ pos + upper @ neg + b, upper @ pos + lower @ neg + b], axis=1)


def make_rollout(learner, env, horizons, policy, trace=False):
    """Reward at s_t, forced a_0, then frozen policy; never stop/reset on a goal."""
    horizons = jnp.asarray(horizons, jnp.int32)
    horizon = int(np.max(horizons))
    actions = jnp.asarray([(a.x, a.y) for a in env.actions], jnp.float32)
    walls = tuple(env.wall_blocks)
    gamma = learner.discount_factor

    @jax.jit
    def rollout(params, starts, goals, first, keys, corrected_tasks):
        tasks = learner.backward_network.apply(params['b'], learner.featuriser.transform(goals))
        n = len(starts)
        def step(carry, t):
            position, keys, total, compensation, saved, collision_count, first_blocked, hits, max_reward = carry
            features = learner.featuriser.transform(position)
            b = learner.backward_network.apply(params['b'], features)
            reward = jnp.stack([jnp.sum(b*tasks, axis=1), jnp.sum(b*corrected_tasks, axis=1)], axis=1)
            # Compensated float32 summation without changing native movement RNG precision.
            addition = gamma**t * reward - compensation
            new_total = total + addition
            compensation = (new_total-total) - addition
            total = new_total
            saved = jnp.where((t+1 == horizons)[None, :, None], total[:, None, :], saved)
            max_reward = jnp.maximum(max_reward, jnp.abs(reward))
            splits = jax.vmap(lambda k: jax.random.split(k, 4))(keys)
            keys, _, ak, mk = [splits[:, i] for i in range(4)]
            f = learner.forward_network.apply(params['f'], features, tasks)
            q = jnp.einsum('bda,bd->ba', f, tasks)
            if policy == 'softmax200':
                selected = jax.vmap(lambda k, logits: jax.random.categorical(k, logits))(ak, q/learner.temperature)
            elif policy == 'wall_masked':
                def mask(p):
                    return jax.vmap(lambda a: _move(Point(*p), Point(*a), jax.random.PRNGKey(0),
                        0., env.size, walls)[2])(actions)
                blocked = jax.vmap(mask)(position)
                scores = jnp.where(blocked, -jnp.inf, q)
                selected = jnp.argmax(jnp.where(jnp.all(blocked, axis=1)[:, None], q, scores), axis=1)
            else:
                selected = jnp.argmax(q, axis=1)
            selected = jnp.where(t == 0, first, selected)
            def move(p, a, key):
                nxt, _, blocked = _move(Point(*p), Point(*actions[a]), key, env._noise, env.size, walls)
                return jnp.asarray([nxt.x, nxt.y]), blocked
            following, blocked = jax.vmap(move)(position, selected, mk)
            first_blocked = jnp.where(t == 0, blocked, first_blocked)
            collision_count += blocked.astype(jnp.int32)
            reached = jnp.all(abs(following-goals) <= env.threshold_distance, axis=1)
            hits = jnp.where((hits > horizon) & reached, t+1, hits)
            carry = (following, keys, total, compensation, saved, collision_count, first_blocked, hits, max_reward)
            return carry, dict(position=position, following=following, reward=reward, action=selected, blocked=blocked) if trace else None
        zeros = jnp.zeros((n, 2), jnp.float32)
        carry = (starts, keys, zeros, zeros, jnp.zeros((n, len(horizons), 2), jnp.float32),
            jnp.zeros(n, jnp.int32), jnp.zeros(n, bool), jnp.full(n, horizon+1, jnp.int32), zeros)
        final, history = jax.lax.scan(step, carry, jnp.arange(horizon))
        result = dict(returns=final[4], wall_actions=final[5], first_blocked=final[6], hits=final[7], max_abs_reward=final[8])
        if trace: result['trace'] = history
        return result
    return rollout


def select_probes(traces, metrics, pairs, first, count, env):
    selected, used = [], set()
    order = [first] + sorted(range(len(metrics)), key=lambda i: abs(metrics[i]['longest_stationary_collision_run']-80))
    for i in order:
        row = metrics[i]
        if row['success'] or row['pair'] in used or row['longest_same_action_collision_run'] < 30:
            continue
        state = np.asarray(row['anchor'], np.float32)
        matches = np.all(traces['position'][:, i] == state, axis=1) & traces['blocked'][:, i]
        if not matches.any(): continue
        time = int(np.flatnonzero(matches)[0])
        q = traces['q'][time, i]
        _, blocked = nominal_transitions(env, state[None])
        if not blocked[0, q.argmax()]: continue
        selected.append(dict(trial=i, pair=row['pair'], replica=row['replica'],
            state=state.tolist(), goal=pairs['goals'][row['pair']].tolist(), recorded_step=time+1,
            recorded_q=q.tolist(), recorded_collisions=row['collisions'],
            recorded_longest_stationary_run=row['longest_stationary_collision_run']))
        used.add(row['pair'])
        if len(selected) == count: break
    if len(selected) != count: raise ValueError('Not enough distinct recorded blocked-greedy traps')
    return selected


def world(ax, env):
    for wall in env.wall_blocks:
        ax.add_patch(Rectangle((wall.xmin, wall.ymin), wall.xmax-wall.xmin, wall.ymax-wall.ymin,
            facecolor='#343b43', zorder=3))
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal', xlabel='x', ylabel='y')


def maps(learner, env, probes, tasks, output):
    axis = np.linspace(0, 1, 201, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis); points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    free = np.ones(len(points), bool)
    for w in env.wall_blocks:
        free &= ~((points[:, 0] >= w.xmin) & (points[:, 0] <= w.xmax) & (points[:, 1] >= w.ymin) & (points[:, 1] <= w.ymax))
    b = np.concatenate([np.asarray(learner.embed_B(p)) for p in np.array_split(points[free], 40)])
    values = np.full((len(points), len(probes)), np.nan); values[free] = b @ tasks.T
    np.savez_compressed(output/'reward_maps.npz', x=axis, y=axis, rewards=values.reshape(201, 201, -1), free=free.reshape(201, 201))
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for i, (ax, probe) in enumerate(zip(axes.flat, probes)):
        im = ax.imshow(values[:, i].reshape(201, 201), origin='lower', extent=(0, 1, 0, 1), cmap='viridis')
        world(ax, env); ax.scatter(*probe['state'], color='#ee4242', marker='x', s=65, lw=2, zorder=5)
        ax.scatter(*probe['goal'], color='#ffbc35', marker='*', edgecolors='black', s=100, zorder=5)
        ax.set_title(f'Trap {i+1}: pair {probe["pair"]}\nr(trap)={probe["reward_at_trap"]:.3f}, r(goal)={probe["reward_at_goal"]:.3f}', fontsize=10)
        fig.colorbar(im, ax=ax, fraction=.046, label='B(s) dot B(goal)')
    fig.suptitle('Frozen learned reward: red cross = trap, gold star = goal; walls masked')
    fig.tight_layout(); fig.savefig(output/'reward_landscapes.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    surrounding = []
    for i, (ax, probe) in enumerate(zip(axes.flat, probes)):
        state = np.array(probe['state'])
        im = ax.imshow(values[:, i].reshape(201, 201), origin='lower', extent=(0, 1, 0, 1), cmap='viridis')
        world(ax, env); ax.scatter(*state, color='#ee4242', marker='x', s=65, lw=2, zorder=5)
        ax.set(xlim=(max(0, state[0]-.125), min(1, state[0]+.125)),
               ylim=(max(0, state[1]-.125), min(1, state[1]+.125)))
        ax.set_title(f'Trap {i+1}: FB chooses {probe["fb_action"]}', fontsize=11)
        fig.colorbar(im, ax=ax, fraction=.046, label='B(s) dot B(goal)')
        offsets = np.array([-.1, -.05, -.025, 0, .025, .05, .1])
        dx, dy = np.meshgrid(offsets, offsets)
        local = state + np.stack([dx.ravel(), dy.ravel()], axis=1)
        valid = np.all((local >= 0) & (local <= 1), axis=1)
        for w in env.wall_blocks:
            valid &= ~((local[:, 0] >= w.xmin) & (local[:, 0] <= w.xmax) & (local[:, 1] >= w.ymin) & (local[:, 1] <= w.ymax))
        local = local[valid].astype(np.float32)
        rewards = np.asarray(learner.embed_B(local)) @ tasks[i]
        for p, r in zip(local, rewards): surrounding.append(dict(trap=i+1, x=float(p[0]), y=float(p[1]), reward=float(r)))
    fig.suptitle('Local learned-reward neighborhoods (approximately +/- 0.125 around each trap)')
    fig.tight_layout(); fig.savefig(output/'local_reward_neighborhoods.png', dpi=170); plt.close(fig)
    with (output/'surrounding_rewards.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['trap', 'x', 'y', 'reward']); writer.writeheader(); writer.writerows(surrounding)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=EXPERIMENT/'runs/width2_replay_continuing_threshold025_seed0')
    parser.add_argument('--traces', type=Path, default=EXPERIMENT/'analyses/replay_threshold025_failures')
    parser.add_argument('--evaluation', type=Path, default=EXPERIMENT/'analyses/replay_threshold025_comparison')
    parser.add_argument('--output', type=Path, default=EXPERIMENT/'analyses/implicit_reward_wall_traps')
    parser.add_argument('--rollouts', type=int, default=256)
    parser.add_argument('--horizon', type=int, default=1500)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=63018)
    parser.add_argument('--policies', nargs='+', choices=['greedy', 'softmax200', 'wall_masked'], default=['greedy', 'softmax200', 'wall_masked'])
    args = parser.parse_args()
    if args.rollouts < 2 or args.horizon < 2 or args.batch_size < 1: parser.error('Invalid sampling counts')
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.run/'config.json').read_text())
    env = AdjustableFourRooms(**config['environment'])
    learner = FBLearner.from_checkpoint(args.run/'epoch_0800.pkl', device=args.device)
    if learner.goal_terminated: raise ValueError('Requires continuing FB')
    digest = hashlib.sha256((args.run/'epoch_0800.pkl').read_bytes()).hexdigest()
    traces = dict(np.load(args.traces/'traces.npz'))
    assert str(traces['checkpoint_sha256']) == digest
    metrics = json.loads((args.traces/'metrics_150.json').read_text())
    pairs = dict(np.load(args.evaluation/'pairs.npz'))
    first = json.loads((args.traces/'summary.json').read_text())['recorded_example']['index']
    probes = select_probes(traces, metrics, pairs, first, 6, env)
    states = np.array([p['state'] for p in probes], np.float32)
    goals = np.array([p['goal'] for p in probes], np.float32)
    tasks = np.asarray(learner.embed_B(goals))
    b_states = np.asarray(learner.embed_B(states))
    q = np.einsum('bda,bd->ba', np.asarray(learner.embed_F(states, tasks)), tasks)
    _, nominal_blocked = nominal_transitions(env, states)
    for i, p in enumerate(probes):
        p.update(reward_at_trap=float(b_states[i] @ tasks[i]), reward_at_goal=float(tasks[i] @ tasks[i]),
            q=q[i].tolist(), fb_action=ACTIONS[q[i].argmax()], nominal_blocked=nominal_blocked[i].tolist())
        np.testing.assert_allclose(q[i], p['recorded_q'], rtol=2e-5, atol=.02)
        assert nominal_blocked[i, q[i].argmax()]
    horizons = sorted({h for h in (150, 500, 1000, args.horizon) if h <= args.horizon})
    settings = dict(checkpoint_sha256=digest, run=str(args.run.resolve()), rollouts=args.rollouts,
        horizons=horizons, gamma=learner.discount_factor, temperature=learner.temperature,
        seed=args.seed, policies=args.policies, device=args.device, jax=jax.__version__,
        reward='online B(s_t) dot online B(goal), no normalization, current-state timing',
        continuing=True, stop_at_goal=False, seed_shared_across_initial_actions=True,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), probes=probes)
    manifest = args.output/'config.json'
    if manifest.exists() and json.loads(manifest.read_text()) != settings: raise ValueError('Use a fresh output directory')
    manifest.write_text(json.dumps(settings, indent=2))

    # Covariance audit: direct Q comparison to raw reward assumes E_rho[BB^T] = I.
    with (args.run/'resume.pkl').open('rb') as stream: recovery = pickle.load(stream)
    storage, size = recovery['carry'][1], int(recovery['carry'][3])
    idx = np.random.default_rng(args.seed).integers(size, size=32768)
    reference = np.asarray(storage['obs'])[idx]
    b_reference = np.concatenate([np.asarray(learner.embed_B(p)) for p in np.array_split(reference, 32)]).astype(np.float64)
    covariance = b_reference.T @ b_reference / len(b_reference)
    eig = np.linalg.eigvalsh(covariance)
    corrected_tasks = (np.linalg.pinv(covariance, rcond=1e-6, hermitian=True) @ tasks.T).T.astype(np.float32)
    bounds_raw = reward_bounds(learner, tasks); bounds_corrected = reward_bounds(learner, corrected_tasks)
    covariance_stats = dict(samples=len(reference), source='uniform sample of final replay observations', rcond=1e-6,
        eigenvalues=eig.tolist(), mean_eigenvalue=float(eig.mean()),
        relative_frobenius_distance_from_identity=float(np.linalg.norm(covariance-np.eye(len(covariance)))/np.sqrt(len(covariance))),
        raw_reward_bounds=bounds_raw.tolist(), corrected_reward_bounds=bounds_corrected.tolist(),
        raw_tail_bound=(np.max(abs(bounds_raw), axis=1)*learner.discount_factor**args.horizon/(1-learner.discount_factor)).tolist(),
        corrected_tail_bound=(np.max(abs(bounds_corrected), axis=1)*learner.discount_factor**args.horizon/(1-learner.discount_factor)).tolist())
    (args.output/'covariance_audit.json').write_text(json.dumps(covariance_stats, indent=2))
    np.savez_compressed(args.output/'covariance.npz', covariance=covariance, corrected_tasks=corrected_tasks, reference=reference)
    maps(learner, env, probes, tasks, args.output)
    del traces, recovery, storage
    pids = np.repeat(np.arange(len(probes)), 4*args.rollouts)
    first_actions = np.tile(np.repeat(np.arange(4), args.rollouts), len(probes))
    replica = np.tile(np.arange(args.rollouts), len(probes)*4)
    key_ids = pids*args.rollouts + replica
    for policy in args.policies:
        cache = args.output/f'{policy}.npz'
        if cache.exists(): continue
        fn = make_rollout(learner, env, horizons, policy)
        batches = []
        for begin in range(0, len(pids), args.batch_size):
            sl = slice(begin, begin+args.batch_size); ids = pids[sl]
            keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(args.seed), i))(jnp.asarray(key_ids[sl], jnp.uint32))
            result = fn(learner.state['params'], jnp.asarray(states[ids]), jnp.asarray(goals[ids]),
                jnp.asarray(first_actions[sl]), keys, jnp.asarray(corrected_tasks[ids]))
            batches.append(jax.device_get(result))
            print(json.dumps(dict(policy=policy, completed=min(begin+args.batch_size, len(pids)), total=len(pids))), flush=True)
        record = {k: np.concatenate([b[k] for b in batches]).reshape((len(probes), 4, args.rollouts)+batches[0][k].shape[1:]) for k in batches[0]}
        assert all(np.isfinite(v).all() for v in record.values())
        np.savez_compressed(cache, **record)
    print('Monte Carlo recordings complete.', flush=True)


if __name__ == '__main__': main()
