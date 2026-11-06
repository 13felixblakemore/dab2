"""Paired wall-trap diagnostic for an existing FB checkpoint; no training.

Reuses pairs, seeds, tolerance and horizon from evaluate_original_goals_jax.py.
The diagnostic mask rejects nominal (noiseless) segments intersecting walls.
It does not anticipate movement noise or exclude boundary-clipped actions.
Run: python diagnose_fb_walls.py --output evaluation_original_walls
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from continuous_four_rooms_jax import ContinuousFourRooms, Point, _move
from forward_backward_jax import FBLearner, ReplayBuffer
from run_fb_jax import make_collector


def nominal_action_mask(positions, actions, size, walls):
    """Same clipped-segment intersection predicate as _move, without noise."""
    def at_state(p):
        def allowed(delta):
            end = jnp.clip(p + delta, 0., size)
            collision = jnp.any(jnp.asarray([
                wall.intersects_segment(Point(*p), Point(*end)) for wall in walls
            ], dtype=bool))
            return ~collision
        return jax.vmap(allowed)(actions)
    return jax.vmap(at_state)(positions)


def make_trace(learner, env, horizon, wall_aware):
    actions = jnp.asarray([(a.x, a.y) for a in env.actions])
    walls = tuple(env.wall_blocks)
    availability = jnp.zeros(learner.n_actions, bool).at[jnp.array(env.get_available_actions(None))].set(True)

    @jax.jit
    def trace(params, starts, goals, tasks, keys):
        def step(carry, t):
            position, hits, keys = carry
            split = jax.vmap(lambda k: jax.random.split(k, 3))(keys)
            keys, movement_keys = split[:, 0], split[:, 2]
            f = learner.forward_network.apply(params, learner.featuriser.transform(position), tasks)
            q = jnp.einsum('nda,nd->na', f, tasks)
            valid = nominal_action_mask(position, actions, env.size, walls) & availability
            empty = ~jnp.any(valid, axis=1)
            # Report any fallback explicitly; it should never occur in free space.
            mask = jnp.where(empty[:, None], availability, valid) if wall_aware else availability
            selected = jnp.argmax(jnp.where(mask, q, -jnp.inf), axis=1)

            def move(p, a, k):
                nxt, _, collision = _move(Point(*p), Point(*actions[a]), k,
                                          env._noise, env.size, walls)
                return jnp.asarray((nxt.x, nxt.y)), collision
            nxt, collision = jax.vmap(move)(position, selected, movement_keys)
            active = hits > horizon
            nxt = jnp.where(active[:, None], nxt, position)
            reached = jnp.all(jnp.abs(nxt-goals) <= env.threshold_distance, axis=1)
            hits = jnp.where(active & reached, t+1, hits)
            record = dict(position=position, next_position=nxt, action=selected, q=q,
                          valid=valid, active=active, collision=collision & active,
                          empty_mask=empty & active)
            return (nxt, hits, keys), record
        (_, hits, _), records = jax.lax.scan(step, (starts, jnp.full(len(starts), horizon+1, jnp.int32), keys), jnp.arange(horizon))
        return hits, jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), records)
    return trace


def collision_runs(records):
    a, c = records['action'], records['collision']
    runs = np.zeros_like(a, dtype=np.int32)
    for t in range(a.shape[1]):
        previous = runs[:, t-1] if t else np.zeros(len(a), dtype=np.int32)
        repeat = a[:, t] == a[:, t-1] if t else np.zeros(len(a), bool)
        runs[:, t] = np.where(c[:, t], np.where(repeat, previous+1, 1), 0)
    return runs


def summarise(hits, records, horizon):
    good = hits <= horizon
    runs = collision_runs(records).max(axis=1)
    failed = ~good
    active = records['active']
    blocked = records['collision']
    summary = dict(success_rate=float(good.mean()),
                   mean_success_steps=float(hits[good].mean()) if good.any() else None,
                   median_success_steps=float(np.median(hits[good])) if good.any() else None,
                   wall_action_fraction=float(blocked.sum()/active.sum()),
                   failed_wall_action_fraction=float(blocked[failed].sum()/active[failed].sum()) if failed.any() else None,
                   empty_mask_steps=int(records['empty_mask'].sum()),
                   collision_nonzero_displacement=int(np.any(records['position'] != records['next_position'], axis=-1)[blocked].sum()))
    for minimum in (5, 10, 20):
        summary[f'failed_fraction_same_action_collision_run_at_least_{minimum}'] = float(np.mean(runs[failed] >= minimum)) if failed.any() else None
    return summary


def audit_collection(env, seed, episodes=256, steps=30):
    """Audit fresh data from the actual training collector, without updating FB."""
    _, batch = jax.jit(make_collector(env, episodes, steps))(*jax.random.split(jax.random.PRNGKey(seed), 3))
    data = jax.tree_util.tree_map(np.asarray, batch)
    moved = np.any(data['obs'] != data['obs_next'], axis=1)
    walls = tuple(env.wall_blocks)
    def intersects(p, q):
        return jnp.any(jnp.asarray([w.intersects_segment(Point(*p), Point(*q)) for w in walls]))
    crossing = np.asarray(jax.jit(jax.vmap(intersects))(batch['obs'], batch['obs_next'])) & moved
    replay = ReplayBuffer(len(moved), 2, len(env.actions), device='cpu', seed=seed)
    replay.add_batch(data)
    identical = all(np.array_equal(replay.storage[k], v) for k, v in data.items())
    # Force a move into the central vertical wall; actual noise is retained.
    origin = Point(.53, .30)
    def probe(k):
        nxt, _, collided = _move(origin, Point(.05, 0.), k, env._noise, env.size, walls)
        return jnp.asarray((nxt.x, nxt.y)), collided
    positions, collided = jax.jit(jax.vmap(probe))(jax.random.split(jax.random.PRNGKey(seed+1), 1000))
    positions, collided = np.asarray(positions), np.asarray(collided)
    stationary = np.all(positions == np.asarray((.53, .30), np.float32), axis=1)
    result = dict(sampled_transitions=len(moved), stationary_transitions=int((~moved).sum()),
                  nonstationary_wall_crossings=int(crossing.sum()), replay_preserves_transitions=bool(identical),
                  blocked_probe_count=1000, blocked_probe_collisions=int(collided.sum()),
                  blocked_probe_collision_nonstationary=int((collided & ~stationary).sum()))
    assert not crossing.any() and identical and not np.any(collided & ~stationary), result
    return result


def draw_environment(ax, env):
    for w in env.wall_blocks:
        ax.add_patch(Rectangle((w.xmin, w.ymin), w.xmax-w.xmin, w.ymax-w.ymin, color='.35'))
    ax.set(xlim=(0, env.size), ylim=(0, env.size), xlabel='x', ylabel='y', aspect='equal')


def run(args):
    source = Path(args.pairs)
    original_metrics = json.loads((source.parent/'metrics.json').read_text())
    seed, horizon, trials = original_metrics['seed'], original_metrics['horizon'], original_metrics['rollouts']
    data = np.load(source)
    starts, goals = data['starts'], data['goals']
    if args.samples:
        starts, goals = starts[:args.samples], goals[:args.samples]
    count = len(starts)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != original_metrics['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the paired reference evaluation')
    learner = FBLearner.from_checkpoint(checkpoint, device=args.device, buffer_device='cpu')
    records, hitsets, rows = {}, {}, []
    with jax.default_device(learner.device):
        env = ContinuousFourRooms(include_stay_action=learner.n_actions == 5)
        tasks = learner.embed_B(jnp.asarray(goals))
        keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(seed+1), i))(jnp.arange(count*trials))
        for snapshot in ('online', 'target'):
            params = learner.state['params' if snapshot == 'online' else 'target']['f']
            for masked in (False, True):
                name = snapshot+('_wall_aware' if masked else '_greedy')
                print('Evaluating '+name, flush=True)
                fn = make_trace(learner, env, horizon, masked)
                batches, hits = [], []
                for first in range(0, count, args.batch_pairs):
                    last = min(first+args.batch_pairs, count)
                    h, r = fn(params, jnp.repeat(jnp.asarray(starts[first:last]), trials, axis=0),
                              jnp.repeat(jnp.asarray(goals[first:last]), trials, axis=0),
                              jnp.repeat(tasks[first:last], trials, axis=0), keys[first*trials:last*trials])
                    hits.append(np.asarray(h)); batches.append(jax.tree_util.tree_map(np.asarray, r))
                h = np.concatenate(hits)
                r = {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
                records[name], hitsets[name] = r, h
                row = dict(policy=name, **summarise(h, r, horizon))
                if not masked:
                    old = data[f'greedy_{snapshot}_hits'][:count].reshape(-1)
                    row['reference_hit_mismatches'] = int(np.sum(old != h))
                    assert np.array_equal(old, h), 'Baseline rollout mismatch'
                else:
                    baseline = hitsets[snapshot+'_greedy'] <= horizon
                    improved = (h <= horizon).astype(float)-baseline.astype(float)
                    # Cluster by endpoint pair; noisy repeats are not independent pairs.
                    pair_delta = improved.reshape(count, trials).mean(axis=1)
                    boot = np.random.default_rng(seed+10).choice(pair_delta, size=(2000, count), replace=True).mean(axis=1)
                    row['paired_success_improvement'] = float(pair_delta.mean())
                    row['paired_success_improvement_ci95'] = np.quantile(boot, [.025, .975]).tolist()
                rows.append(row)
                np.savez_compressed(out/(name+'_traces.npz'), hits=h, **r)
                print(json.dumps(row), flush=True)
        audit = audit_collection(env, seed+100)
        print('Collection audit: '+json.dumps(audit), flush=True)

    examples = []
    fig, axes = plt.subplots(4, 2, figsize=(11, 16), constrained_layout=True)
    for row_index, (snapshot, category) in enumerate((('online', 'rescued'), ('online', 'still_failed'), ('target', 'rescued'), ('target', 'still_failed'))):
        base, masked = snapshot+'_greedy', snapshot+'_wall_aware'
        r, hits = records[base], hitsets[base]
        runs = collision_runs(r)
        longest = runs.max(axis=1)
        eligible = (hits > horizon) & ((hitsets[masked] <= horizon) if category == 'rescued' else (hitsets[masked] > horizon))
        if not eligible.any():
            for ax in axes[row_index]: ax.set_visible(False)
            continue
        trial_index = int(np.argmax(np.where(eligible, longest, -1)))
        end = int(np.argmax(runs[trial_index])); t = end-int(longest[trial_index])+1
        pair, repeat = divmod(trial_index, trials)
        q = r['q'][trial_index, t]; valid = r['valid'][trial_index, t]
        chosen = int(r['action'][trial_index, t]); alternate = int(np.argmax(np.where(valid, q, -np.inf)))
        example = dict(snapshot=snapshot, outcome=category, pair=pair, repeat=repeat,
                       start=starts[pair].tolist(), goal=goals[pair].tolist(),
                       trapped_position=r['position'][trial_index, t].tolist(),
                       first_stuck_step=t+1, repeated_collisions=int(longest[trial_index]),
                       action_names=env.actions_str, q_values=q.tolist(), nominal_valid=valid.tolist(),
                       chosen_action=env.actions_str[chosen], best_allowed_action=env.actions_str[alternate],
                       q_advantage_over_allowed=float(q[chosen]-q[alternate]),
                       wall_aware_hit_step=int(hitsets[masked][trial_index]))
        examples.append(example)
        ax, bar = axes[row_index]; draw_environment(ax, env)
        for name, color in ((base, 'tab:red'), (masked, 'tab:blue')):
            n = min(int(hitsets[name][trial_index]), horizon)
            points = np.vstack((records[name]['position'][trial_index, :n], records[name]['next_position'][trial_index, n-1]))
            ax.plot(*points.T, '.-', color=color, ms=2, lw=1, label=name)
        ax.scatter(*starts[pair], color='black', marker='o', s=25)
        ax.scatter(*goals[pair], color='green', marker='*', s=70)
        ax.scatter(*r['position'][trial_index, t], color='orange', marker='x', s=60)
        ax.add_patch(Rectangle(goals[pair]-.1, .2, .2, fill=False, edgecolor='green', ls=':'))
        ax.set_title(f'{snapshot}: {category}; {longest[trial_index]} repeated wall actions'); ax.legend(fontsize=7)
        bar.bar(env.actions_str, q, color=['tab:blue' if v else 'tab:red' for v in valid])
        bar.set(ylabel='Q(s,a,B(goal))', title=f'Stuck state at step {t+1}; red = wall collision')
        for i, value in enumerate(q): bar.annotate(f'{value:.2f}', (i, value), ha='center', va='bottom', fontsize=8)
    fig.savefig(out/'trapped_trajectories_and_q.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for name, h in hitsets.items():
        axes[0].plot(np.arange(horizon+1), [np.mean(h <= t) for t in range(horizon+1)], label=name)
    axes[0].set(xlabel='Action steps', ylabel='Goal success fraction'); axes[0].legend(fontsize=8)
    axes[1].bar([r['policy'] for r in rows], [r['wall_action_fraction'] for r in rows])
    axes[1].tick_params(axis='x', rotation=30); axes[1].set_ylabel('Fraction of active actions colliding with walls')
    fig.savefig(out/'mask_comparison.png', dpi=150); plt.close(fig)
    metrics = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest, pairs=count, repeats=trials,
                   horizon=horizon, seed=seed, environment=original_metrics['dynamics'],
                   results=rows, collection_audit=audit, examples=examples,
                   notes=['Goal tasks use online B(goal) for both forward-network snapshots.',
                          'Wall-aware mask checks noiseless clipped segments; actual movement noise is unchanged.',
                          'All four conditions use identical pairs and per-step movement keys.',
                          'Outer-boundary clipping is not masked. Goal tolerance is the native L-infinity .1 predicate.',
                          'Fresh collector data were audited; historical replay is absent from this checkpoint.',
                          'The checkpoint is continuing Cauchy FB; evaluation stops at the goal only to measure navigation.',
                          'Conditional successful path lengths exclude failures. No learner updates were performed.'])
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2))
    with (out/'summary.csv').open('w', newline='') as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    report = ['# FB wall-trap diagnostic', '', f'{count} identical pairs, {trials} repeats, {horizon} actions.', '',
              '| Policy | Success | Mean successful steps | Wall actions | Failed with >=10 identical collision actions |',
              '|---|---:|---:|---:|---:|']
    for r in rows:
        mean = 'N/A' if r['mean_success_steps'] is None else f'{r["mean_success_steps"]:.1f}'
        stuck = r['failed_fraction_same_action_collision_run_at_least_10']
        report.append(f'| {r["policy"]} | {r["success_rate"]:.1%} | {mean} | {r["wall_action_fraction"]:.1%} | {stuck:.1%} |')
    report += ['', '## Q-values at selected trapped states', '',
               '| Snapshot / masked outcome | Pair / trial | LEFT | RIGHT | UP | DOWN | Blocked argmax | Repeated collisions |',
               '|---|---|---:|---:|---:|---:|---|---:|']
    for e in examples:
        values = ' | '.join(f'{q:.2f}' for q in e['q_values'])
        report.append(f'| {e["snapshot"]} / {e["outcome"]} | {e["pair"]} / {e["repeat"]} | {values} | {e["chosen_action"]} | {e["repeated_collisions"]} |')
    report += ['', 'Collection audit: '+json.dumps(audit), '', *metrics['notes'], '',
               'Collision self-loops keep state, task and greedy argmax unchanged. Without sufficient movement noise to escape, the same action is selected repeatedly.',
               'Masking tests this behavioural mechanism; it does not establish the exact cause of the learned Q ranking.',
               'The learner was trained for target softmax at T=200; greedy is a different policy. Approximation errors and this policy mismatch are plausible sources of incorrect wall-action rankings.',
               'No raw one-step goal reward is present in the FB loss: Q uses an embedding-defined surrogate reward, so a wrong ranking need not imply a collision-dynamics bug.', '',
               'The environment reports all four movement actions available, including wall-rejected actions. Existing availability masking therefore does not prohibit collisions.',
               'Residual wall collisions under the diagnostic mask arise because the actual movement is noisy. The plotted unresolved example instead selects LEFT into the outer boundary after UP is masked; outer-boundary clipping is deliberately unchanged.', '',
               'Run: python diagnose_fb_walls.py. Smoke: add --samples 16 --output evaluation_original_walls_smoke.']
    (out/'RESULTS.md').write_text('\n'.join(report))
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='models/fb_fourrooms_jax_original.pkl')
    parser.add_argument('--pairs', default='evaluation_original_short_horizon/trials.npz')
    parser.add_argument('--output', default='evaluation_original_walls')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--samples', type=int, default=0, help='0 uses all saved pairs')
    parser.add_argument('--batch-pairs', type=int, default=100)
    run(parser.parse_args())
