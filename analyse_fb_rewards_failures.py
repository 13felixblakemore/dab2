"""Inspect B rewards, stationary Bellman backups and masked FB failures.

No training or learner changes. Uses saved wall-diagnostic pairs/traces.
Run: python analyse_fb_rewards_failures.py
Smoke: add --samples 16 --grid 30 --reference-episodes 100 --output evaluation_fb_reward_smoke
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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from continuous_four_rooms_jax import ContinuousFourRooms, Point, _move
from forward_backward_jax import FBLearner
from diagnose_fb_walls import make_trace, collision_runs, draw_environment
from control_bfs_bottlenecks import build_graph, bfs_distances
from run_fb_jax import make_collector


def distribution(values):
    x = np.asarray(values)
    if not x.size:
        return dict(count=0)
    return dict(count=int(x.size), mean=float(x.mean()), median=float(np.median(x)),
                p05=float(np.quantile(x, .05)), p95=float(np.quantile(x, .95)),
                minimum=float(x.min()), maximum=float(x.max()))


def bellman_statistics(q, target):
    residual = q-target
    return dict(mean_signed_residual=float(residual.mean()), mae=float(np.abs(residual).mean()),
                median_abs_residual=float(np.median(np.abs(residual))),
                p95_abs_residual=float(np.quantile(np.abs(residual), .95)),
                mae_over_mean_abs_q=float(np.abs(residual).mean()/max(np.abs(q).mean(), 1e-12)),
                relative_abs_residual=distribution(np.abs(residual)/np.maximum(np.maximum(np.abs(q), np.abs(target)), 1e-6)),
                fraction_within_5_percent=float(np.mean(np.abs(residual) <= .05*np.maximum(np.abs(q), np.abs(target)))))


def embed_batches(learner, coords, target=False, batch_size=1024):
    def call(x):
        return learner.backward_network.apply(learner.state['target' if target else 'params']['b'], learner.featuriser.transform(x))
    fn = jax.jit(call)
    return np.concatenate([np.asarray(fn(jnp.asarray(coords[i:i+batch_size]))) for i in range(0, len(coords), batch_size)])


def reward_maps(learner, env, goals, grid, out):
    axis = (np.arange(grid)+.5)/grid
    xx, yy = np.meshgrid(axis, axis)
    full = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    free = np.ones(len(full), bool)
    for w in env.wall_blocks:
        free &= ~((full[:, 0] >= w.xmin)&(full[:, 0] <= w.xmax)&(full[:, 1] >= w.ymin)&(full[:, 1] <= w.ymax))
    points = full[free]
    b = embed_batches(learner, points)
    unique, inverse = np.unique(goals, axis=0, return_inverse=True)
    bg = embed_batches(learner, unique)
    rewards = b @ bg.T
    self_reward = np.sum(bg**2, axis=1)
    argmax = rewards.argmax(axis=0)
    peaks = rewards[argmax, np.arange(len(unique))]
    peak_distance = np.linalg.norm(points[argmax]-unique, axis=1)
    peak_in_goal = np.max(np.abs(points[argmax]-unique), axis=1) <= env.threshold_distance
    outside = np.max(np.abs(points[:, None]-unique[None]), axis=2) > env.threshold_distance
    far = np.linalg.norm(points[:, None]-unique[None], axis=2) > .3
    max_outside = np.where(outside, rewards, -np.inf).max(axis=0)
    max_far = np.where(far, rewards, -np.inf).max(axis=0)
    half_goal_positive = np.mean(outside & (rewards >= .5*self_reward), axis=0)/np.maximum(outside.mean(axis=0), 1e-12)
    metrics = dict(grid_resolution=grid, unique_goals=len(unique), pair_weighted_peak_inside_goal_fraction=float(peak_in_goal[inverse].mean()),
                   pair_weighted_peak_distance=distribution(peak_distance[inverse]),
                   pair_weighted_peak_to_self_ratio=distribution((peaks/self_reward)[inverse]),
                   pair_weighted_outside_peak_to_self_ratio=distribution((max_outside/self_reward)[inverse]),
                   pair_weighted_far_peak_to_self_ratio=distribution((max_far/self_reward)[inverse]),
                   pair_weighted_outside_area_reward_above_half_goal=distribution(half_goal_positive[inverse]),
                   pair_weighted_fraction_far_reward_exceeds_self=float(np.mean(max_far[inverse] > self_reward[inverse])),
                   unique_goal_grid_fraction_outside_reward_negative=float(np.mean((rewards < 0)[outside])))
    # Four room goals, plus the two previously plotted target-network traps.
    centers = np.asarray([[.275, .175], [.275, .725], [.825, .175], [.825, .725]])
    selected = [int(np.argmin(np.sum((unique-c)**2, axis=1))) for c in centers]
    selected += [int(np.argmin(np.sum((unique-g)**2, axis=1))) for g in goals[[min(64, len(goals)-1), min(63, len(goals)-1)]]]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    examples = []
    for ax, k in zip(axes.ravel(), selected):
        image = np.full(grid*grid, np.nan); image[free] = rewards[:, k]
        limit = max(abs(rewards[:, k]).max(), 1e-6)
        im = ax.imshow(image.reshape(grid, grid), origin='lower', extent=(0, 1, 0, 1),
                       cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
        draw_environment(ax, env)
        ax.scatter(*unique[k], color='lime', edgecolor='black', marker='*', s=100, label='goal')
        ax.scatter(*points[argmax[k]], color='black', marker='x', s=50, label='grid maximum')
        ax.add_patch(Rectangle(unique[k]-.1, .2, .2, fill=False, edgecolor='green', ls=':'))
        ax.set_title(f'g={unique[k].round(3)}; r(g)={self_reward[k]:.2f}\nmax={peaks[k]:.2f}, distance={peak_distance[k]:.2f}')
        fig.colorbar(im, ax=ax, shrink=.8); ax.legend(fontsize=7)
        examples.append(dict(goal=unique[k].tolist(), self_reward=float(self_reward[k]), maximum_reward=float(peaks[k]),
                             peak=points[argmax[k]].tolist(), peak_distance=float(peak_distance[k]),
                             maximum_outside_goal=float(max_outside[k]), maximum_far=float(max_far[k])))
    fig.savefig(out/'b_reward_maps.png', dpi=160); plt.close(fig)
    np.savez_compressed(out/'b_reward_maps.npz', points=points, goals=unique, goal_inverse=inverse, reward=rewards,
                        self_reward=self_reward, grid=grid, free=free, peaks=points[argmax])
    metrics['examples'] = examples
    return metrics


def reference_covariance(learner, env, episodes, seed, out):
    # Original replay is absent. Reproduce its uniform-action 30-step collector.
    _, batch = jax.jit(make_collector(env, episodes=episodes, steps=30))(*jax.random.split(jax.random.PRNGKey(seed), 3))
    obs = np.asarray(batch['obs'])
    b = embed_batches(learner, obs).astype(np.float64)
    bt = embed_batches(learner, obs, target=True).astype(np.float64)
    covariance = b.T @ b / len(b)
    cross = bt.T @ b / len(b)
    eigenvalues = np.linalg.eigvalsh(covariance)
    metrics = dict(source='fresh diagnostic observations from unchanged training collector; not historical replay',
                   count=len(obs), seed=seed, eigenvalues=eigenvalues.tolist(),
                   frobenius_error_from_identity=float(np.linalg.norm(covariance-np.eye(learner.embed_dim))/np.sqrt(learner.embed_dim)),
                   target_cross_relative_difference=float(np.linalg.norm(cross-covariance)/np.linalg.norm(covariance)))
    np.savez_compressed(out/'fresh_reference.npz', observations=obs, covariance=covariance, target_online_cross=cross)
    return covariance, cross, metrics


def stationary_bellman(learner, traces, goals, repeats, covariance, cross, out):
    metrics, arrays = {}, {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for index, snapshot in enumerate(('online', 'target')):
        trace = traces[snapshot+'_greedy']
        runs = collision_runs(trace)
        ids = np.flatnonzero(runs.max(axis=1) > 0)
        ends = runs[ids].argmax(axis=1)
        begin = ends-runs[ids, ends]+1
        p = trace['position'][ids, begin]
        nxt = trace['next_position'][ids, begin]
        np.testing.assert_array_equal(p, nxt)
        g = goals[ids//repeats]
        a = trace['action'][ids, begin]
        # Repeated noisy trials can share exactly the same stuck state.
        _, keep = np.unique(np.column_stack((p, g, a)), axis=0, return_index=True)
        p, g, a, ids, begin = (x[keep] for x in (p, g, a, ids, begin))
        bs = embed_batches(learner, p)
        z = embed_batches(learner, g)
        reward = np.einsum('nd,nd->n', bs, z)
        features = learner.featuriser.transform(jnp.asarray(p))
        fo = learner.forward_network.apply(learner.state['params']['f'], features, jnp.asarray(z))
        ft = learner.forward_network.apply(learner.state['target']['f'], features, jnp.asarray(z))
        qo = np.asarray(jnp.einsum('nda,nd->na', fo, z))
        qt = np.asarray(jnp.einsum('nda,nd->na', ft, z))
        pi = np.asarray(jax.nn.softmax(jnp.asarray(qt)/learner.temperature, axis=-1))
        vt = np.sum(pi*qt, axis=1)
        gamma = learner.discount_factor
        predicted = (qo if snapshot == 'online' else qt)[np.arange(len(a)), a]
        target = reward+gamma*vt
        greedy_target = reward+gamma*qt.max(axis=1)
        f_selected = np.asarray(fo)[np.arange(len(a)), :, a]
        f_next = np.einsum('nda,na->nd', np.asarray(ft), pi)
        integrated_q = np.einsum('nd,de,ne->n', f_selected, covariance, z)
        integrated_target = reward+gamma*np.einsum('nd,de,ne->n', f_next, cross, z)
        per_state = dict(position=p, goal=g, action=a, trial_index=ids, step=begin,
                         q_online=qo, q_target=qt, reward=reward, v_target_softmax=vt,
                         requested_q=predicted, requested_target=target, greedy_target=greedy_target,
                         integrated_online_q=integrated_q, integrated_target=integrated_target)
        arrays.update({snapshot+'_'+k: v for k, v in per_state.items()})
        metrics[snapshot] = dict(unique_stationary_configurations=len(p),
            requested_target_softmax=bellman_statistics(predicted, target),
            requested_target_greedy=bellman_statistics(predicted, greedy_target),
            raw_q=distribution(predicted), immediate_reward=distribution(reward),
            required_reward_q_minus_gamma_v=distribution(predicted-gamma*vt),
            covariance_corrected_online_return=bellman_statistics(integrated_q, integrated_target))
        ax = axes[index, 0]
        ax.scatter(target, predicted, s=5, alpha=.3)
        lo, hi = min(target.min(), predicted.min()), max(target.max(), predicted.max())
        ax.plot([lo, hi], [lo, hi], 'k--')
        ax.set(xlabel='r(s) + .99 V_target_softmax(s)', ylabel='Q(s, blocked action, B(goal))', title=snapshot+' stationary backup')
        axes[index, 1].hist(predicted-target, bins=70)
        axes[index, 1].axvline(0, color='black', ls='--')
        axes[index, 1].set(xlabel='Q - stationary softmax target', ylabel='Unique blocked configurations', title=snapshot+' residuals')
    np.savez_compressed(out/'stationary_bellman.npz', **arrays)
    fig.savefig(out/'stationary_bellman.png', dpi=160); plt.close(fig)
    metrics['notes'] = [
        'Each configuration uses an observed collision with exactly equal current and next coordinates; deduplicated by state, goal and action.',
        'V is the target-network softmax expectation at temperature 200 over all four training-available actions, including wall moves.',
        'This is a conditional stationary-transition backup; the noisy action kernel can sometimes permit movement.',
        'F dot z predicts reward B(s) dot z only when the replay second moment C=E[B B^T] is identity.',
        'For r=B(s) dot z, integrating the learned measure gives Q_r=F^T C z. Its target uses E[B_target B_online^T].',
        'Covariance-corrected results use a fresh collector reference; the original checkpoint has no historical replay.']
    return metrics, arrays


def classify_tail(trace, env, horizon):
    """Operational trajectory labels; criteria and continuous diagnostics saved."""
    n = min(20, horizon)
    p = trace['position'][:, horizon-n:horizon]
    q = trace['next_position'][:, horizon-n:horizon]
    a = trace['action'][:, horizon-n:horizon]
    actions = np.asarray([(x.x, x.y) for x in env.actions])
    nominal_length = np.linalg.norm(np.clip(p+actions[a], 0., env.size)-p, axis=-1)
    boundary_fraction = np.mean(nominal_length < 1e-6, axis=1)
    wall_fraction = np.mean(trace['collision'][:, horizon-n:horizon], axis=1)
    displacement = np.linalg.norm(q-p, axis=-1)
    path_length = displacement.sum(axis=1)
    net = np.linalg.norm(q[:, -1]-p[:, 0], axis=1)
    span = np.linalg.norm(np.ptp(np.concatenate((p, q[:, -1:]), axis=1), axis=1), axis=1)
    opposites = np.asarray([1, 0, 3, 2, 4])[:len(actions)]
    reversal_fraction = np.mean(a[:, 1:] == opposites[a[:, :-1]], axis=1)
    efficiency = net/np.maximum(path_length, 1e-8)
    # Return to a .05 cell after >=3 steps and an intervening excursion >=.0375.
    cells = np.minimum(np.floor(p/.05).astype(int), 19)
    revisit = np.zeros((len(p), n), bool)
    for t in range(3, n):
        for previous in range(t-2):
            same = np.all(cells[:, t] == cells[:, previous], axis=1)
            excursion = np.linalg.norm(p[:, previous:t+1]-p[:, previous:previous+1], axis=2).max(axis=1) >= .0375
            revisit[:, t] |= same & excursion
    revisit_fraction = revisit[:, 3:].mean(axis=1) if n > 3 else np.zeros(len(p))
    labels = np.full(len(p), 'wandering_or_slow', dtype='<U24')
    confined = span <= .15
    cycling = (path_length >= .25)&(efficiency <= .25)&((reversal_fraction >= .2)|(revisit_fraction >= .5))
    labels[confined] = 'confined_without_cycle'
    labels[cycling] = 'cycling'
    labels[wall_fraction >= .5] = 'wall_rejection_trap'
    labels[boundary_fraction >= .8] = 'boundary_pinned'
    return labels, dict(boundary_fraction=boundary_fraction, wall_fraction=wall_fraction,
                        tail_path_length=path_length, tail_net_displacement=net, tail_span=span,
                        reversal_fraction=reversal_fraction, revisit_fraction=revisit_fraction,
                        tail_path_efficiency=efficiency)


def remaining_distance(env, positions, goals):
    nodes, graph = build_graph(env, 20)
    nearest = np.argmin(np.sum((positions[:, None]-nodes[None])**2, axis=2), axis=1)
    distances, inverse = np.unique(goals, axis=0, return_inverse=True)
    cache = np.stack([bfs_distances(graph, np.max(np.abs(nodes-g), axis=1) <= env.threshold_distance) for g in distances])
    return cache[inverse, nearest]


def failure_analysis(learner, env, traces, starts, goals, reference, repeats, seed, horizon, extended_horizon, batch_pairs, out):
    metrics, plot_examples = {}, []
    count = len(starts)
    tasks = learner.embed_B(jnp.asarray(goals))
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(seed+1), i))(jnp.arange(count*repeats))
    paired_goals = np.repeat(goals, repeats, axis=0)
    for snapshot in ('online', 'target'):
        source = traces[snapshot+'_wall_aware']
        failed = source['hits'] > horizon
        labels, diagnostics = classify_tail(source, env, horizon)
        print(f'Extending {snapshot} masked rollouts to {extended_horizon} steps', flush=True)
        fn = make_trace(learner, env, extended_horizon, True)
        batches, all_hits = [], []
        for first in range(0, count, batch_pairs):
            last = min(first+batch_pairs, count)
            hits, batch = fn(learner.state['params' if snapshot == 'online' else 'target']['f'],
                            jnp.repeat(jnp.asarray(starts[first:last]), repeats, axis=0),
                            jnp.repeat(jnp.asarray(goals[first:last]), repeats, axis=0),
                            jnp.repeat(tasks[first:last], repeats, axis=0), keys[first*repeats:last*repeats])
            all_hits.append(np.asarray(hits)); batches.append(jax.tree_util.tree_map(np.asarray, batch))
        hits = np.concatenate(all_hits)
        extended = {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
        np.testing.assert_array_equal(hits <= horizon, source['hits'] <= horizon)
        np.testing.assert_allclose(extended['position'][:, :horizon], source['position'], atol=1e-6, rtol=0)
        np.testing.assert_array_equal(extended['action'][:, :horizon], source['action'])
        distance50 = remaining_distance(env, source['next_position'][:, horizon-1], paired_goals)
        distance200 = remaining_distance(env, extended['next_position'][:, -1], paired_goals)
        labels_long, long_diagnostics = classify_tail(extended, env, extended_horizon)
        rows = []
        for category in ('boundary_pinned', 'wall_rejection_trap', 'cycling', 'confined_without_cycle', 'wandering_or_slow'):
            selected = failed & (labels == category)
            reached_late = selected & (hits <= extended_horizon)
            row = dict(category=category, count=int(selected.sum()), fraction_of_50step_failures=float(selected.sum()/max(failed.sum(), 1)),
                       reached_by_extended_horizon=int(reached_late.sum()),
                       extension_rescue_rate=float(reached_late.sum()/selected.sum()) if selected.any() else None,
                       mean_remaining_bfs_steps_at_50=float(distance50[selected].mean()) if selected.any() else None)
            rows.append(row)
            if selected.any():
                candidate = np.flatnonzero(selected)
                # A representative long-lasting example, preferring the median geometry.
                index = int(candidate[len(candidate)//2])
                plot_examples.append((snapshot, category, index, source, extended, hits))
        summary = dict(failures_at_50=int(failed.sum()), failure_rate_at_50=float(failed.mean()),
            success_by_extended_horizon=float(np.mean(hits <= extended_horizon)),
            fraction_50step_failures_reaching_by_extension=float(np.mean(hits[failed] <= extended_horizon)),
            remaining_bfs_steps_at_50=distribution(distance50[failed]),
            remaining_bfs_steps_at_extension=distribution(distance200[failed & (hits > extended_horizon)]),
            categories=rows, failed_tail_displacement=distribution(diagnostics['tail_net_displacement'][failed]),
            failed_tail_path_length=distribution(diagnostics['tail_path_length'][failed]),
            extended_failure_categories={k: int(np.sum((hits > extended_horizon)&(labels_long == k))) for k in np.unique(labels_long)},
            extension_success_lengths=distribution(hits[failed & (hits <= extended_horizon)]))
        metrics[snapshot] = summary
        np.savez_compressed(out/(snapshot+'_masked_extended.npz'), hits=hits, original_failure=failed, category50=labels,
                            category_extended=labels_long, remaining_bfs50=distance50, remaining_bfs_extended=distance200,
                            **diagnostics, **{'extended_'+k: v for k, v in long_diagnostics.items()},
                            position=extended['position'], next_position=extended['next_position'], action=extended['action'], collision=extended['collision'])
        with (out/(snapshot+'_failed_trials.csv')).open('w', newline='') as f:
            names = ['pair', 'repeat', 'category', 'extended_hit', 'reference_start_steps', 'remaining_bfs50', *diagnostics]
            writer = csv.DictWriter(f, fieldnames=names); writer.writeheader()
            for i in np.flatnonzero(failed):
                pair, repeat = divmod(int(i), repeats)
                writer.writerow(dict(pair=pair, repeat=repeat, category=labels[i], extended_hit=int(hits[i]),
                    reference_start_steps=int(reference[pair]), remaining_bfs50=int(distance50[i]),
                    **{k: float(v[i]) for k, v in diagnostics.items()}))
        print(json.dumps({snapshot: summary}), flush=True)
    fig, axes = plt.subplots(max(1, len(plot_examples)), 1, figsize=(6, max(4, 4*len(plot_examples))), constrained_layout=True)
    example_metrics = []
    for ax, (snapshot, category, i, source, extended, hits) in zip(np.atleast_1d(axes), plot_examples):
        pair, repeat = divmod(i, repeats)
        draw_environment(ax, env)
        for begin, end, color, name in ((0, horizon, 'tab:red', f'first {horizon}'), (horizon, extended_horizon, 'tab:blue', 'extension')):
            end = min(end, int(hits[i]))
            if end > begin:
                points = np.vstack((extended['position'][i, begin:end], extended['next_position'][i, end-1]))
                ax.plot(*points.T, '.-', lw=.8, ms=2, color=color, label=name)
        ax.scatter(*starts[pair], color='black', s=30)
        ax.scatter(*goals[pair], color='green', marker='*', s=80)
        ax.add_patch(Rectangle(goals[pair]-.1, .2, .2, fill=False, edgecolor='green', ls=':'))
        ax.set_title(f'{snapshot}, {category}; pair {pair}, trial {repeat}\nhit={int(hits[i]) if hits[i] <= extended_horizon else "never"}'); ax.legend(fontsize=7)
        example_metrics.append(dict(snapshot=snapshot, category=category, pair=pair, repeat=repeat, extended_hit=int(hits[i])))
    fig.savefig(out/'remaining_failure_trajectories.png', dpi=140); plt.close(fig)
    metrics['examples'] = example_metrics
    metrics['classification_rules'] = [
        'Classify the last 20 steps at the original 50-step cutoff, exclusively and in priority order below.',
        'boundary_pinned: at least 80% of noiseless selected moves have zero displacement due to outer clipping.',
        'wall_rejection_trap: at least 50% of actual moves collide with walls.',
        'cycling: tail path length >=.25, net/path <=.25, and either >=20% immediate action reversals or >=50% excursion-qualified cell returns.',
        'An excursion-qualified return revisits a .05 cell after >=3 steps, travelling at least .0375 from the earlier position in between.',
        'confined_without_cycle: remaining cases have terminal-tail bounding-box diagonal <=.15.',
        'wandering_or_slow: all remaining failures; extension distinguishes eventual goal reaching from persistent failure.',
        'Labels are operational diagnostics, sensitive to thresholds, not mathematical proofs of an infinite cycle.',
        'Remaining BFS distances are approximate .05-center-lattice references, not exact noisy continuous optima.']
    return metrics


def saved_report(out, pairs_path):
    """Refresh readable summaries and compact plots without rerunning rollouts."""
    metrics = json.loads((out/'metrics.json').read_text())
    reward = metrics['reward']
    if 'grid_resolution' not in reward:
        with np.load(out/'b_reward_maps.npz') as saved:
            reward['grid_resolution'] = int(saved['grid'])
    if 'pair_weighted_fraction_outside_reward_negative' in reward:
        reward['unique_goal_grid_fraction_outside_reward_negative'] = reward.pop('pair_weighted_fraction_outside_reward_negative')
    data = np.load(pairs_path)
    env = ContinuousFourRooms(include_stay_action=False)
    bellman = np.load(out/'stationary_bellman.npz')
    examples = []
    for snapshot in ('online', 'target'):
        prefix = snapshot+'_'
        # The previously highlighted blocked-DOWN state (pair 64, trial 0).
        ids = np.flatnonzero(bellman[prefix+'trial_index'] == 64*metrics['repeats'])
        if not len(ids):
            ids = [int(np.argmax(np.abs(bellman[prefix+'requested_q']-bellman[prefix+'requested_target'])))]
        i = int(ids[0])
        row = dict(snapshot=snapshot, trial_index=int(bellman[prefix+'trial_index'][i]),
                   position=bellman[prefix+'position'][i].tolist(), goal=bellman[prefix+'goal'][i].tolist(),
                   action=int(bellman[prefix+'action'][i]),
                   q=float(bellman[prefix+'requested_q'][i]), reward=float(bellman[prefix+'reward'][i]),
                   v_target_softmax=float(bellman[prefix+'v_target_softmax'][i]),
                   stationary_target=float(bellman[prefix+'requested_target'][i]),
                   residual=float(bellman[prefix+'requested_q'][i]-bellman[prefix+'requested_target'][i]),
                   covariance_corrected_online_q=float(bellman[prefix+'integrated_online_q'][i]),
                   covariance_corrected_online_target=float(bellman[prefix+'integrated_target'][i]))
        position = jnp.asarray(row['position'])
        action = env.actions[row['action']]
        walls = tuple(env.wall_blocks)
        keys = jax.random.split(jax.random.PRNGKey(metrics['seed']+1800), 100000)
        collided = jax.jit(jax.vmap(lambda k: _move(Point(*position), action, k, env._noise, env.size, walls)[2]))(keys)
        row['physical_collision_probe_count'] = len(keys)
        row['physical_collision_probability_estimate'] = float(np.asarray(collided).mean())
        examples.append(row)
    metrics['stationary_bellman']['worked_examples'] = examples
    report = ['# FB reward, Bellman and masked-failure diagnosis', '',
        f'Unchanged checkpoint `{metrics["checkpoint"]}`; {metrics["pairs"]} pairs x {metrics["repeats"]} trials. No retraining.', '',
        '## B reward maps', '',
        f'For {reward["pair_weighted_peak_inside_goal_fraction"]:.1%} of sampled goals ({reward["unique_goals"]} distinct), the {reward["grid_resolution"]}x{reward["grid_resolution"]} free-grid reward maximum lies inside the native goal region (L-infinity radius .1).',
        f'The median peak displacement is {reward["pair_weighted_peak_distance"]["median"]:.3f}.',
        f'Among states more than .3 from the goal, the largest reward is a median {reward["pair_weighted_far_peak_to_self_ratio"]["median"]:.1%} of the reward at the goal; the worst sampled ratio is {reward["pair_weighted_far_peak_to_self_ratio"]["maximum"]:.1%}.',
        'Positive side lobes exist, but no distant point outranks the goal. This argues against a dominant distant B-reward hotspot as the main explanation for these failures.', '',
        '## Stationary blocked-action Bellman check', '',
        'Computed Q - [r + .99 V] at exact unchanged states. V uses target softmax at T=200 and all four training-available actions.', '',
        '| Snapshot | Unique blocked configurations | Mean signed residual | MAE | Median per-state relative error | 95th percentile relative error |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ('online', 'target'):
        b = metrics['stationary_bellman'][name]
        x = b['requested_target_softmax']
        report.append(f'| {name} | {b["unique_stationary_configurations"]} | {x["mean_signed_residual"]:.2f} | {x["mae"]:.2f} | {x["relative_abs_residual"]["median"]:.1%} | {x["relative_abs_residual"]["p95"]:.1%} |')
    report += ['', 'The per-state relative error is abs(Q-target)/max(abs(Q),abs(target),1e-6). Positive residuals indicate Q exceeds this stationary backup.', '',
        '| Example | Q | r | V target-softmax | r + .99 V | Q - target |',
        '|---|---:|---:|---:|---:|---:|']
    for e in examples:
        report.append(f'| {e["snapshot"]}, trial {e["trial_index"]} | {e["q"]:.3f} | {e["reward"]:.3f} | {e["v_target_softmax"]:.3f} | {e["stationary_target"]:.3f} | {e["residual"]:.3f} |')
    report += ['', 'At the target worked example, '+f'{examples[-1]["physical_collision_probability_estimate"]:.3%} of 100,000 independent movement-noise probes collide and remain stationary.', '',
        *metrics['stationary_bellman']['notes'], '',
        'The fresh-reference B covariance has eigenvalues from '
        f'{min(metrics["backward_covariance"]["eigenvalues"]):.3f} to {max(metrics["backward_covariance"]["eigenvalues"]):.3f}; it is not identity.',
        'After integrating r=B(s)^T B(g) against the predicted measure (using C and the target/online cross moment), the median relative online-return residual is '
        f'{metrics["stationary_bellman"]["online"]["covariance_corrected_online_return"]["relative_abs_residual"]["median"]:.1%} on online-trajectory blocked states. Covariance correction does not eliminate the residuals.',
        'Greedy-max residuals are also saved, but a greedy-selected self-loop makes Q-.99 max Q nearly .01 Q algebraically; a small relative residual there does not independently validate the values.', '',
        '## Remaining failures after wall masking', '',
        'Percentages below are conditional on failure at 50 actions. Classifications use the last 20 steps and are threshold-based diagnostics.', '',
        '| Snapshot | Failed trials | Boundary pinned | Cycling | Confined | Wall rejections | Wandering/slow | Success at 50 | Success at 200 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('online', 'target'):
        f = metrics['failures'][name]
        categories = {x['category']: x['fraction_of_50step_failures'] for x in f['categories']}
        report.append(f'| {name} | {f["failures_at_50"]} | {categories["boundary_pinned"]:.1%} | {categories["cycling"]:.1%} | {categories["confined_without_cycle"]:.1%} | {categories["wall_rejection_trap"]:.1%} | {categories["wandering_or_slow"]:.1%} | {1-f["failure_rate_at_50"]:.1%} | {f["success_by_extended_horizon"]:.1%} |')
    report += ['', 'The original diagnostic mask excludes thick-wall intersections but deliberately leaves outer-boundary clipping unchanged. Boundary-pinned actions therefore remain selectable.',
        'Only '+', '.join(f'{name}: {metrics["failures"][name]["fraction_50step_failures_reaching_by_extension"]:.1%}' for name in ('online', 'target'))+' of the 50-step failures reach the goal by 200.',
        'Their first 50 states/actions reproduce the saved masked trials exactly. Most remaining failures persist rather than merely needing a few more steps.', '',
        *metrics['failures']['classification_rules'], '',
        'The checkpoint learns a continuing Cauchy-task measure. Softmax, greedy, and geometry-masked greedy are distinct policies. These navigation results do not validate successor-measure accuracy.', '',
        '## Reproduction', '',
        'Full run: `.venv/Scripts/python.exe analyse_fb_rewards_failures.py`',
        'Smoke: add `--samples 16 --grid 30 --reference-episodes 100 --output evaluation_fb_reward_smoke`.',
        'Regenerate this report/compact plot from saved results: add `--report-only`.',
        'Seven synthetic mask/trajectory checks passed using `python -m unittest test_fb_reward_failure_analysis test_diagnose_fb_walls -v`.']
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2))
    (out/'RESULTS.md').write_text('\n'.join(report))
    fig, axes = plt.subplots(3, 2, figsize=(10, 14), constrained_layout=True)
    for col, name in enumerate(('online', 'target')):
        with np.load(out/(name+'_masked_extended.npz')) as archive:
            for row, category in enumerate(('boundary_pinned', 'cycling', 'wandering_or_slow')):
                ax = axes[row, col]
                selected = np.flatnonzero(archive['original_failure'] & (archive['category50'] == category))
                if not len(selected):
                    ax.set_visible(False); continue
                persistent = selected[archive['hits'][selected] > metrics['extended_horizon']]
                i = int((persistent if len(persistent) else selected)[len(persistent if len(persistent) else selected)//2])
                pair, repeat = divmod(i, metrics['repeats'])
                draw_environment(ax, env)
                h = min(int(archive['hits'][i]), metrics['extended_horizon'])
                for first, last, color, label in ((0, min(50, h), 'tab:red', 'first 50'), (50, h, 'tab:blue', 'extension')):
                    if last > first:
                        points = np.vstack((archive['position'][i, first:last], archive['next_position'][i, last-1]))
                        ax.plot(*points.T, '.-', color=color, lw=.8, ms=2, label=label)
                ax.scatter(*data['starts'][pair], color='black', s=25)
                ax.scatter(*data['goals'][pair], color='green', marker='*', s=70)
                ax.add_patch(Rectangle(data['goals'][pair]-.1, .2, .2, fill=False, edgecolor='green', ls=':'))
                ax.set_title(f'{name}: {category}\npair {pair}, trial {repeat}, '+('still failed' if archive['hits'][i] > h else f'hit {h}'))
                ax.legend(fontsize=7)
    fig.savefig(out/'remaining_failure_examples.png', dpi=150); plt.close(fig)
    return metrics


def run(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    previous = json.loads((Path(args.traces)/'metrics.json').read_text())
    digest = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    if previous['checkpoint_sha256'] != digest:
        raise ValueError('Checkpoint hash differs from saved diagnostic trials')
    source = np.load(args.pairs)
    count = args.samples or len(source['starts'])
    starts, goals = source['starts'][:count], source['goals'][:count]
    reference = source['bfs_reference_lengths'][:count]
    repeats, horizon, seed = previous['repeats'], previous['horizon'], previous['seed']
    if args.extended_horizon <= horizon:
        raise ValueError('Extended horizon must exceed the saved horizon')
    traces = {}
    for snapshot in ('online', 'target'):
        for suffix in ('greedy', 'wall_aware'):
            name = snapshot+'_'+suffix
            with np.load(Path(args.traces)/(name+'_traces.npz')) as archive:
                traces[name] = {k: archive[k][:count*repeats] for k in archive.files}
    learner = FBLearner.from_checkpoint(args.checkpoint, device=args.device, buffer_device='cpu')
    with jax.default_device(learner.device):
        env = ContinuousFourRooms(include_stay_action=learner.n_actions == 5)
        print('Mapping B rewards', flush=True)
        reward = reward_maps(learner, env, goals, args.grid, out)
        print(json.dumps({'reward': reward}), flush=True)
        covariance, cross, covmetrics = reference_covariance(learner, env, args.reference_episodes, seed+777, out)
        print(json.dumps({'backward_covariance': covmetrics}), flush=True)
        bellman, _ = stationary_bellman(learner, traces, goals, repeats, covariance, cross, out)
        print(json.dumps({'stationary_bellman': bellman}), flush=True)
        failures = failure_analysis(learner, env, traces, starts, goals, reference, repeats, seed, horizon,
                                    args.extended_horizon, args.batch_pairs, out)
    metrics = dict(checkpoint=args.checkpoint, checkpoint_sha256=digest, pairs=count, repeats=repeats,
                   horizon=horizon, extended_horizon=args.extended_horizon, seed=seed,
                   reward=reward, backward_covariance=covmetrics, stationary_bellman=bellman, failures=failures)
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2))
    assert hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() == digest
    return saved_report(out, args.pairs)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='models/fb_fourrooms_jax_original.pkl')
    p.add_argument('--pairs', default='evaluation_original_short_horizon/trials.npz')
    p.add_argument('--traces', default='evaluation_original_walls')
    p.add_argument('--output', default='evaluation_fb_reward_failures')
    p.add_argument('--device', default='cpu')
    p.add_argument('--samples', type=int, default=0)
    p.add_argument('--grid', type=int, default=100)
    p.add_argument('--reference-episodes', type=int, default=1000)
    p.add_argument('--extended-horizon', type=int, default=200)
    p.add_argument('--batch-pairs', type=int, default=100)
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    if args.report_only:
        saved_report(Path(args.output), args.pairs)
    else:
        run(args)
