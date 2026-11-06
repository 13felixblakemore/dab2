"""Noise-averaged FB Bellman, replay-coverage and independent return audit.

The default checkpoint is continuing FB. All MC returns therefore continue
after goal entry, with the target softmax policy and unchanged native dynamics.
No learner updates are performed. See output/RESULTS.md for assumptions.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fb_navigation.environments.continuous_four_rooms_jax import ContinuousFourRooms, Point, _move, _sample_point
from fb_navigation.agents.forward_backward_jax import FBLearner
from fb_navigation.training.run_fb_jax import make_collector
from fb_navigation.diagnostics.diagnose_fb_walls import draw_environment
from fb_navigation.diagnostics.analyse_fb_rewards_failures import embed_batches, distribution, bellman_statistics


def spatial_regions(points, env):
    boundary = np.minimum(points, env.size-points).min(axis=-1)
    distance = np.full(len(points), np.inf)
    for w in env.wall_blocks:
        dx = np.maximum(np.maximum(w.xmin-points[:, 0], points[:, 0]-w.xmax), 0)
        dy = np.maximum(np.maximum(w.ymin-points[:, 1], points[:, 1]-w.ymax), 0)
        distance = np.minimum(distance, np.hypot(dx, dy))
    threshold = .05+3*env._noise
    near_wall, near_boundary = distance <= threshold, boundary <= threshold
    labels = np.full(len(points), 'interior', dtype='<U24')
    labels[near_wall] = 'internal_wall'
    labels[near_boundary] = 'outer_boundary'
    labels[near_wall & near_boundary] = 'wall_and_boundary'
    return labels, distance, boundary


def free_points(points, env):
    free = np.all((points >= 0)&(points <= env.size), axis=1)
    for w in env.wall_blocks:
        free &= ~((points[:, 0] >= w.xmin)&(points[:, 0] <= w.xmax)&(points[:, 1] >= w.ymin)&(points[:, 1] <= w.ymax))
    return points[free]


def make_cases(env, pair_path, wall_path, grid, probes, limit, seed):
    source = np.load(pair_path)
    axis = (np.arange(grid)+.5)/grid
    xx, yy = np.meshgrid(axis, axis)
    points = [np.column_stack((xx.ravel(), yy.ravel()))]
    for side in (0., 1.):
        points += [np.column_stack((np.repeat(side, grid), axis)), np.column_stack((axis, np.repeat(side, grid)))]
    # Explicit probes .002 outside each wall face, including near door mouths.
    for w in env.wall_blocks:
        x = np.linspace(w.xmin, w.xmax, probes+2)[1:-1]
        y = np.linspace(w.ymin, w.ymax, probes+2)[1:-1]
        for edge in (w.xmin-.002, w.xmax+.002):
            points.append(np.column_stack((np.repeat(edge, len(y)), y)))
        for edge in (w.ymin-.002, w.ymax+.002):
            points.append(np.column_stack((x, np.repeat(edge, len(x)))))
    points = np.unique(free_points(np.concatenate(points), env).astype(np.float32), axis=0)
    all_goals = source['goals']
    centers = np.asarray([[.275, .175], [.275, .725], [.825, .175], [.825, .725]])
    goals = np.asarray([all_goals[np.argmin(np.sum((all_goals-c)**2, axis=1))] for c in centers])
    starts = np.tile(points, (len(goals), 1))
    targets = np.repeat(goals, len(points), axis=0)
    names = ['spatial_probe']*len(starts)
    starts = np.concatenate((starts, source['starts']))
    targets = np.concatenate((targets, all_goals)); names += ['original_pair']*len(all_goals)
    examples = json.loads(Path(wall_path).read_text())['examples']
    special = []
    for e in examples:
        special.append(len(starts))
        starts = np.vstack((starts, e['trapped_position'])).astype(np.float32)
        targets = np.vstack((targets, e['goal'])).astype(np.float32)
        names.append(e['snapshot']+'_'+e['outcome'])
    if limit and limit < len(starts):
        rng = np.random.default_rng(seed)
        selected = np.concatenate((special[:min(len(special), limit)],
                                   rng.choice(len(starts)-len(special), max(0, limit-len(special)), replace=False)))
        starts, targets = starts[selected], targets[selected]
        names = [names[i] for i in selected]
        special = list(range(min(len(special), limit)))
    return starts.astype(np.float32), targets.astype(np.float32), np.asarray(names), np.asarray(special, int)


def detailed_collector(env, episodes):
    """Instrument the unchanged collector; equivalence is checked before use."""
    walls = tuple(env.wall_blocks)
    actions = jnp.asarray([(a.x, a.y) for a in env.actions], jnp.float32)
    def reset(key):
        key, pk = jax.random.split(key)
        p = _sample_point(pk, env.size, walls)
        key, gk = jax.random.split(key)
        g = _sample_point(gk, env.size, walls)
        return key, p, g
    def collect(ek, ak, lk):
        def episode(carry, _):
            ek, ak, lk = carry
            ek, p, g = reset(ek)
            lk, _ = jax.random.split(lk)
            def step(carry, _):
                ek, ak, p, g = carry
                ak, draw = jax.random.split(ak)
                a = jax.random.randint(draw, (), 0, len(actions))
                ek, mk = jax.random.split(ek)
                nxt, proposed, collision = _move(p, Point(*actions[a]), mk, env._noise, env.size, walls)
                clipped = ((proposed.x <= 0)|(proposed.x >= env.size)|(proposed.y <= 0)|(proposed.y >= env.size))
                row = dict(obs=jnp.asarray((p.x, p.y)), action=a, obs_next=jnp.asarray((nxt.x, nxt.y)),
                           next_actions=jnp.ones(len(actions), bool), collision=collision, boundary_clipped=clipped)
                done = nxt.is_close_to(g, diff=env.threshold_distance)
                ek, p, g = jax.lax.cond(done, reset, lambda key: (key, nxt, g), ek)
                return (ek, ak, p, g), row
            (ek, ak, _, _), rows = jax.lax.scan(step, (ek, ak, p, g), None, length=30)
            return (ek, ak, lk), rows
        keys, rows = jax.lax.scan(episode, (ek, ak, lk), None, length=episodes)
        return keys, jax.tree_util.tree_map(lambda x: x.reshape((-1,)+x.shape[2:]), rows)
    return jax.jit(collect)


def reconstructed_key(seed, cycles):
    initial = jax.random.split(jax.random.PRNGKey(seed), 3)[0]
    @jax.jit
    def calculate(key):
        def cycle(key, _):
            key, _ = jax.lax.scan(lambda k, _: (jax.random.split(k, 2)[0], None), key, None, length=4)
            def update(k, _):
                return jax.random.split(jax.random.split(k, 3)[0], 3)[0], None
            key, _ = jax.lax.scan(update, key, None, length=40)
            return key, None
        return jax.lax.scan(cycle, key, None, length=cycles)[0]
    return np.asarray(calculate(initial))


def audit_replay(learner, env, requested, seed, out):
    training_seed = learner.config['seed']
    initial = (jax.random.PRNGKey(training_seed), jax.random.PRNGKey(training_seed),
               jax.random.split(jax.random.PRNGKey(training_seed), 3)[0])
    keys_ref, reference = jax.jit(make_collector(env, 4, 30))(*initial)
    keys_check, checked = detailed_collector(env, 4)(*initial)
    assert all(np.array_equal(a, b) for a, b in zip(keys_ref, keys_check))
    for k, v in reference.items():
        np.testing.assert_array_equal(v, checked[k])
    updates = int(learner.state['optim'][0])
    total = requested or (updates//40)*120
    if total < 30:
        raise ValueError('At least 30 collector transitions required')
    complete = total == (updates//40)*120
    print(f'Reconstructing {total:,} transitions using seed {training_seed}; retaining final replay', flush=True)
    _, rows = detailed_collector(env, (total+29)//30)(*initial)
    # Same population as the final ring buffer; chronological order is sufficient.
    rows = {k: np.asarray(v)[:total][-min(total, learner.buffer.max_size):] for k, v in rows.items()}
    key_matches = np.array_equal(reconstructed_key(training_seed, updates//40), np.asarray(learner.state['key']))
    assert not np.any(np.any(rows['obs'] != rows['obs_next'], axis=1) & rows['collision'])
    regions, _, _ = spatial_regions(rows['obs'], env)
    metrics = dict(source='reconstructed default schedule; historical replay was not saved',
                   training_seed=training_seed, inferred_training_updates=updates, inferred_collected_transitions=total,
                   retained_transitions=len(rows['action']), full_default_schedule=complete,
                   instrumented_collector_matches_original=True, checkpoint_learner_rng_matches_default_schedule=bool(key_matches),
                   collision_fraction=float(rows['collision'].mean()), boundary_clipped_fraction=float(rows['boundary_clipped'].mean()),
                   by_action=[], by_region=[])
    for a, name in enumerate(env.actions_str):
        take = rows['action'] == a
        metrics['by_action'].append(dict(action=name, count=int(take.sum()), collisions=int(rows['collision'][take].sum()),
                                        collision_rate=float(rows['collision'][take].mean()), boundary_clipped=int(rows['boundary_clipped'][take].sum())))
    for region in np.unique(regions):
        take = regions == region
        metrics['by_region'].append(dict(region=region, count=int(take.sum()), collision_rate=float(rows['collision'][take].mean()),
                                        boundary_clipped_rate=float(rows['boundary_clipped'][take].mean())))
    np.savez_compressed(out/'reconstructed_replay.npz', **rows, source=metrics['source'], training_seed=training_seed,
                        checkpoint_sha256=hashlib.sha256(Path(learner._audit_checkpoint).read_bytes()).hexdigest())
    rng = np.random.default_rng(seed+8)
    take = rng.choice(len(rows['action']), min(60000, len(rows['action'])), replace=False)
    b = embed_batches(learner, rows['obs'][take]).astype(np.float64)
    bt = embed_batches(learner, rows['obs'][take], target=True).astype(np.float64)
    covariance, cross = b.T@b/len(b), bt.T@b/len(b)
    metrics['covariance_reference_count'] = len(b)
    metrics['backward_covariance_eigenvalues'] = np.linalg.eigvalsh(covariance).tolist()
    print('Replay audit: '+json.dumps({k: v for k, v in metrics.items() if k != 'backward_covariance_eigenvalues'}), flush=True)
    return rows, covariance, cross, metrics


def local_coverage(rows, starts, actions, radius=.025):
    counts, collisions, clipped = [], [], []
    for a in range(actions):
        selected = rows['action'] == a
        def query(mask):
            if not mask.any():
                return np.zeros(len(starts), int)
            return cKDTree(rows['obs'][mask]).query_ball_point(starts, radius, return_length=True)
        counts.append(query(selected)); collisions.append(query(selected & rows['collision']))
        clipped.append(query(selected & rows['boundary_clipped']))
    return dict(local_count=np.stack(counts, axis=1), local_collision_count=np.stack(collisions, axis=1),
                local_clipped_count=np.stack(clipped, axis=1))


def make_expected_backup(learner, env, samples, covariance, cross):
    actions = jnp.asarray([(a.x, a.y) for a in env.actions])
    walls = tuple(env.wall_blocks)
    covariance, cross = jnp.asarray(covariance), jnp.asarray(cross)
    def move(p, a, key):
        nxt, proposed, hit = _move(Point(*p), Point(*actions[a]), key, env._noise, env.size, walls)
        return jnp.asarray((nxt.x, nxt.y)), hit, ((proposed.x <= 0)|(proposed.x >= 1)|(proposed.y <= 0)|(proposed.y >= 1))
    @jax.jit
    def evaluate(params, target, starts, goals, keys):
        n, na = len(starts), len(actions)
        features = learner.featuriser.transform(starts)
        z = learner.backward_network.apply(params['b'], learner.featuriser.transform(goals))
        b = learner.backward_network.apply(params['b'], features)
        fo = learner.forward_network.apply(params['f'], features, z)
        ft = learner.forward_network.apply(target['f'], features, z)
        qo, qt = jnp.einsum('nda,nd->na', fo, z), jnp.einsum('nda,nd->na', ft, z)
        cz, tz = z@covariance.T, z@cross.T
        qco, qct = jnp.einsum('nda,nd->na', fo, cz), jnp.einsum('nda,nd->na', ft, tz)
        drawkeys = jax.vmap(lambda k: jax.random.split(k, samples))(keys)
        # Common movement noise across all initial actions at each state.
        p = jnp.broadcast_to(starts[:, None, None, :], (n, na, samples, 2)).reshape(-1, 2)
        a = jnp.broadcast_to(jnp.arange(na)[None, :, None], (n, na, samples)).reshape(-1)
        k = jnp.broadcast_to(drawkeys[:, None], (n, na, samples, 2)).reshape(-1, 2)
        nxt, hit, clipped = jax.vmap(move)(p, a, k)
        z_repeat = jnp.broadcast_to(z[:, None, None, :], (n, na, samples, learner.embed_dim)).reshape(-1, learner.embed_dim)
        f = learner.forward_network.apply(target['f'], learner.featuriser.transform(nxt), z_repeat)
        q = jnp.einsum('nda,nd->na', f, z_repeat)
        pi = jax.nn.softmax(q/learner.temperature, axis=-1)
        expected_f = jnp.einsum('nda,na->nd', f, pi)
        v = jnp.einsum('nd,nd->n', expected_f, z_repeat).reshape(n, na, samples)
        vt = jnp.einsum('nd,nd->n', expected_f, z_repeat@cross.T).reshape(n, na, samples)
        reward = jnp.sum(b*z, axis=1)
        gamma = learner.discount_factor
        return dict(q_online=qo, q_target=qt, integrated_online=qco, integrated_target=qct,
                    reward=reward, backup=reward[:, None]+gamma*v.mean(axis=2),
                    backup_se=gamma*v.std(axis=2, ddof=1)/jnp.sqrt(samples),
                    integrated_backup=reward[:, None]+gamma*vt.mean(axis=2),
                    integrated_backup_se=gamma*vt.std(axis=2, ddof=1)/jnp.sqrt(samples),
                    collision_probability=hit.reshape(n, na, samples).mean(axis=2),
                    clipped_probability=clipped.reshape(n, na, samples).mean(axis=2))
    return evaluate


def bellman_evaluate(learner, env, starts, goals, samples, batch_size, covariance, cross, seed):
    fn = make_expected_backup(learner, env, samples, covariance, cross)
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(seed), i))(jnp.arange(len(starts)))
    batches = []
    for first in range(0, len(starts), batch_size):
        end = min(first+batch_size, len(starts))
        batch = fn(learner.state['params'], learner.state['target'], jnp.asarray(starts[first:end]), jnp.asarray(goals[first:end]), keys[first:end])
        batches.append(jax.tree_util.tree_map(np.asarray, batch))
        if (first//batch_size) % 40 == 0:
            print(f'Expected backups: {end}/{len(starts)} cases, {samples} noise draws/action', flush=True)
    return {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}


def select_mc_cases(regions, special, count, seed):
    rng = np.random.default_rng(seed)
    selected = list(special[:min(len(special), count)])
    while len(selected) < count:
        progress = False
        for region in ('internal_wall', 'outer_boundary', 'interior', 'wall_and_boundary'):
            available = np.setdiff1d(np.flatnonzero(regions == region), selected)
            if len(available) and len(selected) < count:
                selected.append(int(rng.choice(available))); progress = True
        if not progress:
            break
    return np.asarray(selected)


def make_returns(learner, env, horizon):
    actions = jnp.asarray([(a.x, a.y) for a in env.actions])
    walls = tuple(env.wall_blocks)
    @jax.jit
    def returns(params, target, starts, z, initial_actions, keys):
        def step(carry, t):
            p, keys, total, middle, hits, clips = carry
            split = jax.vmap(lambda k: jax.random.split(k, 3))(keys)
            keys, ak, mk = split[:, 0], split[:, 1], split[:, 2]
            features = learner.featuriser.transform(p)
            b = learner.backward_network.apply(params['b'], features)
            reward = jnp.sum(b*z, axis=1)
            total += (learner.discount_factor**t)*reward
            middle = jnp.where(t == horizon//2-1, total, middle)
            f = learner.forward_network.apply(target['f'], features, z)
            q = jnp.einsum('nda,nd->na', f, z)
            chosen = jax.vmap(lambda k, scores: jax.random.categorical(k, scores/learner.temperature))(ak, q)
            chosen = jnp.where(t == 0, initial_actions, chosen)
            def move(p, a, key):
                nxt, proposed, collision = _move(Point(*p), Point(*actions[a]), key, env._noise, env.size, walls)
                clip = (proposed.x <= 0)|(proposed.x >= 1)|(proposed.y <= 0)|(proposed.y >= 1)
                return jnp.asarray((nxt.x, nxt.y)), collision, clip
            p, collided, clipped = jax.vmap(move)(p, chosen, mk)
            return (p, keys, total, middle, hits+collided, clips+clipped), None
        n = len(starts)
        zeros, counts = jnp.zeros(n), jnp.zeros(n, jnp.int32)
        (_, _, total, middle, hits, clips), _ = jax.lax.scan(step, (starts, keys, zeros, zeros, counts, counts), jnp.arange(horizon))
        return dict(returns=total, half_horizon_returns=middle, wall_collisions=hits, boundary_clipped=clips)
    return returns


def monte_carlo(learner, env, starts, goals, selected, rollouts, horizon, batch_size, seed, out):
    fn = make_returns(learner, env, horizon)
    z = learner.embed_B(jnp.asarray(goals[selected]))
    batches = []
    for first in range(0, len(selected), batch_size):
        last = min(first+batch_size, len(selected))
        ids = selected[first:last]; n = len(ids); na = learner.n_actions
        keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(seed), i))(
            jnp.asarray(ids[:, None]*rollouts+np.arange(rollouts)[None]).reshape(-1))
        keys = jnp.broadcast_to(keys.reshape(n, 1, rollouts, 2), (n, na, rollouts, 2)).reshape(-1, 2)
        p = jnp.repeat(jnp.asarray(starts[ids]), na*rollouts, axis=0)
        task = jnp.repeat(z[first:last], na*rollouts, axis=0)
        actions = jnp.broadcast_to(jnp.arange(na)[None, :, None], (n, na, rollouts)).reshape(-1)
        result = fn(learner.state['params'], learner.state['target'], p, task, actions, keys)
        result = {k: np.asarray(v).reshape(n, na, rollouts) for k, v in result.items()}
        batches.append(result)
        print(f'MC returns completed: {last}/{len(selected)} cases x {na} actions x {rollouts} rollouts x {horizon} steps', flush=True)
    result = {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
    np.savez_compressed(out/'mc_returns.npz', selected_case_ids=selected, **result)
    return result


def summarise_backup(results, coverage, regions):
    summaries = []
    selectors = {'all': np.ones_like(results['q_online'], bool)}
    selectors.update({name: np.broadcast_to((regions == name)[:, None], results['q_online'].shape) for name in np.unique(regions)})
    selectors['collision_probability_ge_half'] = results['collision_probability'] >= .5
    selectors['clipping_probability_ge_half'] = results['clipped_probability'] >= .5
    for region, take in selectors.items():
        if not take.any():
            continue
        for prediction, target in (('q_online', 'backup'), ('q_target', 'backup'),
                                    ('integrated_online', 'integrated_backup'), ('integrated_target', 'integrated_backup')):
            summaries.append(dict(region=region, prediction=prediction, count=int(take.sum()),
                **bellman_statistics(results[prediction][take], results[target][take]),
                mean_backup_integration_se=float(results['backup_se'][take].mean()),
                local_count=distribution(coverage['local_count'][take]),
                local_collision_count=distribution(coverage['local_collision_count'][take])))
    return summaries


def write_outputs(out, env, starts, goals, source, regions, results, coverage, mc_ids, mc, audit, covariance, cross, args, digest):
    summaries = summarise_backup(results, coverage, regions)
    mean = mc['returns'].mean(axis=2)
    se = mc['returns'].std(axis=2, ddof=1)/np.sqrt(args.mc_rollouts)
    comparisons = []
    for name in ('q_online', 'q_target', 'integrated_online', 'integrated_target'):
        predicted = results[name][mc_ids]
        delta = predicted-mean
        comparisons.append(dict(prediction=name, **bellman_statistics(predicted, mean),
            mc_return=distribution(mean), predicted_value=distribution(predicted),
            mean_mc_se=float(se.mean()), outside_mc95_fraction=float(np.mean(np.abs(delta) > 1.96*se)),
            greedy_action_agreement=float(np.mean(predicted.argmax(axis=1) == mean.argmax(axis=1)))))
    metrics = dict(checkpoint=args.checkpoint, checkpoint_sha256=digest, seed=args.seed,
        dynamics=dict(step=.05, noise=env._noise, discount=.99, temperature=200, goal_terminated=False),
        cases=len(starts), actions=env.actions_str, noise_samples_per_action=args.noise_samples,
        spatial_region_distance_cutoff=.05+3*env._noise,
        replay=audit, bellman=summaries,
        mc=dict(cases=len(mc_ids), rollouts_per_case_action=args.mc_rollouts, horizon=args.mc_horizon,
                comparisons=comparisons, half_horizon_change=distribution(mean-mc['half_horizon_returns'].mean(axis=2))),
        notes=['Original replay is absent. Replay is reconstructed under the default collector schedule inferred from Adam update count, seed and runner defaults.',
               'Collector instrumentation matches the original 120-transition batch and final collector keys exactly. The saved learner RNG is checked against the inferred update schedule.',
               'Historical replay identity cannot be guaranteed without original environment/action RNGs, schedule and backend provenance.',
               'Expected Bellman backups sample the full noisy _move distribution, including wall rejection and outer clipping; no collision conditioning.',
               'Training/evaluation policy is target softmax at T=200 over all four actions. No geometry-based mask is added.',
               'Reward is online B(s) dot online B(goal), counted at t=0 before the forced initial action. MC never stops or resets at goal entry.',
               'For this reward, the integrated prediction is F_online^T E[B_online B_online^T] B(goal). Target predictions use E[B_target B_online^T].',
               'MC is independent of predicted F values except for selecting the fixed target-softmax policy; no value bootstrapping is used.',
               'Finite-horizon MC tail size is assessed using reward-map extrema and half-horizon comparison; the grid-based tail bound is not a rigorous global network bound.'])
    try:
        mapped = np.load('evaluation_fb_reward_failures/b_reward_maps.npz')
        maxima = np.max(np.abs(mapped['reward']), axis=0)
        ids = np.argmin(np.sum((goals[mc_ids, None]-mapped['goals'][None])**2, axis=2), axis=1)
        estimates = maxima[ids]*(.99**args.mc_horizon)/(1-.99)
        metrics['mc']['grid_based_absolute_tail_bound_estimate'] = distribution(estimates)
    except FileNotFoundError:
        pass
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(out/'expected_bellman.npz', starts=starts, goals=goals, source=source, region=regions,
                        covariance=covariance, target_online_cross=cross, **results, **coverage)
    with (out/'action_values.csv').open('w', newline='') as f:
        names = ['case', 'source', 'region', 'x', 'y', 'goal_x', 'goal_y', 'action', 'q_online', 'q_target',
                 'integrated_online', 'integrated_target', 'reward', 'backup', 'backup_se', 'integrated_backup',
                 'collision_probability', 'clipped_probability', *coverage]
        writer = csv.DictWriter(f, fieldnames=names); writer.writeheader()
        for i in range(len(starts)):
            for a, name in enumerate(env.actions_str):
                row = dict(case=i, source=source[i], region=regions[i], x=starts[i, 0], y=starts[i, 1], goal_x=goals[i, 0], goal_y=goals[i, 1], action=name,
                           reward=float(results['reward'][i]))
                row.update({k: float(results[k][i, a]) for k in names if k in results and k != 'reward'})
                row.update({k: int(v[i, a]) for k, v in coverage.items()}); writer.writerow(row)
    with (out/'mc_comparison.csv').open('w', newline='') as f:
        fields = ['case', 'region', 'action', 'mc_mean', 'mc_se', 'half_horizon_mean', 'q_online', 'q_target', 'integrated_online', 'integrated_target']
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for j, i in enumerate(mc_ids):
            for a, name in enumerate(env.actions_str):
                writer.writerow(dict(case=i, region=regions[i], action=name, mc_mean=mean[j, a], mc_se=se[j, a],
                    half_horizon_mean=mc['half_horizon_returns'][j, a].mean(),
                    **{k: results[k][i, a] for k in fields[-4:]}))
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for ax, name in zip(axes.ravel(), ('q_online', 'q_target', 'integrated_online', 'integrated_target')):
        predicted = results[name][mc_ids]
        ax.errorbar(mean.ravel(), predicted.ravel(), xerr=1.96*se.ravel(), fmt='.', ms=5, alpha=.65)
        lo, hi = min(mean.min(), predicted.min()), max(mean.max(), predicted.max())
        ax.plot([lo, hi], [lo, hi], 'k--'); ax.set(xlabel='Monte Carlo return (95% error bars)', ylabel=name, title=name)
    fig.savefig(out/'q_vs_monte_carlo.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, values, title in zip(axes, (np.abs(results['q_target']-results['backup']).mean(axis=1), coverage['local_count'].mean(axis=1), results['collision_probability'].mean(axis=1)),
                                ('Mean absolute target-Q Bellman residual', 'Local replay state-action count (radius .025)', 'Noisy wall-collision probability')):
        # Plot only first canonical-goal spatial probes to avoid overplotting goals.
        selected = (source == 'spatial_probe')&np.all(goals == goals[0], axis=1)
        im = ax.scatter(*starts[selected].T, c=values[selected], s=14, cmap='magma')
        draw_environment(ax, env); ax.set_title(title, fontsize=9); fig.colorbar(im, ax=ax)
    fig.savefig(out/'spatial_residual_and_coverage.png', dpi=150); plt.close(fig)
    report = ['# FB all-action value audit', '', f'Checkpoint: {args.checkpoint}; no training. All actions use the target-softmax policy (T=200), gamma=.99.', '',
        f'{len(starts)} state-goal cases, all four initial actions, {args.noise_samples} independent movement-noise samples per backup.',
        'Expected next values include both collisions and movement; there is no conditioning on blocked outcomes.', '',
        '## Replay coverage', '', json.dumps(audit, indent=2), '',
        '## Noise-averaged Bellman residuals', '',
        '| Region | Target-Q MAE | Median relative error | Local samples, median |', '|---|---:|---:|---:|']
    for row in summaries:
        if row['prediction'] == 'q_target':
            report.append(f'| {row["region"]} | {row["mae"]:.2f} | {row["relative_abs_residual"]["median"]:.1%} | {row["local_count"]["median"]:.0f} |')
    report += ['', '## Independent Monte Carlo returns', '',
        f'{len(mc_ids)} stratified configurations, all four initial actions, {args.mc_rollouts} independent rollouts each, {args.mc_horizon} steps.',
        'Returns count the current-state reward before movement and continue after reaching goals. MC confidence intervals exclude finite-horizon bias.', '',
        '| Prediction | MAE vs MC | Mean signed error | MC SE, mean | Outside MC 95% interval |', '|---|---:|---:|---:|---:|']
    for c in comparisons:
        report.append(f'| {c["prediction"]} | {c["mae"]:.2f} | {c["mean_signed_residual"]:.2f} | {c["mean_mc_se"]:.2f} | {c["outside_mc95_fraction"]:.1%} |')
    report += ['', *metrics['notes'], '', 'Run: python evaluate_fb_action_values.py.',
        'Smoke: --case-limit 24 --grid 6 --wall-probes 2 --replay-transitions 6000 --noise-samples 32 --mc-cases 4 --mc-rollouts 16 --mc-horizon 100 --output evaluation_fb_action_smoke.']
    (out/'RESULTS.md').write_text('\n'.join(report))
    return metrics


def run(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    learner = FBLearner.from_checkpoint(args.checkpoint, device=args.device, buffer_device='cpu')
    if learner.goal_terminated or learner.z_sampling != 'cauchy':
        raise ValueError('This audit targets the original continuing Cauchy checkpoint')
    learner._audit_checkpoint = args.checkpoint
    print(f'Backend: {learner.device}; no learner updates', flush=True)
    with jax.default_device(learner.device):
        env = ContinuousFourRooms(include_stay_action=False)
        starts, goals, source, special = make_cases(env, args.pairs, args.wall_metrics, args.grid, args.wall_probes, args.case_limit, args.seed)
        regions, _, _ = spatial_regions(starts, env)
        rows, covariance, cross, audit = audit_replay(learner, env, args.replay_transitions, args.seed, out)
        coverage = local_coverage(rows, starts, learner.n_actions)
        results = bellman_evaluate(learner, env, starts, goals, args.noise_samples, args.bellman_batch_cases, covariance, cross, args.seed+11)
        # Save the completed expensive phase before starting independent rollouts.
        np.savez_compressed(out/'expected_bellman.npz', starts=starts, goals=goals, source=source, region=regions,
                            covariance=covariance, target_online_cross=cross, **results, **coverage)
        (out/'replay_audit.json').write_text(json.dumps(audit, indent=2))
        ids = select_mc_cases(regions, special, args.mc_cases, args.seed+31)
        mc = monte_carlo(learner, env, starts, goals, ids, args.mc_rollouts, args.mc_horizon, args.mc_batch_cases, args.seed+41, out)
        metrics = write_outputs(out, env, starts, goals, source, regions, results, coverage, ids, mc, audit, covariance, cross, args, digest)
    assert hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() == digest
    print(json.dumps(metrics['mc'], indent=2), flush=True)
    return metrics


def postprocess(args):
    """Validate noise integration and summarise saved MC without repeating it."""
    out = Path(args.output)
    metrics = json.loads((out/'metrics.json').read_text())
    digest = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    if digest != metrics['checkpoint_sha256']:
        raise ValueError('Saved metrics and current checkpoint differ')
    data, mc = np.load(out/'expected_bellman.npz'), np.load(out/'mc_returns.npz')
    selected = mc['selected_case_ids']
    learner = FBLearner.from_checkpoint(args.checkpoint, device=args.device, buffer_device='cpu')
    env = ContinuousFourRooms(include_stay_action=False)
    with jax.default_device(learner.device):
        refined = bellman_evaluate(learner, env, data['starts'][selected], data['goals'][selected],
            args.validation_noise_samples, args.bellman_batch_cases, data['covariance'], data['target_online_cross'], args.seed+121)
    difference = refined['backup']-data['backup'][selected]
    validation = dict(configurations=len(selected), samples_per_action=args.validation_noise_samples,
        mean_absolute_backup_change=float(np.abs(difference).mean()), p95_absolute_backup_change=float(np.quantile(np.abs(difference), .95)),
        initial_mean_integration_se=float(data['backup_se'][selected].mean()), refined_mean_integration_se=float(refined['backup_se'].mean()),
        target_q_mae=bellman_statistics(data['q_target'][selected], refined['backup'])['mae'])
    metrics['noise_integration_validation'] = validation
    coverage = {}
    for name, mask, field in (('wall_collision_actions', data['collision_probability'] >= .5, 'local_collision_count'),
                              ('boundary_clipped_actions', data['clipped_probability'] >= .5, 'local_clipped_count')):
        count = data[field][mask]
        coverage[name] = dict(count=int(mask.sum()), local_event_samples=distribution(count),
                              no_local_event_samples_fraction=float(np.mean(count == 0)),
                              fewer_than_10_local_event_samples_fraction=float(np.mean(count < 10)))
    metrics['local_replay_event_coverage'] = coverage
    with np.load(out/'reconstructed_replay.npz') as replay:
        wall = replay['collision']; clipped = replay['boundary_clipped']
        metrics['replay']['accepted_boundary_clipped_count'] = int(np.sum(clipped & ~wall))
        metrics['replay']['clipped_proposal_then_wall_rejection_count'] = int(np.sum(clipped & wall))
    returns = mc['returns']; mean = returns.mean(axis=2); se = returns.std(axis=2, ddof=1)/np.sqrt(returns.shape[2])
    candidates = np.flatnonzero(data['source'][selected] == 'target_rescued')
    j = int(candidates[0]) if len(candidates) else 0
    case = int(selected[j])
    worked = dict(case=case, state=data['starts'][case].tolist(), goal=data['goals'][case].tolist(), actions=[])
    for a, name in enumerate(env.actions_str):
        worked['actions'].append(dict(action=name, q_target=float(data['q_target'][case, a]),
            q_online=float(data['q_online'][case, a]), integrated_online=float(data['integrated_online'][case, a]),
            expected_backup=float(refined['backup'][j, a]), residual=float(data['q_target'][case, a]-refined['backup'][j, a]),
            mc_mean=float(mean[j, a]), mc_se=float(se[j, a]),
            local_count=int(data['local_count'][case, a]), local_collision_count=int(data['local_collision_count'][case, a])))
    metrics['worked_example'] = worked
    # Paired action comparisons use the common-random-number rollouts, not independent-action SEs.
    rankings = []
    for prediction in ('q_online', 'q_target', 'integrated_online', 'integrated_target'):
        greedy = data[prediction][selected].argmax(axis=1)
        best_mc = mean.argmax(axis=1)
        gaps = returns[np.arange(len(selected)), best_mc]-returns[np.arange(len(selected)), greedy]
        gap_mean, gap_se = gaps.mean(axis=1), gaps.std(axis=1, ddof=1)/np.sqrt(returns.shape[2])
        rankings.append(dict(prediction=prediction, point_action_agreement=float(np.mean(greedy == best_mc)),
                             fraction_mc_best_significantly_better_than_predicted_choice=float(np.mean(gap_mean > 1.96*gap_se))))
    metrics['mc']['paired_action_ranking'] = rankings
    np.savez_compressed(out/'refined_expected_backups.npz', selected_case_ids=selected, **refined)
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2))
    overall = next(x for x in metrics['bellman'] if x['region'] == 'all' and x['prediction'] == 'q_target')
    comparisons = {x['prediction']: x for x in metrics['mc']['comparisons']}
    addition = ['', '## Integration convergence and local event coverage', '', json.dumps(validation, indent=2), '',
        json.dumps(coverage, indent=2), '',
        'Coverage counts use a radius .025 around each evaluated state, restricted to its chosen action. Event counts include the actual noisy collision or clipping outcome.',
        f'Of {round(metrics["replay"]["boundary_clipped_fraction"]*metrics["replay"]["retained_transitions"]):,} clipped proposals, {metrics["replay"]["accepted_boundary_clipped_count"]:,} are accepted boundary-clipped transitions and {metrics["replay"]["clipped_proposal_then_wall_rejection_count"]:,} are subsequently rejected by walls.',
        'Broad collision coverage rules out a simple absence of these transitions under the reconstructed schedule; it does not prove adequate coverage at every relevant resolution or task embedding.', '',
        '## Worked configuration: all four actions', '', f'State {worked["state"]}, goal {worked["goal"]}.', '',
        '| Action | Target Q | Full expected backup | Residual | MC return | MC SE |', '|---|---:|---:|---:|---:|---:|']
    for row in worked['actions']:
        addition.append(f'| {row["action"]} | {row["q_target"]:.2f} | {row["expected_backup"]:.2f} | {row["residual"]:.2f} | {row["mc_mean"]:.2f} | {row["mc_se"]:.2f} |')
    addition += ['', 'One-step residuals of a few percent can coexist with large long-run value error at gamma=.99. The independent MC return comparison is the stronger calibration test.',
                 f'Mean noise-integration SE is {overall["mean_backup_integration_se"]:.2f}, compared with target-Q Bellman MAE {overall["mae"]:.2f}. Refinement changes backups by {validation["mean_absolute_backup_change"]:.2f} on average.',
                 f'Across {mean.size} MC case-action estimates, raw online/target MAEs are {comparisons["q_online"]["mae"]:.2f}/{comparisons["q_target"]["mae"]:.2f}; average MC SE is {se.mean():.2f}.',
                 f'Covariance-integrated online/target MAEs are {comparisons["integrated_online"]["mae"]:.2f}/{comparisons["integrated_target"]["mae"]:.2f}.',
                 'Grid-based tail estimates are saved in metrics.json. They are empirical estimates, not formal bounds on all continuous states.',
                 f'These {len(selected)} diagnostic configurations do not form an unbiased estimate over all possible tasks/states.',
                 'No conclusions about successor-measure accuracy are based on training loss or navigation success.', '',
                 'All three dedicated regression tests passed: noisy clipped-Gaussian integration, original-collector equivalence, and forced-action/current-state MC indexing.',
                 'Refine and append this analysis: python evaluate_fb_action_values.py --postprocess.']
    report_path = out/'RESULTS.md'
    original = report_path.read_text().split('\n## Integration convergence')[0]
    report_path.write_text(original+'\n'+'\n'.join(addition))
    fig, axes = plt.subplots(4, 2, figsize=(10, 16), constrained_layout=True)
    take = (data['source'] == 'spatial_probe')&np.all(data['goals'] == data['goals'][0], axis=1)
    for a, name in enumerate(env.actions_str):
        residual = data['q_target'][take, a]-data['backup'][take, a]
        for ax, values, title in ((axes[a, 0], residual, name+': signed expected Bellman residual'),
                                 (axes[a, 1], data['collision_probability'][take, a], name+': wall-collision probability')):
            im = ax.scatter(*data['starts'][take].T, c=values, cmap='coolwarm' if ax == axes[a, 0] else 'magma', s=14)
            draw_environment(ax, env); ax.set_title(title, fontsize=9); fig.colorbar(im, ax=ax)
    fig.savefig(out/'bellman_by_action.png', dpi=145); plt.close(fig)
    print(json.dumps(dict(integration_validation=validation, local_event_coverage=coverage, worked_example=worked), indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='models/fb_fourrooms_jax_original.pkl')
    p.add_argument('--pairs', default='evaluation_original_short_horizon/trials.npz')
    p.add_argument('--wall-metrics', default='evaluation_original_walls/metrics.json')
    p.add_argument('--output', default='evaluation_fb_action_values')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=2033)
    p.add_argument('--grid', type=int, default=20)
    p.add_argument('--wall-probes', type=int, default=8)
    p.add_argument('--case-limit', type=int, default=0)
    p.add_argument('--replay-transitions', type=int, default=0)
    p.add_argument('--noise-samples', type=int, default=256)
    p.add_argument('--bellman-batch-cases', type=int, default=8)
    p.add_argument('--mc-cases', type=int, default=24)
    p.add_argument('--mc-rollouts', type=int, default=256)
    p.add_argument('--mc-horizon', type=int, default=1000)
    p.add_argument('--mc-batch-cases', type=int, default=4)
    p.add_argument('--postprocess', action='store_true')
    p.add_argument('--validation-noise-samples', type=int, default=2048)
    args = p.parse_args()
    if args.postprocess:
        postprocess(args)
    else:
        run(args)
