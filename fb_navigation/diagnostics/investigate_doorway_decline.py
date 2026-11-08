"""Paired checkpoint investigation; never trains or changes checkpoint files."""
import argparse
import json
from pathlib import Path
import hashlib
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms
from fb_navigation.environments.continuous_four_rooms_jax import Point, _move
from fb_navigation.evaluation.fourrooms import coordinate_room, CATEGORIES
from fb_navigation.diagnostics.control_bfs_bottlenecks import build_graph, bfs_distances
from fb_navigation.agents.forward_backward_jax import FBLearner


def trace_function(learner, env, horizon=150, mask_wall_actions=False, epsilon=.02):
    actions = jnp.array([(a.x, a.y) for a in env.actions])
    walls = tuple(env.wall_blocks)
    @jax.jit
    def trace(params, starts, goals, keys):
        z = learner.backward_network.apply(params['b'], learner.featuriser.transform(goals))
        n = len(starts)
        def step(carry, t):
            position, hit, keys = carry
            split = jax.vmap(lambda k: jax.random.split(k, 4))(keys)
            keys, ek, ak, mk = [split[:, i] for i in range(4)]
            f = learner.forward_network.apply(params['f'], learner.featuriser.transform(position), z)
            q = jnp.einsum('bda,bd->ba', f, z)
            action_scores = q
            if mask_wall_actions:
                def wall_mask(p):
                    return jax.vmap(lambda a: _move(Point(*p), Point(*a), jax.random.PRNGKey(0),
                        0., env.size, walls)[2])(actions)
                blocked = jax.vmap(wall_mask)(position)
                action_scores = jnp.where(blocked, -jnp.inf, q)
                action_scores = jnp.where(jnp.all(blocked, axis=1)[:, None], q, action_scores)
            explore = jax.vmap(lambda k: jax.random.uniform(k) < epsilon)(ek)
            random_action = jax.vmap(lambda k: jax.random.randint(k, (), 0, 4))(ak)
            selected = jnp.where(explore, random_action, jnp.argmax(action_scores, axis=1))
            def move(p, a, key):
                nxt, _, blocked = _move(Point(*p), Point(*actions[a]), key, env._noise, env.size, walls)
                return jnp.array([nxt.x, nxt.y]), blocked
            proposed, blocked = jax.vmap(move)(position, selected, mk)
            active = hit > horizon
            reached = jnp.all(abs(proposed-goals) <= env.threshold_distance, axis=1)
            following = jnp.where(active[:, None], proposed, position)
            hit = jnp.where(active & reached, t+1, hit)
            return (following, hit, keys), dict(position=position, following=following,
                action=selected, blocked=blocked & active, active=active, q=q)
        final, records = jax.lax.scan(step, (starts, jnp.full(n, horizon+1), keys), jnp.arange(horizon))
        return dict(records, hits=final[1])
    return trace


def traces(learner, env, pairs, mask_wall_actions=False):
    fn = trace_function(learner, env, mask_wall_actions=mask_wall_actions)
    blocks = []
    for first in range(0, len(pairs['starts']), 100):
        ids = jnp.arange(first, min(first+100, len(pairs['starts'])), dtype=jnp.uint32)
        keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(int(pairs['trial_seed'])), i))(ids)
        blocks.append(jax.device_get(fn(learner.state['params'], jnp.asarray(pairs['starts'][ids]),
                                       jnp.asarray(pairs['goals'][ids]), keys)))
    return {k: np.concatenate([b[k] for b in blocks], axis=0 if k == 'hits' else 1) for k in blocks[0]}


def door_rectangles(scale):
    h = scale/2
    return np.array([[6, 1.5-h, 7, 1.5+h], [6, 8.5-h, 7, 8.5+h],
                     [2.5-h, 4, 2.5+h, 5], [9.5-h, 4, 9.5+h, 5]])/11


def room_distance(a, b):
    difference = a ^ b
    return int(difference & 1 != 0)+int(difference & 2 != 0)


def failure_stages(trace, pairs, horizon, scale, approach=.05):
    rectangles = door_rectangles(scale)
    connections = [(0, 2), (1, 3), (0, 1), (2, 3)]
    output = []
    shape = trace['following'].shape[:2]
    all_rooms = np.asarray(coordinate_room(jnp.asarray(trace['following'].reshape(-1, 2)))).reshape(shape)
    for i, (start_room, goal_room) in enumerate(zip(pairs['start_rooms'], pairs['goal_rooms'])):
        steps = min(int(trace['hits'][i]), horizon)
        initial_distance = room_distance(start_room, goal_room)
        committed, best_distance = int(start_room), initial_distance
        approached, after_approached, crossings, first_cross = False, False, 0, None
        run, longest, repeated, post_cross_longest = 0, 0, 0, 0
        for t in range(steps):
            p = trace['position'][t, i]
            for rectangle, (a, b) in zip(rectangles, connections):
                if committed not in (a, b):
                    continue
                other = b if committed == a else a
                if room_distance(other, goal_room) >= room_distance(committed, goal_room):
                    continue
                nearest = np.clip(p, rectangle[:2], rectangle[2:])
                if np.linalg.norm(p-nearest) <= approach+1e-7:
                    approached = True
                    if first_cross is not None:
                        after_approached = True
            current = int(all_rooms[t, i])
            if current >= 0 and current != committed:
                crossings += 1
                d = room_distance(current, goal_room)
                if d < best_distance:
                    approached = True
                    if first_cross is None:
                        first_cross = t+1
                    best_distance = d
                committed = current
            blocked = bool(trace['blocked'][t, i])
            same = t > 0 and trace['blocked'][t-1, i] and trace['action'][t, i] == trace['action'][t-1, i]
            run = run+1 if blocked and same else 1 if blocked else 0
            repeated += int(blocked and same)
            longest = max(longest, run)
            if first_cross is not None and t+1 > first_cross:
                post_cross_longest = max(post_cross_longest, run)
        success = int(trace['hits'][i]) <= horizon
        if success:
            stage = 'success'
        elif first_cross is None:
            stage = 'approached_no_progress_crossing' if approached else 'never_approached_useful_door'
        elif best_distance == 0:
            stage = 'entered_goal_room_failed_goal_control'
        elif after_approached:
            stage = 'crossed_then_approached_next_door_no_completion'
        else:
            stage = 'crossed_then_never_approached_next_door'
        output.append(dict(stage=stage, success=success, steps=steps,
            wall_actions=int(trace['blocked'][:steps, i].sum()), repeated_wall_actions=repeated,
            longest_identical_wall_run=longest, crossings=crossings,
            longest_post_cross_wall_run=post_cross_longest,
            first_progress_crossing=first_cross, best_room_distance=best_distance))
    return output


def fixed_probes(trace400, trace800, count=16):
    rng = np.random.default_rng(901)
    probes = []
    for i in range(len(trace400['hits'])):
        states = []
        for trace in (trace400, trace800):
            steps = min(int(trace['hits'][i]), 30)
            states.extend(trace['position'][rng.integers(steps, size=count//2), i])
        probes.append(states)
    return np.asarray(probes, np.float32)


def nominal_transitions(env, states):
    actions = jnp.array([(a.x, a.y) for a in env.actions])
    def move(p, a):
        nxt, _, blocked = _move(Point(*p), Point(*a), jax.random.PRNGKey(0), 0., env.size, tuple(env.wall_blocks))
        return jnp.array([nxt.x, nxt.y]), blocked
    fn = jax.jit(jax.vmap(lambda p: jax.vmap(lambda a: move(p, a))(actions)))
    return tuple(np.asarray(v) for v in fn(jnp.asarray(states)))


def probe_metrics(learner, env, probes, pairs, transitions, blocked, other_obs):
    """Common state/action/goal probes; online vs target FB bootstrap and Q drift."""
    n, per, _ = probes.shape
    states = probes.reshape(-1, 2)
    goals = np.repeat(pairs['goals'], per, axis=0)
    chosen = np.random.default_rng(905).integers(4, size=len(states))
    following = transitions[np.arange(len(states)), chosen]
    feature = learner.featuriser.transform
    @jax.jit
    def calculate(state, goal, nxt, act, other):
        p, target = learner.state['params'], learner.state['target']
        z = learner.backward_network.apply(p['b'], feature(goal))
        f = learner.forward_network.apply(p['f'], feature(state), z)
        q = jnp.einsum('bda,bd->ba', f, z)
        target_f = learner.forward_network.apply(target['f'], feature(state), z)
        target_q = jnp.einsum('bda,bd->ba', target_f, z)
        next_f = learner.forward_network.apply(target['f'], feature(nxt), z)
        pi = jax.nn.softmax(jnp.einsum('bda,bd->ba', next_f, z)/learner.temperature, axis=1)
        average_f = jnp.einsum('bda,ba->bd', next_f, pi)
        selected = jnp.take_along_axis(f, act[:, None, None], axis=-1).squeeze(-1)
        value = selected @ learner.backward_network.apply(p['b'], feature(other)).T
        target_value = learner.discount_factor*(average_f @ learner.backward_network.apply(target['b'], feature(other)).T)
        return q, target_q, jnp.mean((value-target_value)**2, axis=1), jnp.mean(target_value**2, axis=1), jnp.linalg.norm(z, axis=1)
    parts = []
    for first in range(0, len(states), 128):
        sl = slice(first, first+128)
        parts.append(tuple(np.asarray(v) for v in calculate(jnp.asarray(states[sl]), jnp.asarray(goals[sl]),
            jnp.asarray(following[sl]), jnp.asarray(chosen[sl]), jnp.asarray(other_obs))))
    q, tq, mse, target_mse, znorm = [np.concatenate([p[k] for p in parts]) for k in range(5)]
    greedy = q.argmax(axis=1)
    gap = np.sort(q, axis=1)[:, -1]-np.sort(q, axis=1)[:, -2]
    distance = np.full(len(states), np.inf)
    for wall in env.wall_blocks:
        nearest = np.clip(states, [wall.xmin, wall.ymin], [wall.xmax, wall.ymax])
        distance = np.minimum(distance, np.linalg.norm(states-nearest, axis=1))
    near_wall = distance <= .05
    wall_greedy = blocked[np.arange(len(states)), greedy]
    blocked_margin = np.max(np.where(blocked, q, -np.inf), axis=1)-np.max(np.where(~blocked, q, -np.inf), axis=1)
    nodes, graph = build_graph(env, 20)
    ids = np.argmin(np.sum((states[:, None]-nodes[None])**2, axis=-1), axis=1)
    next_ids = np.argmin(np.sum((transitions[:, :, None]-nodes[None, None])**2, axis=-1), axis=2)
    bfs_bad = np.zeros(len(states), bool)
    bfs_comparable = np.zeros(len(states), bool)
    for i, goal in enumerate(pairs['goals']):
        sl = slice(i*per, (i+1)*per)
        d = bfs_distances(graph, np.max(abs(nodes-goal), axis=1) <= env.threshold_distance)
        progress = (d[next_ids[sl]] < d[ids[sl], None]) & ~blocked[sl]
        bfs_comparable[sl] = progress.any(axis=1)
        bfs_bad[sl] = ~progress[np.arange(per), greedy[sl]] & bfs_comparable[sl]
    by_pair = dict(residual_rmse=np.sqrt(mse.reshape(n, per).mean(axis=1)),
        relative_residual=np.sqrt(mse.reshape(n, per).mean(axis=1)/np.maximum(target_mse.reshape(n, per).mean(axis=1), 1e-12)),
        q_rms=np.sqrt((q*q).reshape(n, per*4).mean(axis=1)),
        q_gap=gap.reshape(n, per).mean(axis=1), z_norm=znorm.reshape(n, per).mean(axis=1),
        online_target_q_rmse=np.sqrt(((q-tq)**2).reshape(n, per*4).mean(axis=1)),
        wall_greedy_fraction=wall_greedy.reshape(n, per).mean(axis=1),
        bfs_nonprogress_fraction=bfs_bad.reshape(n, per).sum(axis=1)/np.maximum(bfs_comparable.reshape(n, per).sum(axis=1), 1))
    summary = dict(residual_rmse=float(np.sqrt(mse.mean())), relative_residual=float(np.sqrt(mse.mean()/target_mse.mean())),
        q_rms=float(np.sqrt((q*q).mean())), median_q_gap=float(np.median(gap)),
        mean_z_norm=float(znorm.mean()), online_target_q_rmse=float(np.sqrt(((q-tq)**2).mean())),
        near_wall_probe_count=int(near_wall.sum()), near_wall_greedy_blocked_fraction=float(wall_greedy[near_wall].mean()),
        near_wall_bfs_nonprogress_fraction=float(bfs_bad[near_wall & bfs_comparable].mean()),
        mean_blocked_action_advantage=float(blocked_margin[np.isfinite(blocked_margin)].mean()))
    return summary, by_pair, dict(q=q, target_q=tq, near_wall=near_wall, wall_greedy=wall_greedy,
                                  bfs_bad=bfs_bad, bfs_comparable=bfs_comparable)


def correlation(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('experiments/fourrooms_doorways/runs/width2_seed0'))
    parser.add_argument('--output', type=Path, default=Path('experiments/fourrooms_doorways/analyses/decline_analysis'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.run/'config.json').read_text())
    env = AdjustableFourRooms(**config['environment'])
    with np.load(args.run/'evaluation_pairs.npz') as data:
        pairs = {k: data[k] for k in data.files}
    learners, records, result = {}, {}, dict(checkpoints={}, stages={}, paired={}, approach_sensitivity={})
    for epoch in (400, 800):
        path = args.run/f'epoch_{epoch:04d}.pkl'
        learner = FBLearner.from_checkpoint(path)
        learners[epoch] = learner
        record = traces(learner, env, pairs)
        records[epoch] = record
        np.savez_compressed(args.output/f'trace_{epoch}.npz', **record)
        with np.load(args.run/f'evaluation/epoch_{epoch:04d}_trials.npz') as original:
            reproduction = {str(h): int(np.sum((record['hits'] <= h) != (original[f'hits_{h}'] <= h))) for h in (30, 150)}
        result['checkpoints'][epoch] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            replay_success_disagreements=reproduction)
        print(f'Traced {epoch}: disagreement counts {reproduction}', flush=True)
    probes = fixed_probes(records[400], records[800])
    transitions, blocked = nominal_transitions(env, probes.reshape(-1, 2))
    rng = np.random.default_rng(909)
    # Independent common backward-state batch, uniform in free space.
    other = []
    while len(other) < 128:
        p = rng.uniform(0, 1, 2)
        if not any(b.xmin <= p[0] <= b.xmax and b.ymin <= p[1] <= b.ymax for b in env.wall_blocks):
            other.append(p)
    pair_metrics, probe_records, stage_records = {}, {}, {}
    for epoch in (400, 800):
        summary, per_pair, probes_out = probe_metrics(learners[epoch], env, probes, pairs, transitions, blocked, np.asarray(other, np.float32))
        pair_metrics[epoch], probe_records[epoch] = per_pair, probes_out
        result['checkpoints'][epoch]['fixed_probe_metrics'] = summary
        np.savez_compressed(args.output/f'probes_{epoch}.npz', positions=probes, **per_pair, **probes_out)
        for horizon in (30, 150):
            stages = failure_stages(records[epoch], pairs, horizon, env.doorway_scale)
            stage_records[epoch, horizon] = stages
            (args.output/f'stages_{epoch}_{horizon}.json').write_text(json.dumps(stages, indent=2))
            for category, label in enumerate(CATEGORIES):
                selected = [r for i, r in enumerate(stages) if pairs['categories'][i] == category]
                counts = {s: sum(r['stage'] == s for r in selected) for s in sorted(set(r['stage'] for r in selected))}
                result['stages'][f'{epoch}_{horizon}_{label}'] = dict(counts=counts,
                    wall_action_fraction=sum(r['wall_actions'] for r in selected)/sum(r['steps'] for r in selected),
                    repeated_wall_action_fraction=sum(r['repeated_wall_actions'] for r in selected)/sum(r['steps'] for r in selected),
                    fraction_wall_run_at_least_5=float(np.mean([r['longest_identical_wall_run'] >= 5 for r in selected])),
                    mean_longest_wall_run=float(np.mean([r['longest_identical_wall_run'] for r in selected])))
            for threshold in (.025, .05, .1):
                stages2 = failure_stages(records[epoch], pairs, horizon, env.doorway_scale, threshold)
                cross = [r for i, r in enumerate(stages2) if pairs['categories'][i] > 0]
                result['approach_sensitivity'][f'{epoch}_{horizon}_{threshold}'] = {s: sum(r['stage'] == s for r in cross) for s in sorted(set(r['stage'] for r in cross))}
        print(f'Probed {epoch}: {summary}', flush=True)
    q1, q2 = probe_records[400]['q'], probe_records[800]['q']
    near = probe_records[400]['near_wall']
    result['rank_changes'] = dict(all_argmax_flip=float(np.mean(q1.argmax(1) != q2.argmax(1))),
        near_wall_argmax_flip=float(np.mean(q1[near].argmax(1) != q2[near].argmax(1))),
        near_wall_rank_order_changed=float(np.mean(np.any(q1[near].argsort(1) != q2[near].argsort(1), axis=1))))
    for horizon in (30, 150):
        s1, s2 = records[400]['hits'] <= horizon, records[800]['hits'] <= horizon
        for category, label in enumerate(CATEGORIES):
            selected = pairs['categories'] == category
            failure_change = s1.astype(int)-s2.astype(int)
            groups = {'lost': selected & s1 & ~s2, 'kept_success': selected & s1 & s2,
                      'gained': selected & ~s1 & s2, 'both_failed': selected & ~s1 & ~s2}
            entry = dict(groups={})
            for name, mask in groups.items():
                entry['groups'][name] = dict(count=int(mask.sum()), metrics={
                    k: dict(epoch400=float(v[mask].mean()), epoch800=float(pair_metrics[800][k][mask].mean()))
                    for k, v in pair_metrics[400].items()} if mask.any() else {})
            entry['correlation_with_failure_change'] = {k: correlation((pair_metrics[800][k]-v)[selected], failure_change[selected]) for k, v in pair_metrics[400].items()}
            result['paired'][f'{horizon}_{label}'] = entry
    # Separate objective-level probes under native noisy random collection.
    from fb_navigation.training.run_fb_jax import make_collector
    from fb_navigation.diagnostics.diagnose_continuous_maze import residual_metrics
    collector = jax.jit(make_collector(env, episodes=64, steps=30))
    _, heldout = collector(jax.random.PRNGKey(991), jax.random.PRNGKey(992), jax.random.PRNGKey(993))
    heldout = {k: np.asarray(v) for k, v in heldout.items() if k != 'next_actions'}
    np.savez_compressed(args.output/'heldout_random_transitions.npz', **heldout)
    result['heldout_cauchy_bootstrap'] = {}
    result['oracle_wall_mask_diagnostic'] = {}
    for epoch in (400, 800):
        result['heldout_cauchy_bootstrap'][epoch] = residual_metrics(learners[epoch], heldout, 995, 32)
        b = np.asarray(learners[epoch].embed_B(heldout['obs']))
        eigenvalues = np.linalg.eigvalsh(b.T@b/len(b))
        result['heldout_cauchy_bootstrap'][epoch].update(
            b_second_moment_trace=float(eigenvalues.sum()), b_largest_eigenvalue=float(eigenvalues[-1]),
            b_effective_rank=float(eigenvalues.sum()**2/np.sum(eigenvalues**2)))
        masked = traces(learners[epoch], env, pairs, mask_wall_actions=True)
        np.savez_compressed(args.output/f'oracle_mask_trace_{epoch}.npz', **masked)
        for horizon in (30, 150):
            for category, label in enumerate(CATEGORIES):
                selected = pairs['categories'] == category
                baseline_success = records[epoch]['hits'] <= horizon
                intervention_success = masked['hits'] <= horizon
                result['oracle_wall_mask_diagnostic'][f'{epoch}_{horizon}_{label}'] = dict(
                    success_rate=float(intervention_success[selected].mean()),
                    rescued=int((selected & ~baseline_success & intervention_success).sum()),
                    harmed=int((selected & baseline_success & ~intervention_success).sum()))
        lost = (pairs['categories'] == 2) & (records[400]['hits'] <= 30) & (records[800]['hits'] > 30)
        result['oracle_wall_mask_diagnostic'][f'{epoch}_lost38_rescued'] = int((masked['hits'][lost] <= 30).sum())
        print(f'Extended {epoch}: {result["heldout_cauchy_bootstrap"][epoch]}', flush=True)
    (args.output/'provenance.json').write_text(json.dumps(dict(
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        jax=jax.__version__, numpy=np.__version__, run=str(args.run.resolve()),
        definition='Common union trajectory probes; noiseless action transitions; separate native noisy Cauchy probes; map-oracle masking diagnostic only'), indent=2))
    (args.output/'summary.json').write_text(json.dumps(result, indent=2))
    plot_analysis(args.output, records, pairs, pair_metrics, result, env)
    print('Saved summary and plots to '+str(args.output), flush=True)


def plot_analysis(output, records, pairs, metrics, result, env):
    from fb_navigation.environments.continuous_four_rooms_jax import visualize_environment
    lost = np.flatnonzero((pairs['categories'] == 2) & (records[400]['hits'] <= 30) & (records[800]['hits'] > 30))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, i in zip(axes.flat, lost[:6]):
        visualize_environment(env, ax, draw_initial_position=False, draw_goal=False)
        for epoch, color in [(400, 'tab:blue'), (800, 'tab:red')]:
            length = min(int(records[epoch]['hits'][i]), 30)
            path = np.vstack([pairs['starts'][i], records[epoch]['following'][:length, i]])
            ax.plot(*path.T, color=color, label=str(epoch), linewidth=1.5)
        ax.scatter(*pairs['starts'][i], marker='o', color='green')
        ax.scatter(*pairs['goals'][i], marker='*', color='black')
        ax.set_title(f'Lost diagonal trial {i}')
    axes.flat[0].legend(); fig.tight_layout(); fig.savefig(output/'lost_trajectories.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    selected = pairs['categories'] == 2
    changed = (records[400]['hits'] <= 30).astype(int)-(records[800]['hits'] <= 30).astype(int)
    for ax, key in zip(axes, ['relative_residual', 'q_rms', 'bfs_nonprogress_fraction']):
        delta = metrics[800][key]-metrics[400][key]
        for value, label in [(-1, 'gained'), (0, 'unchanged'), (1, 'lost')]:
            mask = selected & (changed == value)
            ax.scatter(np.flatnonzero(mask), delta[mask], s=13, alpha=.6, label=label)
        ax.axhline(0, color='gray'); ax.set(xlabel='Diagonal trial index', ylabel='Change in '+key)
    axes[0].legend(); fig.tight_layout(); fig.savefig(output/'paired_deltas.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
