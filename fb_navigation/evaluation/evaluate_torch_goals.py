"""Native PyTorch inference with the same goal benchmark/noise as JAX.

Models are neither converted to JAX nor retrained. The common environment
kernel and stochastic-action draws are JAX to permit exactly paired dynamics.
The original NumPy environment's deterministic movement is checked separately.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fb_navigation.agents.forward_backward import FBLearner, RadialBasisFunction2D
from fb_navigation.environments.continuous_four_rooms_jax import ContinuousFourRooms, Point, _move
from fb_navigation.environments.continuous_four_rooms import ContinuousFourRooms as OriginalRooms, Point as OriginalPoint
from fb_navigation.diagnostics.diagnose_fb_walls import nominal_action_mask, collision_runs

POLICIES = ['greedy_online', 'epsilon_online', 'greedy_target', 'epsilon_target',
            'softmax_target', 'wall_aware_online', 'wall_aware_target', 'random']
ROOMS = ['bottom_left', 'top_left', 'bottom_right', 'top_right']


def load_learner(checkpoint, device):
    learner = FBLearner.from_checkpoint(checkpoint, device=device, buffer_device='cpu')
    # map_location moves stored tensors but not the pickled featuriser.device
    # string. Recreate the same features on the evaluation device only.
    feature = learner.featuriser
    if isinstance(feature, RadialBasisFunction2D):
        learner.featuriser = RadialBasisFunction2D(
            (float(np.min(feature.x_mu)), float(np.max(feature.x_mu))),
            (float(np.min(feature.y_mu)), float(np.max(feature.y_mu))),
            feature.dim, feature.sigma, device=device)
    for module in (learner.forward_network, learner.forward_target_network,
                   learner.backward_network, learner.backward_target_network):
        module.eval()
    return learner


def make_environment_step(env, horizon, policy, temperature):
    actions = jnp.asarray([(a.x, a.y) for a in env.actions], jnp.float32)
    walls = tuple(env.wall_blocks)
    available = jnp.zeros(len(actions), bool).at[jnp.asarray(env.get_available_actions(None))].set(True)
    @jax.jit
    def step(position, hits, keys, scores, t):
        split = jax.vmap(lambda k: jax.random.split(k, 3))(keys)
        keys, ak, mk = split[:, 0], split[:, 1], split[:, 2]
        mask = available
        if policy.startswith('wall_aware'):
            mask = nominal_action_mask(position, actions, env.size, walls)&available
            empty = ~jnp.any(mask, axis=1)
            mask = jnp.where(empty[:, None], available, mask)
        else:
            empty = jnp.zeros(len(position), bool)
        logits = jnp.where(mask, scores, -jnp.inf)
        if policy == 'random':
            logits = jnp.broadcast_to(jnp.where(available, 0., -jnp.inf), scores.shape)
            selected = jax.vmap(lambda k, row: jax.random.categorical(k, row))(ak, logits)
        elif policy.startswith('epsilon'):
            split_action = jax.vmap(lambda k: jax.random.split(k, 2))(ak)
            explore = jax.vmap(lambda k: jax.random.uniform(k) < .02)(split_action[:, 0])
            random_action = jax.vmap(lambda k: jax.random.categorical(k, jnp.where(available, 0., -jnp.inf)))(split_action[:, 1])
            selected = jnp.where(explore, random_action, jnp.argmax(logits, axis=1))
        elif policy == 'softmax_target':
            selected = jax.vmap(lambda k, row: jax.random.categorical(k, row/temperature))(ak, logits)
        else:
            selected = jnp.argmax(logits, axis=1)
        def move(p, a, key):
            nxt, _, blocked = _move(Point(*p), Point(*actions[a]), key, env._noise, env.size, walls)
            return jnp.asarray((nxt.x, nxt.y)), blocked
        nxt, collision = jax.vmap(move)(position, selected, mk)
        active = hits > horizon
        nxt = jnp.where(active[:, None], nxt, position)
        return nxt, keys, selected, collision&active, active, empty&active
    return step


def movement_equivalence(seed=91, count=256):
    original = OriginalRooms(movement_noise=0., include_stay_action=False)
    common = ContinuousFourRooms(movement_noise=0., include_stay_action=False)
    rng = np.random.default_rng(seed)
    positions = []
    while len(positions) < count:
        p = rng.uniform(0, 1, 2).astype(np.float32)
        if not any(w.contains(OriginalPoint(*p)) for w in original.wall_blocks):
            positions.append(p)
    positions = np.asarray(positions)
    chosen = rng.integers(0, 4, count)
    def move(p, a):
        actions = jnp.asarray([(x.x, x.y) for x in common.actions])
        nxt, _, hit = _move(Point(*p), Point(*actions[a]), jax.random.PRNGKey(0), 0., common.size, tuple(common.wall_blocks))
        return jnp.asarray((nxt.x, nxt.y)), hit
    nxt, collision = jax.jit(jax.vmap(move))(jnp.asarray(positions), jnp.asarray(chosen))
    reference, flags = [], []
    for p, a in zip(positions, chosen):
        original.reset(state=OriginalPoint(*p), goal=OriginalPoint(.8, .8))
        q, _, _, info = original.step(int(a))
        reference.append(q); flags.append(info['goes_through_wall'])
    np.testing.assert_allclose(nxt, reference, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(collision, flags)
    return dict(noiseless_transition_checks=count, collision_flags_identical=True,
                max_coordinate_difference=float(np.max(np.abs(np.asarray(nxt)-reference))))


@torch.inference_mode()
def rollout(learner, env, starts, goals, tasks, keys, horizon, policy):
    step = make_environment_step(env, horizon, policy, learner.temperature)
    n = len(starts)
    position = jnp.asarray(starts); hits = jnp.full(n, horizon+1, jnp.int32)
    goals_device = jnp.asarray(goals)
    records = {k: [] for k in ('position', 'action', 'collision', 'active', 'empty_mask')}
    network = learner.forward_target_network if policy.endswith('_target') else learner.forward_network
    for t in range(horizon):
        if policy == 'random':
            scores = np.zeros((n, learner.n_actions), np.float32)
        else:
            obs = torch.tensor(np.asarray(position), device=learner.device, dtype=torch.float32)
            features = learner.featuriser.transform(obs)
            f = network(features, tasks)
            scores = torch.einsum('nda,nd->na', f, tasks).cpu().numpy()
        nxt, keys, action, collision, active, empty = step(position, hits, keys, jnp.asarray(scores), t)
        reached = jnp.all(jnp.abs(nxt-goals_device) <= env.threshold_distance, axis=1)
        hits = jnp.where(active&reached, t+1, hits)
        for name, value in (('position', position), ('action', action), ('collision', collision), ('active', active), ('empty_mask', empty)):
            records[name].append(np.asarray(value))
        position = nxt
    records = {k: np.stack(v, axis=1) for k, v in records.items()}
    records['final_position'] = np.asarray(position)
    return np.asarray(hits), records


def summary(hits, collisions, horizon):
    good = hits <= horizon
    return dict(success_rate=float(good.mean()), success_by_25=float(np.mean(hits <= min(25, horizon))),
                mean_success_steps=float(hits[good].mean()) if good.any() else None,
                median_success_steps=float(np.median(hits[good])) if good.any() else None,
                mean_capped_steps=float(np.minimum(hits, horizon).mean()),
                wall_action_fraction=float(collisions.sum()/np.minimum(hits, horizon).sum()))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def run(args):
    if min(args.samples, args.rollouts, args.horizon) < 0 or args.threads < 1:
        raise ValueError('Counts must be nonnegative, threads positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    learner = load_learner(args.checkpoint, device)
    parent = Path(args.pairs).parent
    reference = json.loads((parent/'metrics.json').read_text())
    source = np.load(args.pairs)
    count = args.samples or len(source['starts'])
    count = min(count, len(source['starts']))
    repeats = args.rollouts or reference['rollouts']
    horizon = args.horizon or reference['horizon']
    seed = reference['seed']
    starts, goals = source['starts'][:count], source['goals'][:count]
    sroom, groom = source['start_rooms'][:count], source['goal_rooms'][:count]
    env = ContinuousFourRooms(include_stay_action=learner.n_actions == 5)
    movement_check = movement_equivalence()
    with torch.inference_mode():
        task = learner.embed_B(goals).to(device)
        # Verify native act() and the batched inference Q extraction agree.
        p = torch.tensor(starts[:32], device=device)
        f = learner.forward_network(learner.featuriser.transform(p), task[:32])
        expected = torch.einsum('nda,nd->na', f, task[:32]).argmax(dim=1)
        torch.testing.assert_close(learner.act(p, task[:32]), expected)
    indices = np.arange(count)[:, None]*max(repeats, reference['rollouts'])+np.arange(repeats)[None]
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(seed+1), i))(jnp.asarray(indices.reshape(-1)))
    pair_starts, pair_goals = np.repeat(starts, repeats, axis=0), np.repeat(goals, repeats, axis=0)
    task = task.repeat_interleave(repeats, dim=0)
    print(f'Native PyTorch inference on {device}; {count} pairs x {repeats} trials, limit {horizon}', flush=True)
    all_rows, room_rows, pair_rows, hits_by_policy = [], [], [], {}
    saved = {}
    for policy in POLICIES:
        print('Evaluating '+policy, flush=True)
        h, trace = rollout(learner, env, pair_starts, pair_goals, task, keys, horizon, policy)
        hits = h.reshape(count, repeats)
        collisions = trace['collision'].sum(axis=1).reshape(count, repeats)
        hits_by_policy[policy] = hits
        saved[policy+'_hits'], saved[policy+'_collisions'] = hits, collisions
        row = dict(policy=policy, **summary(hits, collisions, horizon), empty_mask_steps=int(trace['empty_mask'].sum()))
        failed = h > horizon
        runs = collision_runs(trace).max(axis=1)
        row['failed_fraction_10_repeated_wall_actions'] = float(np.mean(runs[failed] >= 10)) if failed.any() else None
        if policy == 'random' and learner.n_actions == 4 and repeats <= reference['rollouts'] and horizon == reference['horizon']:
            expected = source['random_hits'][:count, :repeats]
            np.testing.assert_array_equal(hits, expected)
            row['jax_random_control_mismatches'] = 0
        all_rows.append(row)
        for s in range(4):
            for g in range(4):
                take = (sroom == s)&(groom == g)
                if take.any():
                    room_rows.append(dict(policy=policy, start_room=ROOMS[s], goal_room=ROOMS[g], pairs=int(take.sum()),
                                          **summary(hits[take], collisions[take], horizon)))
        for i in range(count):
            pair_rows.append(dict(policy=policy, pair=i, start_x=starts[i, 0], start_y=starts[i, 1], goal_x=goals[i, 0], goal_y=goals[i, 1],
                                  start_room=ROOMS[sroom[i]], goal_room=ROOMS[groom[i]], **summary(hits[i], collisions[i], horizon)))
        np.savez_compressed(out/(policy+'_traces.npz'), hits=h, **trace)
        print(json.dumps(row), flush=True)
    np.savez_compressed(out/'trials.npz', starts=starts, goals=goals, start_rooms=sroom, goal_rooms=groom, **saved)
    feature = learner.featuriser
    config = {k: v for k, v in learner.config.items() if k != 'featuriser'}
    config['featuriser'] = dict(type=type(feature).__name__, dim=getattr(feature, 'dim', None), sigma=getattr(feature, 'sigma', None))
    updates = max((int(value['step'].item()) for value in learner.optim.state.values() if 'step' in value), default=0)
    comparisons = []
    def compare(row, jax_hits):
        jax_hits = jax_hits[:count, :repeats]
        delta = (hits_by_policy[row['policy']] <= horizon).mean(axis=1)-(jax_hits <= horizon).mean(axis=1)
        boot = np.random.default_rng(seed+100).choice(delta, size=(2000, count), replace=True).mean(axis=1)
        comparisons.append(dict(policy=row['policy'], torch_success=row['success_rate'], jax_success=float(np.mean(jax_hits <= horizon)),
                                paired_success_difference=float(delta.mean()), paired_difference_ci95=np.quantile(boot, [.025, .975]).tolist()))
    paired_reference = horizon <= reference['horizon'] and repeats <= reference['rollouts']
    for row in all_rows:
        if paired_reference and row['policy']+'_hits' in source.files:
            compare(row, source[row['policy']+'_hits'])
    if paired_reference and Path(args.jax_walls).exists():
        walls = json.loads(Path(args.jax_walls).read_text())
        mapping = {'wall_aware_online': 'online_wall_aware', 'wall_aware_target': 'target_wall_aware'}
        for row in all_rows:
            if row['policy'] in mapping:
                with np.load(Path(args.jax_walls).parent/(mapping[row['policy']]+'_traces.npz')) as archive:
                    old_hits = archive['hits'].reshape(-1, reference['rollouts'])
                    compare(row, old_hits)
    metrics = dict(checkpoint=args.checkpoint, checkpoint_sha256=digest, torch_version=torch.__version__,
        device=device, config=config, optimizer_updates=updates, temperature=learner.temperature,
        temperature_source='run_fb.py and FBLearner default; temperature is omitted from the Torch checkpoint config',
        pairs=count, repeats=repeats, horizon=horizon, seed=seed, movement_check=movement_check,
        dynamics=dict(step=.05, noise=env._noise, goal_tolerance=env.threshold_distance, goal_norm='L-infinity'),
        results=all_rows, jax_comparison=comparisons,
        notes=['Actual PyTorch networks and original PyTorch RBF preprocessing are used; no weight conversion or learning.',
               'The common JAX environment kernel supplies exactly paired movement noise and stochastic-action draws.',
               'Its noiseless movement is checked against the original NumPy environment, including exact collision flags.',
               'Goal tasks are online B(goal) for both forward snapshots. First successful entry freezes the trial.',
               'Greedy/epsilon/softmax use all environment-available actions. Wall-aware is a separate diagnostic using noiseless wall intersections; outer clipping remains available.',
               'Featuriser device placement is repaired only on the in-memory loaded object, preserving bounds, sigma and weights.',
               'Reported successful trajectory lengths exclude failures. Success denominators include every trial.',
               'Checkpoint comparisons concern these two independently trained models; they do not isolate a framework effect.'])
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2))
    write_csv(out/'summary.csv', all_rows); write_csv(out/'room_pairs.csv', room_rows); write_csv(out/'pair_metrics.csv', pair_rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for name, hits in hits_by_policy.items():
        axes[0].plot(np.arange(horizon+1), [np.mean(hits <= t) for t in range(horizon+1)], label=name)
    axes[0].set(xlabel='Action steps', ylabel='Goal success fraction'); axes[0].legend(fontsize=7); axes[0].grid(alpha=.2)
    labels = [r['policy'] for r in comparisons]; x = np.arange(len(labels))
    axes[1].bar(x-.2, [r['torch_success'] for r in comparisons], .4, label='Torch checkpoint')
    axes[1].bar(x+.2, [r['jax_success'] for r in comparisons], .4, label='JAX original checkpoint')
    axes[1].set_xticks(x, labels, rotation=45, ha='right', fontsize=7); axes[1].set_ylabel('Goal success fraction'); axes[1].legend(fontsize=8)
    fig.savefig(out/'goal_reaching_comparison.png', dpi=160); plt.close(fig)
    report = ['# PyTorch FB goal-reaching evaluation', '',
        f'{args.checkpoint}; {updates:,} optimizer updates; {count} identical cross-room pairs x {repeats} trials; {horizon} actions maximum. No retraining.', '',
        '| Policy | Success | Mean successful steps | Median successful steps | Wall actions |', '|---|---:|---:|---:|---:|']
    for r in all_rows:
        mean = 'N/A' if r['mean_success_steps'] is None else f'{r["mean_success_steps"]:.1f}'
        median = 'N/A' if r['median_success_steps'] is None else f'{r["median_success_steps"]:.1f}'
        report.append(f'| {r["policy"]} | {r["success_rate"]:.1%} | {mean} | {median} | {r["wall_action_fraction"]:.1%} |')
    report += ['', *metrics['notes'], '', metrics['temperature_source'], '',
               'The random control must reproduce the saved JAX first-hit times exactly when the action set, repeats and horizon match.',
               'Per-room-pair results: room_pairs.csv. Per-endpoint-pair results: pair_metrics.csv. Full trajectories are saved for every policy.', '',
               'Run: python evaluate_torch_goals.py. Smoke: --samples 16 --rollouts 2 --output evaluation_torch_goals_smoke.']
    (out/'RESULTS.md').write_text('\n'.join(report))
    assert hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() == digest
    return metrics


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='models/fb_fourrooms.pt')
    p.add_argument('--pairs', default='evaluation_original_short_horizon/trials.npz')
    p.add_argument('--jax-walls', default='evaluation_original_walls/metrics.json')
    p.add_argument('--output', default='evaluation_torch_goals')
    p.add_argument('--device', default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--samples', type=int, default=0)
    p.add_argument('--rollouts', type=int, default=0)
    p.add_argument('--horizon', type=int, default=0)
    run(p.parse_args())
