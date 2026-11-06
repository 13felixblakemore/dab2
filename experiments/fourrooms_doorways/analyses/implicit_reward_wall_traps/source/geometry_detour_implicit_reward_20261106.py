"""Geometry-only goal detours, scored with frozen B rewards in native dynamics."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import json
import types
import jax
import jax.numpy as jnp
import numpy as np
from adjustable_four_rooms import AdjustableFourRooms
from continuous_four_rooms_jax import Point, _move
from control_bfs_bottlenecks import build_graph, bfs_distances
from forward_backward_jax import FBLearner, IdentityFeaturiser
from experiments.maintenance.diagnose_implicit_reward_traps_20261106 import make_rollout

OUTPUT = ROOT/'experiments/fourrooms_doorways/analyses/implicit_reward_wall_traps'


def potential_fields(env, goals):
    nodes, graph = build_graph(env, 20)
    distances = []
    for goal in goals:
        target = np.argmin(np.sum((nodes-goal)**2, axis=1))
        assert np.linalg.norm(nodes[target]-goal) < 1e-6
        d = bfs_distances(graph, np.arange(len(nodes)) == target)
        assert np.all(d >= 0)
        distances.append(d*.05)
    distances = np.array(distances)
    axis = np.linspace(0, 1, 201)
    xx, yy = np.meshgrid(axis, axis); points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    blocks = []
    for begin in range(0, len(points), 512):
        p = points[begin:begin+512]
        distance2 = np.sum((p[:, None]-nodes[None])**2, axis=2)
        near = np.argsort(distance2, axis=1)[:, :8]
        end = nodes[near]; start = p[:, None, :]
        blocked = np.zeros(near.shape, bool)
        for w in env.wall_blocks:
            tmin = np.zeros(near.shape); tmax = np.ones(near.shape); valid = np.ones(near.shape, bool)
            for dim, lower, upper in ((0, w.xmin, w.xmax), (1, w.ymin, w.ymax)):
                delta = end[:, :, dim]-start[:, :, dim]; parallel = abs(delta) < 1e-12
                safe = np.where(parallel, 1., delta)
                a = (lower-start[:, :, dim])/safe; b = (upper-start[:, :, dim])/safe
                tmin = np.where(parallel, tmin, np.maximum(tmin, np.minimum(a, b)))
                tmax = np.where(parallel, tmax, np.minimum(tmax, np.maximum(a, b)))
                valid &= ~parallel | ((start[:, :, dim] >= lower) & (start[:, :, dim] <= upper))
            blocked |= valid & (tmax > 1e-6) & ((tmax-tmin) > 1e-6)
        length = np.sqrt(np.take_along_axis(distance2, near, axis=1))
        costs = distances[:, near] + length[None]
        cost = np.min(np.where(blocked[None], np.inf, costs), axis=2)
        # Cells inside a wall are only rounding fallbacks; actual moves are checked below.
        fallback = distances[:, near[:, 0]] + length[:, 0][None]
        blocks.append(np.where(np.isfinite(cost), cost, fallback).T)
    return np.concatenate(blocks).reshape(201, 201, len(goals)).astype(np.float32)


def main():
    settings = json.loads((OUTPUT/'config.json').read_text())
    run = Path(settings['run']); config = json.loads((run/'config.json').read_text())
    env = AdjustableFourRooms(**config['environment'])
    learner = FBLearner.from_checkpoint(run/'epoch_0800.pkl', device=settings['device'])
    states = np.array([p['state'] for p in settings['probes']], np.float32)
    goals = np.array([p['goal'] for p in settings['probes']], np.float32)
    tasks = jnp.asarray(learner.embed_B(goals))
    corrected_tasks = np.load(OUTPUT/'covariance.npz')['corrected_tasks']
    potential = jnp.asarray(potential_fields(env, goals))
    actions = jnp.asarray([(a.x, a.y) for a in env.actions], jnp.float32)
    walls = tuple(env.wall_blocks)

    class RewardBackward:
        def apply(self, params, obs):
            return learner.backward_network.apply(params, learner.featuriser.transform(obs))

    class GeometryScores:
        def apply(self, params, obs, z):
            goal_ids = jnp.argmin(jnp.sum((z[:, None]-tasks[None])**2, axis=2), axis=1)
            def candidates(p):
                def move(a):
                    nxt, _, blocked = _move(Point(*p), Point(*a), jax.random.PRNGKey(0), 0., env.size, walls)
                    point = jnp.array([nxt.x, nxt.y])
                    return point, blocked | (jnp.sum((point-p)**2) < 1e-12)
                return jax.vmap(move)(actions)
            nxt, blocked = jax.vmap(candidates)(obs)
            cells = jnp.clip(jnp.rint(nxt*200).astype(jnp.int32), 0, 200)
            cost = potential[cells[:, :, 1], cells[:, :, 0], goal_ids[:, None]]
            cost += .001*jnp.linalg.norm(nxt-jnp.asarray(goals)[goal_ids, None], axis=2)
            scores = -jnp.where(blocked, 1e6, cost)
            # Adapter for the shared rollout engine; scores are geometry costs, not FB predictions.
            return scores[:, None, :] * z[:, :, None] / jnp.sum(z*z, axis=1)[:, None, None]

    controller = types.SimpleNamespace(discount_factor=learner.discount_factor, temperature=learner.temperature,
        featuriser=IdentityFeaturiser(), backward_network=RewardBackward(), forward_network=GeometryScores())
    n = len(states); reps = settings['rollouts']; batch = 512
    pids = np.repeat(np.arange(n), 4*reps)
    first = np.tile(np.repeat(np.arange(4), reps), n)
    replicas = np.tile(np.arange(reps), n*4)
    ids = pids*reps+replicas
    cache = OUTPUT/'geometry_detour.npz'
    if not cache.exists():
        fn = make_rollout(controller, env, settings['horizons'], 'greedy')
        batches = []
        for begin in range(0, len(ids), batch):
            sl = slice(begin, begin+batch); selected = pids[sl]
            keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(settings['seed']), i))(jnp.asarray(ids[sl], jnp.uint32))
            result = fn(learner.state['params'], jnp.asarray(states[selected]), jnp.asarray(goals[selected]),
                jnp.asarray(first[sl]), keys, jnp.asarray(corrected_tasks[selected]))
            batches.append(jax.device_get(result))
            print(json.dumps(dict(policy='geometry_detour', completed=min(begin+batch, len(ids)), total=len(ids))), flush=True)
        record = {k: np.concatenate([b[k] for b in batches]).reshape((n, 4, reps)+batches[0][k].shape[1:]) for k in batches[0]}
        np.savez_compressed(cache, **record)
    record = dict(np.load(cache))
    metadata = dict(policy='geometry-only shortest-path potential on 20x20 graph with visible-neighbor interpolation; native noisy movement; no goal termination; continue closest-goal moves after entry',
        goal_hit_rate_by_probe_action=(record['hits'] <= max(settings['horizons'])).mean(2).tolist(),
        no_change_to_FB_or_training=True)
    (OUTPUT/'geometry_detour.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__': main()
