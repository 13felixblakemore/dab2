"""Fixed stratified Four Rooms trials, native dynamics, and lattice BFS reference."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from continuous_four_rooms_jax import Point, _move
from control_bfs_bottlenecks import build_graph, room_ids, bfs_distances

CATEGORIES = ('same_room', 'adjacent_rooms', 'diagonal_rooms')
DOORS = ('bottom', 'top', 'left', 'right')


def make_pairs(env, per_category=200, seed=271828):
    nodes, graph = build_graph(env, 20)
    room = room_ids(nodes, env.size)
    valid = np.flatnonzero(room >= 0)
    rng = np.random.default_rng(seed)
    starts, goals, categories = [], [], []
    for category in range(3):
        count = 0
        while count < per_category:
            a, b = rng.choice(valid, 2)
            difference = room[a] ^ room[b]
            kind = 0 if difference == 0 else 2 if difference == 3 else 1
            if kind != category or np.max(abs(nodes[a]-nodes[b])) <= env.threshold_distance:
                continue
            starts.append(a); goals.append(b); categories.append(category); count += 1
    lengths, cache = [], {}
    for a, b in zip(starts, goals):
        if b not in cache:
            cache[b] = bfs_distances(graph, np.max(abs(nodes-nodes[b]), axis=1) <= env.threshold_distance)
        lengths.append(cache[b][a])
    if min(lengths) < 1:
        raise ValueError('Invalid BFS reference')
    return dict(starts=nodes[starts].astype(np.float32), goals=nodes[goals].astype(np.float32),
                categories=np.array(categories), start_rooms=room[starts], goal_rooms=room[goals],
                bfs_lengths=np.array(lengths), trial_seed=np.array(seed))


def coordinate_room(points):
    x, y = points[:, 0]*11, points[:, 1]*11
    return jnp.where((x < 6)&(y < 4), 0, jnp.where((x < 6)&(y > 5), 1,
           jnp.where((x > 7)&(y < 4), 2, jnp.where((x > 7)&(y > 5), 3, -1))))


def make_rollout(learner, env, horizon, epsilon=.02):
    walls = tuple(env.wall_blocks)
    actions = jnp.array([(a.x, a.y) for a in env.actions])
    @jax.jit
    def rollout(params, starts, goals, keys):
        tasks = learner.backward_network.apply(params['b'], learner.featuriser.transform(goals))
        n = len(starts)
        def step(carry, t):
            position, hit, collisions, committed_room, doors, keys = carry
            split = jax.vmap(lambda k: jax.random.split(k, 4))(keys)
            keys, explore_key, action_key, move_key = [split[:, i] for i in range(4)]
            f = learner.forward_network.apply(params['f'], learner.featuriser.transform(position), tasks)
            q = jnp.einsum('bda,bd->ba', f, tasks)
            explore = jax.vmap(lambda k: jax.random.uniform(k) < epsilon)(explore_key)
            random_action = jax.vmap(lambda k: jax.random.randint(k, (), 0, len(actions)))(action_key)
            selected = jnp.where(explore, random_action, jnp.argmax(q, axis=1))
            def move(p, a, key):
                nxt, _, blocked = _move(Point(*p), Point(*actions[a]), key, env._noise, env.size, walls)
                return jnp.array([nxt.x, nxt.y]), blocked
            proposed, blocked = jax.vmap(move)(position, selected, move_key)
            active = hit > horizon
            position = jnp.where(active[:, None], proposed, position)
            collisions += (active & blocked).astype(jnp.int32)
            current_room = coordinate_room(position/env.size)
            crossed = active & (current_room >= 0) & (committed_room >= 0) & (current_room != committed_room)
            # A completed crossing requires entering a new room, not just a door strip.
            vertical = (current_room ^ committed_room) == 2
            door_id = jnp.where(vertical, committed_room % 2, jnp.where(committed_room < 2, 2, 3))
            doors += jax.nn.one_hot(door_id, 4, dtype=jnp.int32)*crossed[:, None]
            committed_room = jnp.where(active & (current_room >= 0), current_room, committed_room)
            reached = jnp.all(abs(position-goals) <= env.threshold_distance, axis=1)
            hit = jnp.where(active & reached, t+1, hit)
            return (position, hit, collisions, committed_room, doors, keys), position
        initial = (starts, jnp.full(n, horizon+1, jnp.int32), jnp.zeros(n, jnp.int32),
                   coordinate_room(starts/env.size), jnp.zeros((n, 4), jnp.int32), keys)
        final, path = jax.lax.scan(step, initial, jnp.arange(horizon))
        return final[1], final[2], final[4], path
    return rollout


def summarize(hits, collisions, doors, pairs, horizon):
    rows = []
    for category, label in enumerate(CATEGORIES):
        selected = pairs['categories'] == category
        hit, collision, door = hits[selected], collisions[selected], doors[selected]
        bfs = pairs['bfs_lengths'][selected]
        success = hit <= horizon
        feasible = bfs <= horizon
        actions = np.minimum(hit, horizon)
        good = hit[success]
        row = dict(category=label, trials=int(selected.sum()), success_rate=float(success.mean()),
                   mean_successful_steps=float(good.mean()) if len(good) else None,
                   wall_action_percent=float(100*collision.sum()/actions.sum()),
                   mean_doorway_crossings=float(door.sum(axis=1).mean()),
                   any_doorway_crossing_rate=float(np.mean(door.sum(axis=1) > 0)),
                   mean_bfs_steps=float(bfs.mean()), bfs_feasible_fraction=float(feasible.mean()),
                   success_on_bfs_feasible=float(success[feasible].mean()) if feasible.any() else None,
                   mean_success_steps_over_bfs=float(np.mean(good/bfs[success])) if len(good) else None)
        row.update({f'mean_{name}_crossings': float(door[:, i].mean()) for i, name in enumerate(DOORS)})
        rows.append(row)
    return rows


def evaluate_checkpoint(learner, env, pairs, output, epoch, horizons=(30, 150), batch_size=100):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows, trials = [], {}
    for horizon in horizons:
        rollout = make_rollout(learner, env, horizon)
        all_hits, all_collisions, all_doors = [], [], []
        for first in range(0, len(pairs['starts']), batch_size):
            end = min(first+batch_size, len(pairs['starts']))
            keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(int(pairs['trial_seed'])), i))(
                jnp.arange(first, end, dtype=jnp.uint32))
            hit, blocked, doors, _ = rollout(learner.state['params'], jnp.asarray(pairs['starts'][first:end]),
                                            jnp.asarray(pairs['goals'][first:end]), keys)
            all_hits.append(np.asarray(hit)); all_collisions.append(np.asarray(blocked)); all_doors.append(np.asarray(doors))
        hit, blocked, doors = map(np.concatenate, (all_hits, all_collisions, all_doors))
        rows.extend(dict(epoch=epoch, horizon=horizon, **r) for r in summarize(hit, blocked, doors, pairs, horizon))
        trials.update({f'hits_{horizon}': hit, f'collisions_{horizon}': blocked, f'doors_{horizon}': doors})
    np.savez_compressed(output/f'epoch_{epoch:04d}_trials.npz', **trials)
    (output/f'epoch_{epoch:04d}.json').write_text(json.dumps(rows, indent=2))
    return rows


def plot_history(history, output):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for horizon in sorted(set(r['horizon'] for r in history)):
        for category in CATEGORIES:
            rows = [r for r in history if r['horizon'] == horizon and r['category'] == category]
            style = '-' if horizon == 30 else '--'
            for ax, key in zip(axes.flat, ['success_rate', 'wall_action_percent', 'mean_doorway_crossings', 'mean_success_steps_over_bfs']):
                ax.plot([r['epoch'] for r in rows], [np.nan if r[key] is None else r[key] for r in rows], style, label=f'{category}, H={horizon}')
                ax.set(xlabel='Epoch', ylabel=key)
    axes.flat[0].set_ylim(0, 1)
    axes.flat[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(Path(output)/'navigation.png', dpi=160); plt.close(fig)
