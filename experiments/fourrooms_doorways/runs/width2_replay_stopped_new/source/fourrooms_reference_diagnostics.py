"""Fixed-pair Four Rooms evaluation and matched width-2 goal comparisons."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fb_navigation.environments.continuous_four_rooms_jax import Point, _move
from fb_navigation.diagnostics.control_bfs_bottlenecks import build_graph, room_ids, bfs_distances

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


def summarize_width2(record,pairs,label,policy,horizon):
    hits=record['hits'];column=0 if horizon==30 else 1
    rows=[]
    for category,name in enumerate(CATEGORIES):
        selected=pairs['categories']==category;h=hits[selected];good=h<=horizon
        executed=np.minimum(h,horizon);success=good.mean(axis=1)
        rng=np.random.default_rng(1827);means=success[rng.integers(len(success),size=(4000,len(success)))].mean(1)
        rows.append(dict(model=label,policy=policy,horizon=horizon,category=name,pairs=int(selected.sum()),
            rollouts_per_pair=hits.shape[1],success_rate=float(good.mean()),
            success_interval95=np.quantile(means,[.025,.975]).tolist(),
            mean_success_steps=float(h[good].mean()) if good.any() else None,
            wall_action_percent=float(100*record['collisions'][selected,:,column].sum()/executed.sum()),
            repeated_wall_action_percent=float(100*record['repeats'][selected,:,column].sum()/executed.sum()),
            mean_bfs_steps=float(pairs['bfs_lengths'][selected].mean()),
            bfs_feasible_fraction=float((pairs['bfs_lengths'][selected]<=horizon).mean())))
    return rows


def reference_main(argv=None):
    import argparse
    from fb_navigation.environments.continuous_four_rooms_jax import ContinuousFourRooms
    from fb_navigation.agents.forward_backward_jax import FBLearner
    parser = argparse.ArgumentParser(description='Evaluate a fixed-pair reference checkpoint.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epoch', type=int, default=0)
    args = parser.parse_args(argv)
    learner = FBLearner.from_checkpoint(args.checkpoint)
    if learner.n_actions != 4:
        raise ValueError('This comparison requires the unchanged four-action environment')
    with np.load(args.pairs) as data:
        pairs = {k: data[k] for k in data.files}
    rows = evaluate_checkpoint(learner, ContinuousFourRooms(include_stay_action=False), pairs, args.output, args.epoch)
    plot_history(rows, args.output)
    print(json.dumps(rows, indent=2))


def width2_main(argv=None):
    import argparse
    import hashlib
    from fb_navigation.agents.forward_backward_jax import FBLearner
    from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms, fixed_pairs
    from fb_navigation.diagnostics.compare_navigation_stages import mc_rollout
    from fb_navigation.diagnostics.audit_doorway_goals import point_status

    parser=argparse.ArgumentParser(description='Matched width-2 goal evaluation across saved models.')
    parser.add_argument('--root',type=Path,default=Path('experiments/fourrooms_doorways'))
    parser.add_argument('--output',type=Path,default=Path('experiments/fourrooms_doorways/analyses/goal_reaching'))
    parser.add_argument('--pairs-per-category',type=int,default=200)
    parser.add_argument('--rollouts',type=int,default=32)
    parser.add_argument('--batch-size',type=int,default=512)
    parser.add_argument('--seed',type=int,default=20261107)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--goal-threshold',type=float,default=.1,
        help='Common coordinate-wise success tolerance for all models (default: 0.1)')
    parser.add_argument('--models',nargs='+',default=['width2_seed0','width2_replay_stopped_seed0'])
    args=parser.parse_args(argv);args.output.mkdir(parents=True,exist_ok=True)
    if min(args.pairs_per_category,args.rollouts,args.batch_size)<1:parser.error('Counts must be positive')
    env=AdjustableFourRooms(doorway_scale=2,include_stay_action=False,threshold_distance=args.goal_threshold)
    pairs=fixed_pairs(env,args.pairs_per_category,args.seed)
    validity=dict(starts=point_status(pairs['starts'],env),goals=point_status(pairs['goals'],env))
    if validity['starts']['bad_indices'] or validity['goals']['bad_indices']:raise ValueError('Invalid endpoints')
    settings=dict(seed=args.seed,rollouts=args.rollouts,pairs_per_category=args.pairs_per_category,
        jax=jax.__version__,device=args.device,models=args.models,
        goal_threshold=env.threshold_distance,goal_norm='linf',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),validity=validity)
    manifest=args.output/'config.json'
    if manifest.exists() and json.loads(manifest.read_text())!=settings:raise ValueError('Use a fresh output directory')
    manifest.write_text(json.dumps(settings,indent=2));np.savez_compressed(args.output/'pairs.npz',**pairs)
    results=[];n=len(pairs['starts']);ids=np.repeat(np.arange(n),args.rollouts)
    for label in args.models:
        path=args.root/'runs'/label/'epoch_0800.pkl'
        learner=FBLearner.from_checkpoint(path,device=args.device)
        if learner.n_actions!=4 or int(learner.state['optim'][0])!=800000:raise ValueError('Unexpected checkpoint')
        for policy,epsilon in [('epsilon002',.02),('greedy',0.),('random',1.)]:
            if policy=='random' and label!=args.models[0]:continue
            name=f'{label}_{policy}';cache=args.output/f'{name}.npz';sha=hashlib.sha256(path.read_bytes()).hexdigest()
            if cache.exists():
                record=dict(np.load(cache))
                if str(record['checkpoint_sha256'])!=sha:raise ValueError('Checkpoint changed')
            else:
                fn=mc_rollout(learner,env,150,epsilon);batches=[]
                for first in range(0,len(ids),args.batch_size):
                    sl=slice(first,first+args.batch_size);selected=ids[sl]
                    keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(args.seed+1),i))(
                        jnp.arange(first,min(first+args.batch_size,len(ids)),dtype=jnp.uint32))
                    raw=fn(learner.state['params'],jnp.asarray(pairs['starts'][selected]),jnp.asarray(pairs['goals'][selected]),
                        jnp.full(len(selected),30),keys)
                    batches.append({k:np.asarray(v) for k,v in raw.items()})
                record={k:np.concatenate([b[k] for b in batches]).reshape((n,args.rollouts)+batches[0][k].shape[1:]) for k in batches[0]}
                record['checkpoint_sha256']=np.array(sha);np.savez_compressed(cache,**record)
            for horizon in (30,150):results.extend(summarize_width2(record,pairs,label,policy,horizon))
            print(json.dumps(dict(model=label,policy=policy,success30=[r['success_rate'] for r in results[-6:-3]])),flush=True)
    (args.output/'results.json').write_text(json.dumps(results,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,h in zip(axes,(30,150)):
        labels=[*args.models,'random'];width=.8/len(labels)
        for k,label in enumerate(labels):
            rows=[r for r in results if r['horizon']==h and ((r['model']==label and r['policy']=='epsilon002') or (label=='random' and r['policy']=='random'))]
            ax.bar(np.arange(3)+(k-(len(labels)-1)/2)*width,[r['success_rate'] for r in rows],width=width,label=label)
        ax.set_xticks(range(3),CATEGORIES,rotation=12);ax.set(ylim=(0,1),ylabel='Goal success',title=f'{h} steps');ax.legend(fontsize=7)
    fig.suptitle(f'Width 2: common coordinate-wise goal tolerance {env.threshold_distance:g}')
    fig.tight_layout();fig.savefig(args.output/'success.png',dpi=160);plt.close(fig)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('reference', add_help=False, help='Evaluate a reference checkpoint and fixed pairs.')
    commands.add_parser('width2', add_help=False, help='Compare width-2 models on matched goal trials.')
    args, remaining = parser.parse_known_args(argv)
    return {'reference': reference_main, 'width2': width2_main}[args.command](remaining)


if __name__ == '__main__':
    main()
