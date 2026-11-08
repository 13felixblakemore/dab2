"""Fixed doorway advantages and paired Monte Carlo first-action evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms
from fb_navigation.environments.continuous_four_rooms_jax import Point, _move
from fb_navigation.diagnostics.control_bfs_bottlenecks import build_graph, room_ids
from fb_navigation.diagnostics.investigate_doorway_decline import door_rectangles, nominal_transitions
from fb_navigation.agents.forward_backward_jax import FBLearner

ACTIONS = ('LEFT', 'RIGHT', 'UP', 'DOWN')


def advantages(q):
    """Q(a) - max_{b != a} Q(b), including zero advantages for top ties."""
    alternatives = np.where(np.eye(q.shape[-1], dtype=bool), -np.inf, q[..., None, :])
    return q-np.max(alternatives, axis=-1)


def discounted_hits(hits, horizon, gamma=.99):
    return np.where(hits <= horizon, gamma**(hits-1), 0.)


def make_probes(env, seed):
    rng = np.random.default_rng(seed)
    rectangles = door_rectangles(env.doorway_scale)*env.size
    connections = [(0,2),(1,3),(0,1),(2,3)]
    nodes, _ = build_graph(env, 20)
    rooms = room_ids(nodes, env.size)
    rows = []
    for door, (rectangle, (a, b)) in enumerate(zip(rectangles, connections)):
        for side, source in enumerate((a,b)):
            adjacent = b if source == a else a
            for offset in (-.6,-.3,0,.3,.6):
                for distance in (.025,.05):
                    if door < 2:
                        state = np.array([rectangle[0]-distance if side == 0 else rectangle[2]+distance,
                            (rectangle[1]+rectangle[3])/2+offset*(rectangle[3]-rectangle[1])])
                    else:
                        state = np.array([(rectangle[0]+rectangle[2])/2+offset*(rectangle[2]-rectangle[0]),
                            rectangle[1]-distance if side == 0 else rectangle[3]+distance])
                    if any(w.contains(Point(*state)) for w in env.wall_blocks):
                        raise ValueError('Doorway probe falls inside a wall')
                    for kind, goal_room in [('adjacent',adjacent),('diagonal',source^3)]:
                        choices = np.flatnonzero((rooms == goal_room)&(np.max(abs(nodes-state),axis=1) > env.threshold_distance))
                        goal = nodes[rng.choice(choices)]
                        rows.append((state,goal,door,side,offset,distance,kind))
    return dict(states=np.asarray([r[0] for r in rows],np.float32),
                goals=np.asarray([r[1] for r in rows],np.float32),
                door=np.asarray([r[2] for r in rows]), side=np.asarray([r[3] for r in rows]),
                offset=np.asarray([r[4] for r in rows]), distance=np.asarray([r[5] for r in rows]),
                goal_kind=np.asarray([r[6] for r in rows]))


def mc_function(learner, env, horizon):
    walls = tuple(env.wall_blocks)
    actions = jnp.array([(a.x,a.y) for a in env.actions])
    @jax.jit
    def rollout(params, starts, goals, first_actions, keys):
        tasks = learner.backward_network.apply(params['b'],learner.featuriser.transform(goals))
        def step(carry,t):
            position,hits,keys,first_collision = carry
            split = jax.vmap(lambda k:jax.random.split(k,4))(keys)
            keys,ek,ak,mk = [split[:,i] for i in range(4)]
            f = learner.forward_network.apply(params['f'],learner.featuriser.transform(position),tasks)
            greedy = jnp.argmax(jnp.einsum('bda,bd->ba',f,tasks),axis=1)
            explore = jax.vmap(lambda k:jax.random.uniform(k)<.02)(ek)
            random_action = jax.vmap(lambda k:jax.random.randint(k,(),0,4))(ak)
            selected = jnp.where(t==0,first_actions,jnp.where(explore,random_action,greedy))
            def move(p,a,key):
                nxt,_,blocked = _move(Point(*p),Point(*actions[a]),key,env._noise,env.size,walls)
                return jnp.array([nxt.x,nxt.y]),blocked
            following,blocked = jax.vmap(move)(position,selected,mk)
            active = hits>horizon
            position = jnp.where(active[:,None],following,position)
            reached = jnp.all(abs(position-goals)<=env.threshold_distance,axis=1)
            hits = jnp.where(active&reached,t+1,hits)
            first_collision = jnp.where(t==0,blocked,first_collision)
            return (position,hits,keys,first_collision),None
        initial = (starts,jnp.full(len(starts),horizon+1,jnp.int32),keys,jnp.zeros(len(starts),bool))
        final,_ = jax.lax.scan(step,initial,jnp.arange(horizon))
        return final[1],final[3]
    return rollout


def evaluate_actions(learner,env,probes,rollouts,horizon,seed,batch_size):
    n = len(probes['states'])
    # Common random streams across first actions AND checkpoints.
    pids = np.repeat(np.arange(n),4*rollouts)
    first = np.tile(np.repeat(np.arange(4),rollouts),n)
    replica = np.tile(np.arange(rollouts),n*4)
    ids = pids*rollouts+replica
    fn = mc_function(learner,env,horizon)
    hits,blocked = [],[]
    for begin in range(0,len(ids),batch_size):
        sl = slice(begin,begin+batch_size)
        keys = jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(seed),i))(jnp.asarray(ids[sl],jnp.uint32))
        h,b = fn(learner.state['params'],jnp.asarray(probes['states'][pids[sl]]),
            jnp.asarray(probes['goals'][pids[sl]]),jnp.asarray(first[sl]),keys)
        hits.append(np.asarray(h)); blocked.append(np.asarray(b))
    return np.concatenate(hits).reshape(n,4,rollouts),np.concatenate(blocked).reshape(n,4,rollouts)


def interval(values):
    """Bootstrap across fixed state/goal probes; descriptive, not seed uncertainty."""
    if not len(values):return None
    rng = np.random.default_rng(4001)
    means = values[rng.integers(len(values),size=(4000,len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(means,[.025,.975])]


def comparison(previous,current,horizon,gamma=.99):
    old,new = previous['q'].argmax(1),current['q'].argmax(1)
    flip = old!=new
    rows = np.arange(len(old))
    prev_returns = discounted_hits(previous['hits'],horizon,gamma)
    curr_returns = discounted_hits(current['hits'],horizon,gamma)
    # Fix continuation to current checkpoint when comparing first actions.
    paired = curr_returns[rows,new]-curr_returns[rows,old]
    means = paired.mean(axis=1)
    se = paired.std(axis=1,ddof=1)/np.sqrt(paired.shape[1])
    entire_change = curr_returns[rows,new].mean(axis=1)-prev_returns[rows,old].mean(axis=1)
    old_continuation = (prev_returns[rows,new]-prev_returns[rows,old]).mean(axis=1)
    return dict(probes=len(old),preferred_action_change_fraction=float(flip.mean()),changed_probes=int(flip.sum()),
        flip_to_blocked=int(np.sum(flip&current['blocked'][rows,new])),
        flip_from_blocked=int(np.sum(flip&previous['blocked'][rows,old])),
        switch_gain_current_continuation=float(means[flip].mean()) if flip.any() else None,
        switch_gain_current_ci=interval(means[flip]),
        switch_gain_previous_continuation=float(old_continuation[flip].mean()) if flip.any() else None,
        switch_return_improved=int(np.sum(flip&(means>1e-12))),
        switch_return_worsened=int(np.sum(flip&(means < -1e-12))),
        switch_return_tied=int(np.sum(flip&(abs(means)<=1e-12))),
        switch_improved_beyond_approx_mc_noise=int(np.sum(flip&(means>1.96*se))),
        switch_worsened_beyond_approx_mc_noise=int(np.sum(flip&(means < -1.96*se))),
        own_policy_return_change_on_changed=float(entire_change[flip].mean()) if flip.any() else None,
        own_policy_return_change_all=float(entire_change.mean()),
        own_policy_change_all_ci=interval(entire_change))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('experiments/fourrooms_doorways/runs/width2_seed0'))
    parser.add_argument('--output',type=Path,default=Path('experiments/fourrooms_doorways/analyses/advantage_stability'))
    parser.add_argument('--epochs',nargs='+',type=int,default=[100,200,400,600,800])
    parser.add_argument('--rollouts',type=int,default=32)
    parser.add_argument('--horizon',type=int,default=150)
    parser.add_argument('--batch-size',type=int,default=512)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--seed',type=int,default=63017)
    args = parser.parse_args()
    if args.rollouts<2 or args.horizon<30 or args.batch_size<1:
        parser.error('Need at least two rollouts, horizon >=30, and positive batch size')
    args.output.mkdir(parents=True,exist_ok=True)
    config = json.loads((args.run/'config.json').read_text())
    env = AdjustableFourRooms(**config['environment'])
    settings = dict(run=str(args.run.resolve()),rollouts=args.rollouts,horizon=args.horizon,
        seed=args.seed,device=args.device,jax=jax.__version__,numpy=np.__version__,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    manifest = args.output/'config.json'
    if manifest.exists() and json.loads(manifest.read_text())!=settings:
        raise ValueError('Suite changed; choose a new output directory')
    manifest.write_text(json.dumps(settings,indent=2))
    probes = make_probes(env,args.seed)
    np.savez_compressed(args.output/'fixed_probes.npz',**probes)
    _,blocked = nominal_transitions(env,probes['states'])
    available,missing,records,summary = [],[],{},dict(checkpoints={},comparisons={})
    for epoch in sorted(set(args.epochs)):
        path = args.run/f'epoch_{epoch:04d}.pkl'
        if not path.exists():missing.append(epoch);continue
        available.append(epoch)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        cache = args.output/f'epoch_{epoch:04d}.npz'
        if cache.exists():
            with np.load(cache) as data: record = {k:data[k] for k in data.files}
            if str(record['checkpoint_sha256'])!=digest:raise ValueError('Checkpoint changed')
        else:
            learner = FBLearner.from_checkpoint(path,device=args.device)
            q_parts=[]
            for begin in range(0,len(probes['states']),128):
                z=learner.embed_B(probes['goals'][begin:begin+128])
                f=learner.embed_F(probes['states'][begin:begin+128],z)
                q_parts.append(np.asarray(jnp.einsum('bda,bd->ba',f,z)))
            q=np.concatenate(q_parts)
            print(f'Epoch {epoch}: evaluating {len(q)} probes x 4 actions x {args.rollouts} MC trials',flush=True)
            hits,first_blocked=evaluate_actions(learner,env,probes,args.rollouts,args.horizon,args.seed+1,args.batch_size)
            record=dict(q=q,advantages=advantages(q),hits=hits,first_collision=first_blocked,
                        blocked=blocked,checkpoint_sha256=np.array(digest))
            np.savez_compressed(cache,**record)
        records[epoch]=record
        q,hit=record['q'],record['hits']; selected=q.argmax(1); rows=np.arange(len(q))
        entry=dict(blocked_preferred_fraction=float(blocked[rows,selected].mean()),
            mean_preferred_advantage=float(record['advantages'][rows,selected].mean()),
            median_preferred_advantage=float(np.median(record['advantages'][rows,selected])))
        for horizon in (30,args.horizon):
            ret=discounted_hits(hit,horizon,config['learner']['discount_factor'])
            mean=ret.mean(axis=2); value=mean[rows,selected]
            entry[f'h{horizon}']=dict(success_rate=float(np.mean(hit[rows,selected]<=horizon)),
                discounted_return=float(value.mean()),discounted_return_ci=interval(value),
                mean_empirical_mc_regret=float(np.mean(mean.max(1)-value)),
                mc_best_action_fraction=float(np.mean(value>=mean.max(1)-1e-12)))
        summary['checkpoints'][epoch]=entry
        print(json.dumps(dict(epoch=epoch,**entry)),flush=True)
    comparisons=list(zip(available[:-1],available[1:]))
    if 400 in available and 800 in available and (400,800) not in comparisons:comparisons.append((400,800))
    for old,new in comparisons:
        for horizon in (30,args.horizon):
            summary['comparisons'][f'{old}_{new}_{horizon}']=comparison(records[old],records[new],horizon,config['learner']['discount_factor'])
    summary.update(available=available,missing=missing,probes=len(probes['states']),
        states=len(np.unique(probes['states'],axis=0)),rollouts=args.rollouts,
        notes=['First action forced; then online epsilon=.02 policy. Return gamma^(hit-1) once on success, zero if censored.',
               'Ordinal navigation validation: learned continuing FB Q values are not calibrated to stopped sparse-goal returns.',
               'Shared noise/action keys across actions and checkpoints. Switch effects fix the continuation checkpoint.',
               'Intervals resample fixed probes; they do not quantify training-seed uncertainty. MC maximum regret is upward biased.',
               'Small differences between MC actions may be noise. Epoch 100 cannot be recovered from later checkpoints.'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    if available:plot_results(args.output,probes,records,summary,blocked)
    print(json.dumps(dict(available=available,missing=missing)),flush=True)


def plot_results(output,probes,records,summary,blocked):
    epochs=sorted(records)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    axes[0,0].plot(epochs,[summary['checkpoints'][e]['blocked_preferred_fraction'] for e in epochs],'o-')
    axes[0,0].set(ylabel='Blocked greedy action fraction',ylim=(0,1))
    for horizon in (30,max(int(k[1:]) for k in summary['checkpoints'][epochs[0]] if k.startswith('h'))):
        axes[0,1].plot(epochs,[summary['checkpoints'][e][f'h{horizon}']['discounted_return'] for e in epochs],'o-',label=f'H={horizon}')
    axes[0,1].set(ylabel='Mean discounted MC goal return');axes[0,1].legend()
    changes=list(zip(epochs[:-1],epochs[1:]))
    axes[1,0].bar([f'{a}->{b}' for a,b in changes],[np.mean(records[a]['q'].argmax(1)!=records[b]['q'].argmax(1)) for a,b in changes])
    axes[1,0].set(ylabel='Preferred action changes',ylim=(0,1))
    choices=records[epochs[-1]]['q'].argmax(1)
    axes[1,1].scatter(*probes['states'].T,c=blocked[np.arange(len(choices)),choices],cmap='coolwarm',s=15)
    axes[1,1].set(xlabel='x',ylabel='y',title=f'Final probes: red = blocked preference',aspect='equal')
    for ax in [axes[0,0],axes[0,1]]:ax.set_xlabel('Epoch')
    fig.tight_layout();fig.savefig(output/'stability.png',dpi=160);plt.close(fig)
    if 400 in records and 800 in records:
        flipped=records[400]['q'].argmax(1)!=records[800]['q'].argmax(1)
        ids=np.flatnonzero(flipped)[:8]
        fig,axes=plt.subplots(2,4,figsize=(15,7))
        for ax,i in zip(axes.flat,ids):
            for a,name in enumerate(ACTIONS):
                ax.plot(epochs,[records[e]['advantages'][i,a] for e in epochs],'o-',label=name)
            ax.axhline(0,color='gray');ax.set(title=f'Probe {i}, door {probes["door"][i]}',xlabel='Epoch',ylabel='Action advantage')
        axes.flat[0].legend(fontsize=7);fig.tight_layout();fig.savefig(output/'advantage_traces.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
