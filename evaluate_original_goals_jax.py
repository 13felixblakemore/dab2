"""Short-horizon native-continuous goal-reaching evaluation; no retraining."""
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

from continuous_four_rooms_jax import ContinuousFourRooms,Point,_move
from forward_backward_jax import FBLearner,mask_action_scores
from control_bfs_bottlenecks import build_graph,room_ids,bfs_distances,ROOMS

POLICIES=['greedy_online','epsilon_online','greedy_target','epsilon_target','softmax_target','random']


def setup_pairs(env,count,seed):
    nodes,graph=build_graph(env,20)  # Reference grid spacing equals step size .05.
    room=room_ids(nodes,env.size);valid=np.flatnonzero(room>=0)
    rng=np.random.default_rng(seed);u=[];g=[]
    while len(u)<count:
        a=rng.choice(valid,count*2);b=rng.choice(valid,count*2)
        accept=(room[a]!=room[b])&(np.max(abs(nodes[a]-nodes[b]),axis=1)>env.threshold_distance)
        u.extend(a[accept].tolist());g.extend(b[accept].tolist())
    u=np.asarray(u[:count]);g=np.asarray(g[:count])
    lengths=[];cache={}
    for start,goal in zip(u,g):
        if goal not in cache:
            terminal=np.max(abs(nodes-nodes[goal]),axis=1)<=env.threshold_distance
            cache[goal]=bfs_distances(graph,terminal)
        length=cache[goal][start]
        if length<1:raise ValueError('Invalid or disconnected reference pair')
        lengths.append(length)
    return nodes[u].astype(np.float32),nodes[g].astype(np.float32),room[u],room[g],np.asarray(lengths)


def make_rollout(l,env,horizon,policy):
    action_vectors=jnp.asarray([(a.x,a.y) for a in env.actions])
    walls=tuple(env.wall_blocks)
    mask=jnp.zeros(l.n_actions,bool).at[jnp.array(env.get_available_actions(None))].set(True)
    @jax.jit
    def rollout(params,starts,goals,tasks,keys):
        n=len(starts);hits=jnp.full(n,horizon+1,jnp.int32);blocked=jnp.zeros(n,jnp.int32)
        def step(carry,t):
            position,hits,blocked,keys=carry
            split=jax.vmap(lambda key:jax.random.split(key,3))(keys)
            keys,ak,mk=split[:,0],split[:,1],split[:,2]
            if policy=='random':logits=jnp.broadcast_to(jnp.where(mask,0.,-jnp.inf),(n,l.n_actions))
            else:
                f=l.forward_network.apply(params,l.featuriser.transform(position),tasks)
                scores=jnp.einsum('nda,nd->na',f,tasks)
                logits=mask_action_scores(scores,mask)
            if policy.startswith('greedy'):selected=jnp.argmax(logits,axis=1)
            elif policy.startswith('epsilon'):
                split_action=jax.vmap(lambda k:jax.random.split(k,2))(ak)
                explore=jax.vmap(lambda k:jax.random.uniform(k)<.02)(split_action[:,0])
                random_action=jax.vmap(lambda k:jax.random.categorical(k,jnp.where(mask,0.,-jnp.inf)))(split_action[:,1])
                selected=jnp.where(explore,random_action,jnp.argmax(logits,axis=1))
            else:
                if policy=='softmax_target':logits=logits/l.temperature
                selected=jax.vmap(lambda k,row:jax.random.categorical(k,row))(ak,logits)
            def move(p,a,key):
                delta=action_vectors[a]
                nxt,_,collision=_move(Point(*p),Point(*delta),key,env._noise,env.size,walls)
                return jnp.array((nxt.x,nxt.y)),collision
            proposed,collision=jax.vmap(move)(position,selected,mk)
            active=hits>horizon
            position=jnp.where(active[:,None],proposed,position);blocked+=active&collision
            reached=jnp.all(jnp.abs(position-goals)<=env.threshold_distance,axis=1)
            hits=jnp.where(active&reached,t+1,hits)
            return (position,hits,blocked,keys),None
        (_,hits,blocked,_),_=jax.lax.scan(step,(starts,hits,blocked,keys),jnp.arange(horizon))
        return hits,blocked
    return rollout


def run(args):
    if min(args.samples,args.rollouts,args.horizon,args.batch_pairs)<1:raise ValueError('Counts must be positive')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    l=FBLearner.from_checkpoint(args.checkpoint,device=args.device,buffer_device='cpu')
    with jax.default_device(l.device):
        env=ContinuousFourRooms(include_stay_action=l.n_actions==5)
        starts,goals,sroom,groom,reference=setup_pairs(env,args.samples,args.seed)
        print(f'Noiseless center-lattice BFS reference: min={reference.min()}, median={np.median(reference)}, max={reference.max()}',flush=True)
        if args.horizon<reference.max():raise ValueError('Horizon shorter than a sampled reference path')
        tasks=l.embed_B(jnp.asarray(goals))
        keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(args.seed+1),i))(jnp.arange(args.samples*args.rollouts))
        results=[];per_room=[];saved={}
        for policy in POLICIES:
            print('Evaluating '+policy,flush=True)
            fn=make_rollout(l,env,args.horizon,policy)
            params=l.state['target' if policy.endswith('_target') else 'params']['f']
            all_hits=[];all_collisions=[]
            for first in range(0,args.samples,args.batch_pairs):
                end=min(first+args.batch_pairs,args.samples)
                hits,collisions=fn(params,jnp.repeat(jnp.array(starts[first:end]),args.rollouts,axis=0),
                    jnp.repeat(jnp.array(goals[first:end]),args.rollouts,axis=0),
                    jnp.repeat(tasks[first:end],args.rollouts,axis=0),keys[first*args.rollouts:end*args.rollouts])
                all_hits.append(np.asarray(hits));all_collisions.append(np.asarray(collisions))
            hits=np.concatenate(all_hits).reshape(args.samples,args.rollouts)
            collisions=np.concatenate(all_collisions).reshape(args.samples,args.rollouts)
            saved[policy+'_hits']=hits
            def summary(h,c):
                success=h<=args.horizon;good=h[success]
                row=dict(success_rate=float(success.mean()),mean_success_steps=float(good.mean()) if len(good) else None,
                         median_success_steps=float(np.median(good)) if len(good) else None,
                         mean_capped_steps=float(np.minimum(h,args.horizon).mean()),
                         wall_action_fraction=float(c.sum()/np.minimum(h,args.horizon).sum()))
                for limit in (25,50,args.horizon):
                    if limit<=args.horizon:row['success_by_'+str(limit)]=float(np.mean(h<=limit))
                return row
            row=dict(policy=policy,**summary(hits,collisions));results.append(row)
            for s in range(4):
                for g in range(4):
                    selected=(sroom==s)&(groom==g)
                    if selected.any():per_room.append(dict(policy=policy,start_room=ROOMS[s],goal_room=ROOMS[g],pairs=int(selected.sum()),**summary(hits[selected],collisions[selected])))
            print(json.dumps(row),flush=True)
    np.savez_compressed(out/'trials.npz',starts=starts,goals=goals,start_rooms=sroom,goal_rooms=groom,
                        bfs_reference_lengths=reference,**saved)
    metrics=dict(checkpoint=args.checkpoint,checkpoint_sha256=digest,config=l.config,seed=args.seed,
        samples=args.samples,rollouts=args.rollouts,horizon=args.horizon,
        dynamics=dict(size=env.size,step_size=.05,movement_noise=env._noise,goal_tolerance=env.threshold_distance,goal_norm='L-infinity, exact environment predicate'),
        reference=dict(min=int(reference.min()),median=float(np.median(reference)),max=int(reference.max()),
            meaning='Noiseless BFS on .05-spaced free cell centers; a restricted-lattice reference, not an exact global optimum under noisy continuous dynamics'),
        results=results,notes=['Cross-room endpoint pairs sampled from free .05-spaced lattice centers, excluding starts already at goal.',
                            'All evaluations use native continuous movement and noise, not the BFS graph transitions.',
                            'Goal task is online B(goal); all available actions include wall-rejected moves.',
                            'Each trial has the same policy/noise keys across conditions. First successful entry freezes the trial.',
                            'Successful step statistics exclude failures; capped means include failures at the horizon.',
                            'Checkpoint and learner were not trained or modified.'])
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    for filename,rows in (('summary.csv',results),('room_pairs.csv',per_room)):
        with (out/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    for policy in POLICIES:
        h=saved[policy+'_hits'];ax.plot(np.arange(args.horizon+1),[np.mean(h<=t) for t in range(args.horizon+1)],label=policy)
    ax.set_xlabel('Action steps');ax.set_ylabel('Fraction of trials reaching goal');ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'success_by_step.png',dpi=160);plt.close(fig)
    report=['# Original FB short-horizon native goal evaluation','',f'{args.samples} cross-room pairs, {args.rollouts} noisy trials each; limit {args.horizon} actions.',
            '',f'Noiseless restricted-lattice reference: {reference.min()} to {reference.max()} actions; median {np.median(reference)}.',
            '', '| Policy | Success by 25 | Success by limit | Mean successful steps | Median successful steps |','|---|---:|---:|---:|---:|']
    for r in results:
        mean='N/A' if r['mean_success_steps'] is None else f'{r["mean_success_steps"]:.1f}'
        median='N/A' if r['median_success_steps'] is None else f'{r["median_success_steps"]:.1f}'
        report.append(f'| {r["policy"]} | {r.get("success_by_25",float("nan")):.1%} | {r["success_rate"]:.1%} | {mean} | {median} |')
    report+=['',*metrics['notes'],'','Run: python evaluate_original_goals_jax.py. Options select --checkpoint, --samples, --rollouts, --horizon, --seed, --device, --output.']
    (out/'RESULTS.md').write_text('\n'.join(report))
    assert hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()==digest
    return metrics


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default='models/fb_fourrooms_jax_original.pkl');p.add_argument('--samples',type=int,default=1000)
    p.add_argument('--rollouts',type=int,default=8);p.add_argument('--horizon',type=int,default=50)
    p.add_argument('--batch-pairs',type=int,default=100);p.add_argument('--seed',type=int,default=2029)
    p.add_argument('--device',default='cpu');p.add_argument('--output',default='evaluation_original_short_horizon')
    run(p.parse_args())
