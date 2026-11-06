"""Matched Monte Carlo policy x discount ablation on the fixed BFS-control graph.

Loads saved endpoint pairs. Both policies share graph transitions, goal regions,
candidate cells and rollouts/pair. Both discounts reuse identical trajectories.
"""
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
from matplotlib.patches import Rectangle

from continuous_four_rooms_jax import ContinuousFourRooms
from forward_backward_jax import FBLearner
from control_bfs_bottlenecks import bfs_distances,room_ids
from estimate_fb_bottlenecks_jax import grid_regions,eligibility


def policy_probabilities(scores,mode,temperature,epsilon):
    if mode == 'softmax':
        return jax.nn.softmax(scores/temperature,axis=-1)
    if mode == 'epsilon':
        return (1-epsilon)*jax.nn.one_hot(jnp.argmax(scores,axis=-1),scores.shape[-1])+epsilon/scores.shape[-1]
    raise ValueError('Unknown policy mode')


def precompute_policy(learner,nodes,goals,batch,mode='softmax',target_network=True,epsilon=.02):
    """Tabulate a frozen policy once; rollouts then need no networks.

    All four actions are available in this environment, including wall-rejected
    moves. Epsilon exploration is uniform over those same available actions.
    """
    if mode not in ('softmax','epsilon') or not 0<=epsilon<=1:
        raise ValueError('Invalid policy mode or epsilon')
    features=learner.featuriser.transform(jnp.asarray(nodes,dtype=jnp.float32))
    tasks=learner.embed_B(jnp.asarray(goals,dtype=jnp.float32))
    @jax.jit
    def probabilities(params,zs):
        def one(z):
            w=jnp.broadcast_to(z,(len(nodes),learner.embed_dim))
            f=learner.forward_network.apply(params,features,w)
            return policy_probabilities(jnp.einsum('nda,d->na',f,z),mode,learner.temperature,epsilon)
        return jax.vmap(one)(zs)
    chunks=[]
    for i in range(0,len(goals),batch):
        params=learner.state['target' if target_network else 'params']['f']
        chunks.append(np.asarray(probabilities(params,tasks[i:i+batch])))
        if i%max(batch,len(goals)//5//batch*batch)==0:print(f'Policy goals {min(i+batch,len(goals))}/{len(goals)}',flush=True)
    return np.concatenate(chunks)


def simulate_fb(neighbours,terminal,starts,goal_rows,node_candidate,policy,rollouts,horizon,seed):
    rng=np.random.default_rng(seed)
    current=np.repeat(starts,rollouts);g=np.repeat(goal_rows,rollouts)
    n=len(current);first=np.full((n,int(node_candidate.max())+1),-1,np.int32)
    hit_goal=np.full(n,horizon+1,int);active=~terminal[g,current]
    hit_goal[~active]=0
    blocked=np.zeros(n,int)
    for t in range(horizon):
        idx=np.flatnonzero(active)
        if len(idx)==0:break
        candidate=node_candidate[current[idx]];good=candidate>=0
        rows=idx[good];col=candidate[good];unseen=first[rows,col]<0
        first[rows[unseen],col[unseen]]=t
        probs=policy[g[idx],current[idx]].astype(np.float64)
        probs/=probs.sum(axis=1,keepdims=True)
        selected=np.argmax(np.cumsum(probs,axis=1)>rng.random(len(idx))[:,None],axis=1)
        nxt=neighbours[current[idx],selected]
        stay=nxt<0;blocked[idx]+=stay
        current[idx]=np.where(stay,current[idx],nxt)
        reached=terminal[g[idx],current[idx]]
        hit_goal[idx[reached]]=t+1;active[idx[reached]]=False
        if (t+1)%1000==0:print(f'FB step {t+1}/{horizon}: {active.sum()}/{n} active',flush=True)
    return first,hit_goal,blocked


def estimate(first,eligible,rollouts,gamma):
    # -1 is censored/unvisited. Preserve visits at t=0, including for gamma=1.
    values=np.where(first>=0,gamma**np.maximum(first,0),0.)
    pair=values.reshape(len(eligible),rollouts,-1).mean(axis=1)
    count=eligible.sum(axis=0)
    beta=np.divide(np.where(eligible,pair,0).sum(0),count,out=np.full(pair.shape[1],np.nan),where=count>0)
    return beta,pair


def door_statistics(points,beta):
    x,y=(points*11).T
    door_masks=dict(bottom=(x>=6)&(x<=7)&(y>=1)&(y<=2),top=(x>=6)&(x<=7)&(y>=8)&(y<=9),
                    left=(x>=2)&(x<=3)&(y>=4)&(y<=5),right=(x>=9)&(x<=10)&(y>=4)&(y<=5))
    union=np.logical_or.reduce(list(door_masks.values()));finite=np.isfinite(beta)
    stats={}
    for name,mask in door_masks.items():
        values=beta[mask & finite]
        stats[name]=dict(mean=float(values.mean()) if len(values) else None,
                         percentile=float(np.mean(beta[finite]<=values.mean())) if len(values) else None)
    return dict(doors=stats,doorway_enrichment=float(np.mean(beta[union & finite])/np.mean(beta[~union & finite])),
                mean_doorway_score=float(np.mean(beta[union & finite])),mean_other_score=float(np.mean(beta[~union & finite])))


def path_statistics(hit,horizon,shortest,rollouts):
    success=hit<=horizon;length=hit[success]
    baseline=np.repeat(shortest,rollouts)
    return dict(success_rate=float(success.mean()),unfinished=int((~success).sum()),
                mean_successful_steps=float(length.mean()) if len(length) else None,
                median_successful_steps=float(np.median(length)) if len(length) else None,
                mean_capped_steps=float(np.minimum(hit,horizon).mean()),
                median_capped_steps=float(np.median(np.minimum(hit,horizon))),
                mean_successful_length_ratio_to_bfs=float(np.mean(hit[success]/baseline[success])) if len(length) else None)


def run(args):
    if min(args.horizon,args.goal_batch)<1:raise ValueError('Counts must be positive')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    source=Path(args.pairs)
    with np.load(source) as d:
        nodes=d['nodes'];neighbours=d['neighbours'];starts=d['starts'];goals=d['goals']
        keep=d['candidate_mask'];points=d['candidate_points'];eligible=d['eligible']
        saved_first=d['first_hits'];shortest=d['shortest_lengths']
    n=len(starts);saved_rollouts=len(saved_first)//n
    k=saved_rollouts if args.rollouts is None else args.rollouts
    if k<1 or k>saved_rollouts:raise ValueError('Requested rollouts must be within saved BFS rollout count')
    resolution=int(round(np.sqrt(len(keep))))
    learner=FBLearner.from_checkpoint(args.checkpoint,device=args.device,buffer_device='cpu')
    if learner.n_actions!=4:raise ValueError('The shared graph has four actions')
    env=ContinuousFourRooms(include_stay_action=False)
    _,lo,hi,current_keep=grid_regions(env,resolution)
    np.testing.assert_array_equal(current_keep,keep)
    np.testing.assert_array_equal(eligible,eligibility(nodes[goals],lo[keep],hi[keep],learner.goal_threshold))
    self_rooms=room_ids(nodes,1.);assert np.all(self_rooms[starts]!=self_rooms[goals])
    unique,inverse=np.unique(goals,return_inverse=True)
    terminal=np.sum((nodes[None]-nodes[unique,None])**2,axis=-1)<=learner.goal_threshold**2
    distances=np.stack([bfs_distances(neighbours,t) for t in terminal])
    np.testing.assert_array_equal(distances[inverse,starts],shortest)
    if args.horizon<int(shortest.max()):raise ValueError('Horizon shorter than a BFS path')
    xy=np.clip(np.floor(nodes*resolution).astype(int),0,resolution-1)
    flat=xy[:,1]*resolution+xy[:,0]
    index=np.full(len(keep),-1,int);index[np.flatnonzero(keep)]=np.arange(keep.sum())
    node_candidate=index[flat]
    bfs_first=saved_first.reshape(n,saved_rollouts,-1)[:,:k].reshape(n*k,-1)
    with jax.default_device(learner.device):
        policy=precompute_policy(learner,nodes,nodes[unique],args.goal_batch)
    fb_first,fb_goal_hits,blocked=simulate_fb(neighbours,terminal,starts,inverse,node_candidate,policy,k,args.horizon,args.seed)
    bfs_goal_hits=np.repeat(shortest,k)
    paths=dict(BFS=path_statistics(bfs_goal_hits,args.horizon,shortest,k),
               FB=path_statistics(fb_goal_hits,args.horizon,shortest,k))
    paths['FB']['blocked_action_fraction']=float(blocked.sum()/np.minimum(fb_goal_hits,args.horizon).sum())
    maps={};statistics={};raw={}
    for name,first in (('BFS',bfs_first),('FB',fb_first)):
        for gamma in (1.,.99):
            label=f'{name}_gamma_{gamma:g}'
            beta,pair=estimate(first,eligible,k,gamma)
            maps[label]=beta;raw[label+'_pair']=pair
            statistics[label]=door_statistics(points,beta)
    unknown=((fb_first<0)&(fb_goal_hits>args.horizon)[:,None]).reshape(n,k,-1).mean(axis=1)
    tail=np.where(eligible,unknown,0.).sum(axis=0)/eligible.sum(axis=0)
    raw['FB_gamma_1_upper_bound']=maps['FB_gamma_1']+tail
    metrics=dict(checkpoint=args.checkpoint,checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        pairs_file=str(source),pairs_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        samples=n,rollouts_per_pair=k,horizon=args.horizon,seed=args.seed,candidate_grid=resolution,
        goal_threshold=learner.goal_threshold,temperature=learner.temperature,graph_nodes=len(nodes),
        policy='BFS uniform shortest-action ties vs frozen FB target-softmax with z=B_online(goal)',
        dynamics='identical deterministic four-neighbor graph; invalid movement actions stay in place',
        paths=paths,maps=statistics,
        fb_undiscounted_max_remaining_cell_probability=float(np.max(tail)),
        mean_policy_entropy=float(np.mean(-np.sum(policy*np.log(np.maximum(policy,1e-30)),axis=-1))),
        caveats=['Gamma=1 maps are finite-horizon lower bounds if unfinished FB trajectories remain.',
                 'Discounts reweight exactly the same trajectories within each policy; discount cannot change goal success or path lengths.',
                 'This controls graph dynamics across policies. FB was trained on continuous noisy movements of a different step size, so its graph performance is a transfer diagnostic.',
                 'The FB gamma=.99 panel is the original discount/policy combination, not an exact reproduction of the earlier continuous-dynamics experiment.',
                 'Both estimators use identical saved cross-room pairs, candidate grid and per-pair goal-overlap exclusions.',
                 'Endpoint sampling, finite cell resolution and policy tie behavior affect rankings. No model is retrained.'])
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    np.savez_compressed(out/'intermediate.npz',nodes=nodes,starts=starts,goals=goals,candidate_points=points,candidate_mask=keep,
        eligible=eligible,bfs_first=bfs_first,fb_first=fb_first,fb_goal_hits=fb_goal_hits,bfs_goal_hits=bfs_goal_hits,**maps,**raw)
    fig,axes=plt.subplots(2,2,figsize=(11,10),constrained_layout=True)
    max_score=max(np.nanmax(m) for m in maps.values())
    for ax,(label,beta) in zip(axes.ravel(),maps.items()):
        values=np.full(len(keep),np.nan);values[keep]=beta
        image=ax.imshow(values.reshape(resolution,resolution),origin='lower',extent=(0,1,0,1),vmin=0,vmax=max_score,cmap='viridis')
        for w in env.wall_blocks:ax.add_patch(Rectangle((w.xmin,w.ymin),w.xmax-w.xmin,w.ymax-w.ymin,color='black'))
        ax.set_title(label);ax.set_aspect('equal');fig.colorbar(image,ax=ax,label='Mean first-hit weight')
    fig.savefig(out/'four_way_heatmaps.png',dpi=180);plt.close(fig)
    # Separate normalized display scales reveal structure even if amplitudes differ.
    fig,axes=plt.subplots(2,2,figsize=(11,10),constrained_layout=True)
    for ax,(label,beta) in zip(axes.ravel(),maps.items()):
        values=np.full(len(keep),np.nan);values[keep]=beta
        image=ax.imshow(values.reshape(resolution,resolution),origin='lower',extent=(0,1,0,1),vmin=0,cmap='viridis')
        for w in env.wall_blocks:ax.add_patch(Rectangle((w.xmin,w.ymin),w.xmax-w.xmin,w.ymax-w.ymin,color='black'))
        ax.set_title(label+' (own scale)');fig.colorbar(image,ax=ax)
    fig.savefig(out/'four_way_heatmaps_own_scales.png',dpi=180);plt.close(fig)
    report=['# Matched policy / discount comparison','',f'{n} identical saved cross-room pairs; {k} rollouts/pair; {resolution}x{resolution} candidate grid; horizon {args.horizon}.',
            '', '| Policy | Success | Mean successful steps | Median successful steps | Mean capped steps |','|---|---:|---:|---:|---:|']
    for name,r in paths.items():
        mean='N/A' if r['mean_successful_steps'] is None else f'{r["mean_successful_steps"]:.2f}'
        median='N/A' if r['median_successful_steps'] is None else f'{r["median_successful_steps"]:.1f}'
        report.append(f'| {name} | {r["success_rate"]:.2%} | {mean} | {median} | {r["mean_capped_steps"]:.2f} |')
    report+=['', '| Configuration | Door / other enrichment | Bottom percentile | Top percentile | Left percentile | Right percentile |',
             '|---|---:|---:|---:|---:|---:|']
    for name,r in statistics.items():
        cells=[f'{r["doors"][door]["percentile"]:.1%}' for door in ('bottom','top','left','right')]
        report.append(f'| {name} | {r["doorway_enrichment"]:.2f}x | '+' | '.join(cells)+' |')
    report+=['',*metrics['caveats'], '', 'Run: python compare_bottleneck_policies.py. The saved control pairs are required; --pairs selects their NPZ. --checkpoint, --horizon, --rollouts, --seed, --goal-batch, --device and --output are configurable.']
    (out/'RESULTS.md').write_text('\n'.join(report));print(json.dumps(metrics,indent=2),flush=True)
    return metrics


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',default='evaluation_bfs_control/intermediate.npz')
    p.add_argument('--checkpoint',default='models/fb_fourrooms_jax_goal.pkl')
    p.add_argument('--device',default='cpu');p.add_argument('--horizon',type=int,default=10000)
    p.add_argument('--rollouts',type=int,default=None);p.add_argument('--goal-batch',type=int,default=4)
    p.add_argument('--seed',type=int,default=2027);p.add_argument('--output',default='evaluation_policy_discount_control')
    run(p.parse_args())
