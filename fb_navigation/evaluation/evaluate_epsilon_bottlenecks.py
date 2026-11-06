"""Matched epsilon=.02 FB bottleneck Monte Carlo, gamma=1 and .99.

Reuses exact prior BFS-control pairs and deterministic graph. No training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from fb_navigation.environments.continuous_four_rooms_jax import ContinuousFourRooms
from fb_navigation.agents.forward_backward_jax import FBLearner
from fb_navigation.diagnostics.control_bfs_bottlenecks import bfs_distances
from fb_navigation.diagnostics.estimate_fb_bottlenecks_jax import grid_regions,eligibility
from fb_navigation.diagnostics.compare_bottleneck_policies import precompute_policy,simulate_fb,estimate,door_statistics,path_statistics


def doorway_arrivals(first,eligible,points,rollouts):
    x,y=(points*11).T
    masks=dict(bottom=(x>=6)&(x<=7)&(y>=1)&(y<=2),top=(x>=6)&(x<=7)&(y>=8)&(y<=9),
               left=(x>=2)&(x<=3)&(y>=4)&(y<=5),right=(x>=9)&(x<=10)&(y>=4)&(y<=5))
    allowed=np.repeat(eligible,rollouts,axis=0)
    def summarise(mask):
        observed=(first>=0)&allowed&mask[None]
        pooled=first[observed]
        any_first=np.min(np.where(observed,first,np.iinfo(np.int32).max),axis=1)
        arrived=any_first<np.iinfo(np.int32).max
        return dict(arrival_rate=float(arrived.mean()),
            mean_first_any_cell=float(any_first[arrived].mean()) if arrived.any() else None,
            median_first_any_cell=float(np.median(any_first[arrived])) if arrived.any() else None,
            mean_pooled_cell_arrival=float(pooled.mean()) if len(pooled) else None,
            median_pooled_cell_arrival=float(np.median(pooled)) if len(pooled) else None)
    return dict(any_door=summarise(np.logical_or.reduce(list(masks.values()))),
                per_door={name:summarise(mask) for name,mask in masks.items()})


def run(args):
    if min(args.horizon,args.goal_batch)<1 or not 0<=args.epsilon<=1:raise ValueError('Invalid configuration')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    pair_path=Path(args.pairs)
    with np.load(pair_path) as d:
        nodes=d['nodes'];graph=d['neighbours'];starts=d['starts'];goals=d['goals'];keep=d['candidate_mask']
        points=d['candidate_points'];eligible=d['eligible'];shortest=d['shortest_lengths']
        saved_rollouts=len(d['first_hits'])//len(starts)
    k=saved_rollouts if args.rollouts is None else args.rollouts
    if k<1:raise ValueError('Rollouts must be positive')
    n=len(starts);resolution=int(np.sqrt(len(keep)))
    l=FBLearner.from_checkpoint(args.checkpoint,device=args.device,buffer_device='cpu')
    if l.n_actions!=4:raise ValueError('Shared graph uses four actions')
    env=ContinuousFourRooms(include_stay_action=False)
    _,lo,hi,check_keep=grid_regions(env,resolution)
    np.testing.assert_array_equal(keep,check_keep)
    np.testing.assert_array_equal(eligible,eligibility(nodes[goals],lo[keep],hi[keep],l.goal_threshold))
    unique,inverse=np.unique(goals,return_inverse=True)
    terminal=np.sum((nodes[None]-nodes[unique,None])**2,axis=-1)<=l.goal_threshold**2
    distances=np.stack([bfs_distances(graph,t) for t in terminal])
    np.testing.assert_array_equal(distances[inverse,starts],shortest)
    if args.horizon<shortest.max():raise ValueError('Horizon shorter than BFS paths')
    xy=np.clip(np.floor(nodes*resolution).astype(int),0,resolution-1);flat=xy[:,1]*resolution+xy[:,0]
    index=np.full(len(keep),-1,int);index[np.flatnonzero(keep)]=np.arange(keep.sum());node_candidate=index[flat]
    metrics=dict(checkpoint=args.checkpoint,checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        pairs_file=args.pairs,pairs_sha256=hashlib.sha256(pair_path.read_bytes()).hexdigest(),samples=n,rollouts_per_pair=k,
        horizon=args.horizon,epsilon=args.epsilon,seed=args.seed,goal_threshold=l.goal_threshold,candidate_grid=resolution,
        checkpoint_z_sampling=l.z_sampling,checkpoint_goal_terminated=l.goal_terminated,
        goal_task='z=B_online(goal), unchanged scale',dynamics='same prior deterministic graph; blocked moves stay',policies={},maps={})
    raw=dict(nodes=nodes,starts=starts,goals=goals,candidate_points=points,candidate_mask=keep,eligible=eligible)
    maps={}
    for name,target in (('online',False),('target',True)):
        print('Evaluating epsilon-greedy '+name,flush=True)
        with jax.default_device(l.device):
            policy=precompute_policy(l,nodes,nodes[unique],args.goal_batch,mode='epsilon',target_network=target,epsilon=args.epsilon)
        first,hit,blocked=simulate_fb(graph,terminal,starts,inverse,node_candidate,policy,k,args.horizon,args.seed)
        stat=path_statistics(hit,args.horizon,shortest,k)
        stat['blocked_action_fraction']=float(blocked.sum()/np.minimum(hit,args.horizon).sum())
        stat['doorway_arrivals']=doorway_arrivals(first,eligible,points,k)
        metrics['policies'][name]=stat
        raw[name+'_first_hits']=first;raw[name+'_goal_hits']=hit
        for gamma in (1.,.99):
            label=f'{name}_gamma_{gamma:g}'
            beta,pair=estimate(first,eligible,k,gamma)
            maps[label]=beta;raw[label]=beta;raw[label+'_pair']=pair
            metrics['maps'][label]=door_statistics(points,beta)
        unknown=((first<0)&(hit>args.horizon)[:,None]).reshape(n,k,-1).mean(axis=1)
        tail=np.where(eligible,unknown,0).sum(0)/eligible.sum(0)
        raw[name+'_gamma_1_upper']=maps[name+'_gamma_1']+tail
        stat['max_remaining_cell_probability']=float(tail.max())
        print(name,json.dumps(stat),flush=True)
    metrics['caveats']=[
        'Gamma=1 estimates are lower bounds if any rollouts are unfinished. Conditional arrival means exclude trajectories that never arrive by the cap.',
        'First any-door arrival is a per-trajectory minimum across eligible doorway candidate cells. Pooled cell arrival matches the earlier 22/1701-step comparison and weights multiple doorway-cell visits separately.',
        'The goal disk, wall exclusions, candidate grid, endpoint pairs, graph dynamics and policy snapshots are held fixed. Gamma changes weights only.',
        'Blocked actions remain available, matching the environment. Epsilon=.02 samples uniformly over all four actions, including the greedy action.',
        'The graph step size and lack of noise differ from FB training; compare within this controlled graph, not directly with earlier continuous-navigation percentages.',
        'No checkpoint parameters or training hyperparameters were changed.']
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False));np.savez_compressed(out/'intermediate.npz',**raw)
    for shared,filename in ((True,'heatmaps_shared_scale.png'),(False,'heatmaps_own_scales.png')):
        fig,axes=plt.subplots(2,2,figsize=(11,10),constrained_layout=True);maximum=max(np.max(m) for m in maps.values())
        for ax,(label,beta) in zip(axes.ravel(),maps.items()):
            data=np.full(len(keep),np.nan);data[keep]=beta
            im=ax.imshow(data.reshape(resolution,resolution),origin='lower',extent=(0,1,0,1),vmin=0,vmax=maximum if shared else None,cmap='viridis')
            for w in env.wall_blocks:ax.add_patch(Rectangle((w.xmin,w.ymin),w.xmax-w.xmin,w.ymax-w.ymin,color='black'))
            ax.set_title(f'epsilon={args.epsilon}, '+label);fig.colorbar(im,ax=ax)
        fig.savefig(out/filename,dpi=180);plt.close(fig)
    report=['# Epsilon-greedy FB Monte Carlo bottlenecks','',f'{n} identical saved cross-room pairs; {k} rollouts each; epsilon={args.epsilon}; horizon {args.horizon}.',
            '', '| Snapshot | Goal success | Mean successful steps | Mean capped steps | Mean first any-door arrival | Mean pooled doorway-cell arrival |',
            '|---|---:|---:|---:|---:|---:|']
    def fmt(value):return 'N/A' if value is None else f'{value:.1f}'
    for name,s in metrics['policies'].items():
        a=s['doorway_arrivals']['any_door']
        report.append(f'| {name} | {s["success_rate"]:.2%} | {fmt(s["mean_successful_steps"])} | {fmt(s["mean_capped_steps"])} | {fmt(a["mean_first_any_cell"])} | {fmt(a["mean_pooled_cell_arrival"])} |')
    report+=['','| Configuration | Door/other enrichment | Bottom percentile | Top percentile | Left percentile | Right percentile |','|---|---:|---:|---:|---:|---:|']
    for name,s in metrics['maps'].items():
        report.append(f'| {name} | {s["doorway_enrichment"]:.2f}x | '+' | '.join(f'{s["doors"][d]["percentile"]:.1%}' for d in ('bottom','top','left','right'))+' |')
    report+=['',*metrics['caveats'],'','Run: python evaluate_epsilon_bottlenecks.py. Options: --pairs, --checkpoint, --epsilon, --horizon, --rollouts, --seed, --goal-batch, --device, --output.']
    (out/'RESULTS.md').write_text('\n'.join(report));print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',default='evaluation_bfs_control/intermediate.npz');p.add_argument('--checkpoint',default='models/fb_fourrooms_jax_goal.pkl')
    p.add_argument('--epsilon',type=float,default=.02);p.add_argument('--horizon',type=int,default=10000)
    p.add_argument('--rollouts',type=int,default=None);p.add_argument('--seed',type=int,default=2027)
    p.add_argument('--goal-batch',type=int,default=4);p.add_argument('--device',default='cpu')
    p.add_argument('--output',default='evaluation_epsilon_bottlenecks')
    run(p.parse_args())
