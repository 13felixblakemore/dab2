"""Evaluation-only temperature sweep with common per-trial/time random numbers.
No learner attributes, parameters, checkpoint files or training code are changed.
"""
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
from matplotlib.patches import Rectangle

from fb_navigation.agents.forward_backward_jax import FBLearner
from fb_navigation.environments.continuous_four_rooms_jax import ContinuousFourRooms
from fb_navigation.diagnostics.control_bfs_bottlenecks import bfs_distances
from fb_navigation.diagnostics.estimate_fb_bottlenecks_jax import grid_regions,eligibility
from fb_navigation.diagnostics.bottleneck_policies import estimate,door_statistics,path_statistics
from fb_navigation.diagnostics.bottleneck_policies import doorway_arrivals

TEMPERATURES=[200.,100.,50.,20.,10.,2.,.2]


def distribution(values):
    a=np.asarray(values).ravel()
    return dict(mean=float(a.mean()),std=float(a.std()),**{f'p{p}':float(np.percentile(a,p)) for p in (0,1,10,25,50,75,90,99,100)})


def probabilities(q,mode,temperature=200.,epsilon=.02):
    if mode=='softmax':
        scaled=(q-q.max(axis=-1,keepdims=True))/temperature
        p=np.exp(scaled);return p/p.sum(axis=-1,keepdims=True)
    greedy=np.eye(q.shape[-1],dtype=np.float32)[q.argmax(axis=-1)]
    if mode=='greedy':return greedy
    if mode=='epsilon':return (1-epsilon)*greedy+epsilon/q.shape[-1]
    raise ValueError('Unknown policy')


def tabulate_q(l,nodes,goals,batch):
    features=l.featuriser.transform(jnp.asarray(nodes,dtype=jnp.float32))
    tasks=l.embed_B(jnp.asarray(goals,dtype=jnp.float32))
    @jax.jit
    def calculate(params,zs):
        def one(z):
            f=l.forward_network.apply(params,features,jnp.broadcast_to(z,(len(nodes),l.embed_dim)))
            return jnp.einsum('nda,d->na',f,z)
        return jax.vmap(one)(zs)
    values=[]
    for i in range(0,len(goals),batch):
        values.append(np.asarray(calculate(l.state['target']['f'],tasks[i:i+batch])))
    return np.concatenate(values)


def simulate(graph,terminal,starts,goal_rows,node_candidate,policy,k,horizon,seed):
    # Always draw for ALL trial IDs, including completed ones. Thus trial i at
    # time t receives the same uniform variate under every policy/checkpoint.
    rng=np.random.default_rng(seed)
    current=np.repeat(starts,k);g=np.repeat(goal_rows,k);n=len(current)
    first=np.full((n,int(node_candidate.max())+1),-1,np.int32)
    goal_hit=np.full(n,horizon+1,int);active=~terminal[g,current];goal_hit[~active]=0
    blocked=np.zeros(n,int)
    for t in range(horizon):
        idx=np.flatnonzero(active)
        if not len(idx):break
        draws=rng.random(n)
        c=node_candidate[current[idx]];ok=c>=0;rows=idx[ok];col=c[ok]
        unseen=first[rows,col]<0;first[rows[unseen],col[unseen]]=t
        p=policy[g[idx],current[idx]].astype(np.float64);p/=p.sum(axis=1,keepdims=True)
        action=np.argmax(np.cumsum(p,axis=1)>draws[idx,None],axis=1)
        nxt=graph[current[idx],action];wall=nxt<0;blocked[idx]+=wall
        current[idx]=np.where(wall,current[idx],nxt)
        hit=terminal[g[idx],current[idx]];goal_hit[idx[hit]]=t+1;active[idx[hit]]=False
    return first,goal_hit,blocked


def audit(l,nodes,goals,graph):
    env=ContinuousFourRooms(include_stay_action=False)
    np.testing.assert_array_equal(env.get_available_actions(None),np.arange(4))
    training_directions=np.asarray([(a.x,a.y) for a in env.actions])
    training_directions/=np.linalg.norm(training_directions,axis=1,keepdims=True)
    points=jnp.asarray(nodes[[0,len(nodes)//3,-1]],jnp.float32)
    z=l.embed_B(jnp.asarray(goals[:1],jnp.float32));w=jnp.repeat(z,len(points),axis=0)
    f=l._embed_f(l.state['target']['f'],points,w)
    q=np.einsum('nda,nd->na',np.asarray(f),np.asarray(w))
    np.testing.assert_array_equal(l.act(points,w,target_network=True),q.argmax(axis=-1))
    rbf=l.featuriser
    if hasattr(rbf,'x_mu'):
        np.testing.assert_allclose(np.asarray(rbf.transform(points)),rbf.transform(np.asarray(points)),rtol=2e-5,atol=1e-6)
    expected=np.array([[-1,0],[1,0],[0,1],[0,-1]])
    np.testing.assert_array_equal(training_directions,expected)
    for a,delta in enumerate(expected):
        valid=graph[:,a]>=0
        movement=nodes[graph[valid,a]]-nodes[valid]
        step=np.linalg.norm(movement,axis=1)
        np.testing.assert_allclose(movement/step[:,None],np.broadcast_to(delta,movement.shape),atol=1e-6)
    return dict(actions=['LEFT','RIGHT','UP','DOWN'],mask='all four available; blocked actions are self-transitions',
                feature_preprocessing='existing learner RBF, CPU/JAX equivalence checked',
                actor='target F and unnormalised online B(goal); greedy checked against learner.act',
                task_normalization='existing ForwardNet internal scaling only; no extra normalization',
                transitions='identical saved graph across temperatures/models; differs from training continuous step/noise')


def q_diagnostics(q,distances,graph):
    sorted_q=np.sort(q,axis=-1)
    active=distances>0
    greedy=q.argmax(axis=-1)
    dest=graph[np.arange(len(graph))[None,:],greedy]
    nxt=np.where(dest>=0,dest,np.arange(len(graph))[None,:])
    next_distance=np.take_along_axis(distances,nxt,axis=1)
    return dict(raw_q=distribution(q),centered_q=distribution(q-q.mean(axis=-1,keepdims=True)),
                action_span=distribution(q.max(axis=-1)-q.min(axis=-1)),
                best_second_gap=distribution(sorted_q[:,:,-1]-sorted_q[:,:,-2]),
                greedy_wall_fraction=float(np.mean((dest<0)[active])),
                greedy_bfs_descent_fraction=float(np.mean((next_distance==distances-1)[active])))


def run(args):
    if min(args.rollouts,args.horizon,args.goal_batch)<1:raise ValueError('Counts must be positive')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    with np.load(args.pairs) as d:
        nodes=d['nodes'];graph=d['neighbours'];starts=d['starts'];goals=d['goals'];points=d['candidate_points']
        keep=d['candidate_mask'];eligible=d['eligible'];shortest=d['shortest_lengths']
    n=len(starts);unique,inverse=np.unique(goals,return_inverse=True)
    env=ContinuousFourRooms(include_stay_action=False);resolution=int(np.sqrt(len(keep)))
    xy=np.clip(np.floor(nodes*resolution).astype(int),0,resolution-1)
    index=np.full(len(keep),-1,int);index[np.flatnonzero(keep)]=np.arange(keep.sum())
    node_candidate=index[xy[:,1]*resolution+xy[:,0]]
    conditions=[(f'T={t:g}','softmax',t) for t in TEMPERATURES]+[('greedy','greedy',0.),('epsilon=.02','epsilon',0.)]
    summaries=[];diagnostics={};raw={};maps={};audits={};provenance={}
    for model,path in [('goal_terminated',args.checkpoint),('continuing',args.continuing_checkpoint)]:
        if not Path(path).exists():
            if model=='goal_terminated':raise FileNotFoundError(path)
            continue
        print('Loading '+model,flush=True)
        digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        l=FBLearner.from_checkpoint(path,device=args.device,buffer_device='cpu')
        terminal=np.sum((nodes[None]-nodes[unique,None])**2,axis=-1)<=l.goal_threshold**2
        distances=np.stack([bfs_distances(graph,t) for t in terminal])
        np.testing.assert_array_equal(distances[inverse,starts],shortest)
        _,lo,hi,mask=grid_regions(env,resolution)
        np.testing.assert_array_equal(mask,keep)
        np.testing.assert_array_equal(eligible,eligibility(nodes[goals],lo[keep],hi[keep],l.goal_threshold))
        with jax.default_device(l.device):
            audits[model]=audit(l,nodes,nodes[unique],graph)
            q=tabulate_q(l,nodes,nodes[unique],args.goal_batch)
        diagnostics[model]=q_diagnostics(q,distances,graph)
        provenance[model]=dict(path=path,sha256=digest,config=l.config)
        np.savez_compressed(out/(model+'_q.npz'),q=q,goal_nodes=unique,nodes=nodes)
        raw[model+'_q_sample']=q.ravel()[::max(1,q.size//20000)]
        # Full-grid Q statistics and initial-state checks separate bad action
        # rankings from a softmax temperature effect without changing the learner.
        state_q=q[inverse,starts]
        diagnostics[model]['initial_pair_action_gap']=distribution(np.sort(state_q,axis=1)[:,-1]-np.sort(state_q,axis=1)[:,-2])
        for label,mode,t in conditions:
            print(model+' '+label,flush=True)
            p=probabilities(q,mode,t)
            entropy=-np.sum(p*np.log(np.maximum(p,1e-30)),axis=-1)
            first,hit,wall=simulate(graph,terminal,starts,inverse,node_candidate,p,args.rollouts,args.horizon,args.seed)
            path_stats=path_statistics(hit,args.horizon,shortest,args.rollouts)
            arrivals=doorway_arrivals(first,eligible,points,args.rollouts)['any_door']
            beta,_=estimate(first,eligible,args.rollouts,.99);door=door_statistics(points,beta)
            row=dict(model=model,policy=label,success_rate=path_stats['success_rate'],
                mean_successful_steps=path_stats['mean_successful_steps'],median_successful_steps=path_stats['median_successful_steps'],
                mean_capped_steps=path_stats['mean_capped_steps'],wall_action_percent=float(100*wall.sum()/np.minimum(hit,args.horizon).sum()),
                first_door_mean=arrivals['mean_first_any_cell'],door_arrival_rate=arrivals['arrival_rate'],
                pooled_door_arrival_mean=arrivals['mean_pooled_cell_arrival'],door_enrichment=door['doorway_enrichment'],
                entropy_mean=float(entropy.mean()),entropy_normalized=float(entropy.mean()/np.log(4)),
                approximately_uniform_fraction=float(np.mean(entropy/np.log(4)>.99)),
                mean_max_action_probability=float(p.max(axis=-1).mean()))
            summaries.append(row);maps[model+'_'+label]=beta
            raw[model+'_'+label+'_goal_hits']=hit
            raw[model+'_'+label+'_pair_success']=(hit<=args.horizon).reshape(n,args.rollouts).mean(axis=1)
            # Each condition is written immediately, so the evaluation is reviewable
            # even before the entire sweep finishes.
            (out/'progress.json').write_text(json.dumps(summaries,indent=2,allow_nan=False))
            print(json.dumps(row),flush=True)
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise RuntimeError('Checkpoint changed during evaluation')
    metadata=dict(samples=n,rollouts_per_pair=args.rollouts,horizon=args.horizon,seed=args.seed,
        pair_source=args.pairs,pair_sha256=hashlib.sha256(Path(args.pairs).read_bytes()).hexdigest(),
        provenance=provenance,audits=audits,q_diagnostics=diagnostics,results=summaries,
        notes=['All policies use target F and online B(goal); snapshot selection is fixed.',
               'Uniform random numbers are identical for each trial/time across policies, including after other trials finish.',
               'First doorway arrival and successful path lengths are conditional on arrival/success; mean capped length includes failures.',
               'Actions rejected by walls remain valid actions. The learner and trained target policy are not altered.',
               'Changing evaluation T changes the executed policy, not the policy whose successor measure was trained.',
               'Primary control uses the fixed discretised deterministic graph. Continuous-environment transfer limits apply.',
               'Q/entropy statistics weight sampled unique goals and free grid states equally, not by on-policy occupancy.',
               'The older and newer checkpoints also differ in seed, task sampling and stopping objective; this is not a causal training ablation.'])
    (out/'metrics.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    np.savez_compressed(out/'intermediate.npz',starts=starts,goals=goals,points=points,candidate_mask=keep,**raw,**maps)
    plot_results(out,summaries,diagnostics,maps,keep,env)
    report=['# Evaluation-only softmax temperature sweep','',f'{n} fixed cross-room pairs, {args.rollouts} rollouts/pair, horizon {args.horizon}.',
            '', '| Checkpoint | Policy | Success | Mean success steps | Median success steps | Wall actions | First doorway mean | Door arrival | Door emphasis (.99) |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x):return 'N/A' if x is None else f'{x:.1f}'
    for r in summaries:
        report.append(f'| {r["model"]} | {r["policy"]} | {r["success_rate"]:.1%} | {fmt(r["mean_successful_steps"])} | {fmt(r["median_successful_steps"])} | {r["wall_action_percent"]:.1f}% | {fmt(r["first_door_mean"])} | {r["door_arrival_rate"]:.1%} | {r["door_enrichment"]:.2f}x |')
    report+=['',*metadata['notes'],'', 'Run: python sweep_fb_temperature.py. Options: --checkpoint, --continuing-checkpoint, --pairs, --rollouts, --horizon, --goal-batch, --seed, --device, --output. No retraining is performed.']
    (out/'RESULTS.md').write_text('\n'.join(report));return metadata


def plot_results(out,rows,diagnostics,maps,keep,env):
    labels=list(dict.fromkeys(r['policy'] for r in rows));models=list(diagnostics)
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    panels=[('success_rate','Goal success'),('mean_successful_steps','Mean successful steps'),('wall_action_percent','Wall actions (%)'),
            ('first_door_mean','Mean first doorway time (arrivals only)'),('door_enrichment','Door / other emphasis, gamma=.99'),('entropy_normalized','Policy entropy / log(4)')]
    for ax,(key,title) in zip(axes.ravel(),panels):
        for model in models:
            subset=[r for r in rows if r['model']==model]
            ax.plot(range(len(subset)),[r[key] if r[key] is not None else np.nan for r in subset],marker='o',label=model)
        ax.set_xticks(range(len(labels)),labels,rotation=55,ha='right');ax.set_title(title);ax.grid(alpha=.2)
    axes[0,0].legend();fig.savefig(out/'temperature_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(len(models),2,figsize=(11,4*len(models)),squeeze=False,constrained_layout=True)
    for i,model in enumerate(models):
        with np.load(out/(model+'_q.npz')) as d:q=d['q']
        gap=np.sort(q,axis=-1)[:,:,-1]-np.sort(q,axis=-1)[:,:,-2]
        span=q.max(axis=-1)-q.min(axis=-1)
        for ax,values,title in zip(axes[i],(gap,span),('best-second gap','max-min action span')):
            ax.hist(values.ravel()[::max(1,values.size//100000)],bins=80);ax.set_title(model+': '+title);ax.set_ylabel('Count')
    fig.savefig(out/'q_gap_distributions.png',dpi=160);plt.close(fig)
    for model in models:
        subset=[(label,beta) for label,beta in maps.items() if label.startswith(model+'_')]
        fig,axes=plt.subplots(3,3,figsize=(13,12),constrained_layout=True)
        maximum=max(float(beta.max()) for _,beta in subset);res=int(np.sqrt(len(keep)))
        for ax,(label,beta) in zip(axes.ravel(),subset):
            data=np.full(len(keep),np.nan);data[keep]=beta
            im=ax.imshow(data.reshape(res,res),origin='lower',extent=(0,1,0,1),vmin=0,vmax=maximum,cmap='viridis')
            for w in env.wall_blocks:ax.add_patch(Rectangle((w.xmin,w.ymin),w.xmax-w.xmin,w.ymax-w.ymin,color='black'))
            ax.set_title(label.removeprefix(model+'_'));fig.colorbar(im,ax=ax)
        fig.savefig(out/(model+'_betweenness.png'),dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default='models/fb_fourrooms_jax_goal.pkl');p.add_argument('--continuing-checkpoint',default='models/fb_fourrooms_jax.pkl')
    p.add_argument('--pairs',default='evaluation_bfs_control/intermediate.npz');p.add_argument('--rollouts',type=int,default=4)
    p.add_argument('--horizon',type=int,default=10000);p.add_argument('--seed',type=int,default=2028)
    p.add_argument('--goal-batch',type=int,default=4);p.add_argument('--device',default='cpu');p.add_argument('--output',default='evaluation_temperature_sweep')
    run(p.parse_args())
