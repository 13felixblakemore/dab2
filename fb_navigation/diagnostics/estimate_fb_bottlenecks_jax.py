"""Vectorised stopped-FB bottleneck proxy with matched first-hit Monte Carlo.

See FB_BOTTLENECK_EVALUATION.md. No training or terminal correction is performed.
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
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Rectangle

from fb_navigation.environments.continuous_four_rooms_jax import ContinuousFourRooms, Point, _move
from fb_navigation.agents.forward_backward_jax import FBLearner, mask_action_scores


def grid_regions(env,resolution):
    edges=np.linspace(0,env.size,resolution+1)
    x,y=np.meshgrid(edges[:-1],edges[:-1])
    lower=np.column_stack((x.ravel(),y.ravel()))
    upper=lower+env.size/resolution
    keep=np.ones(len(lower),bool)
    for wall in env.wall_blocks:
        # Positive-area overlap, rather than testing only the cell center.
        keep &= ~((lower[:,0]<wall.xmax)&(upper[:,0]>wall.xmin)&
                  (lower[:,1]<wall.ymax)&(upper[:,1]>wall.ymin))
    return ((lower+upper)/2).astype(np.float32),lower,upper,keep


def free_points(rng,count,env):
    result=[];left=count
    while left:
        points=rng.uniform(0,env.size,(max(1024,left*2),2)).astype(np.float32)
        valid=np.ones(len(points),bool)
        for w in env.wall_blocks:
            valid &= ~((points[:,0]>=w.xmin)&(points[:,0]<=w.xmax)&(points[:,1]>=w.ymin)&(points[:,1]<=w.ymax))
        take=points[valid][:left];result.append(take);left-=len(take)
    return np.concatenate(result)


def sample_pairs(env,count,threshold,seed):
    """Joint uniform free-space pairs, conditioned on starting outside G."""
    rng=np.random.default_rng(seed);starts=[];goals=[];left=count
    while left:
        n=max(64,left*2)
        u=free_points(rng,n,env);g=free_points(rng,n,env)
        valid=np.sum((u-g)**2,axis=1)>threshold**2
        take_u=u[valid][:left];take_g=g[valid][:left]
        starts.append(take_u);goals.append(take_g);left-=len(take_u)
    return np.concatenate(starts),np.concatenate(goals)


def eligibility(goals,lower,upper,threshold):
    nearest=np.minimum(np.maximum(goals[:,None,:],lower[None]),upper[None])
    return np.sum((goals[:,None,:]-nearest)**2,axis=-1)>threshold**2


def make_fb_estimator(learner,candidates,backward):
    """Vmap across goals, with candidate F evaluations in bounded pair batches."""
    features=learner.featuriser.transform(jnp.asarray(candidates))
    mask=jnp.ones(learner.n_actions,bool)  # Four Rooms exposes all actions.
    def average(params,target,features,z):
        w=jnp.broadcast_to(z,(len(features),learner.embed_dim))
        f=learner.forward_network.apply(params,features,w)
        pf=learner.forward_network.apply(target,features,w)
        logits=mask_action_scores(jnp.einsum('nda,d->na',pf,z)/learner.temperature,mask)
        pi=jax.nn.softmax(logits,axis=-1)
        return jnp.einsum('nda,na->nd',f,pi)
    @jax.jit
    def estimate(params,target,start_features,tasks):
        def one(start,z):
            fu=average(params,target,start[None],z)[0]
            fc=average(params,target,features,z)
            return backward@fu,jnp.sum(fc*backward,axis=1)
        return jax.vmap(one)(start_features,tasks)
    return estimate


def make_mc_estimator(learner,env,resolution,horizon):
    actions=jnp.asarray([(a.x,a.y) for a in env.actions])
    mask=jnp.zeros(learner.n_actions,bool).at[jnp.asarray(env.get_available_actions(None))].set(True)
    walls=tuple(env.wall_blocks)
    @jax.jit
    def estimate(target,starts,goals,tasks,keys):
        n=len(starts)
        first=jnp.full((n,resolution**2),horizon+1,jnp.int32)
        active=jnp.ones(n,bool)
        goal_hit=jnp.full(n,horizon+1,jnp.int32)
        def step(carry,t):
            positions,first,active,goal_hit,keys=carry
            at_goal=jnp.sum((positions-goals)**2,axis=1)<=learner.goal_threshold**2
            active=active & ~at_goal
            xy=jnp.clip(jnp.floor(positions/env.size*resolution).astype(jnp.int32),0,resolution-1)
            ids=xy[:,1]*resolution+xy[:,0]
            old=first[jnp.arange(n),ids]
            first=first.at[jnp.arange(n),ids].set(jnp.where(active,jnp.minimum(old,t),old))
            split=jax.vmap(lambda k:jax.random.split(k,3))(keys)
            keys,ak,mk=split[:,0],split[:,1],split[:,2]
            f=learner.forward_network.apply(target,learner.featuriser.transform(positions),tasks)
            logits=mask_action_scores(jnp.einsum('nda,nd->na',f,tasks)/learner.temperature,mask)
            selected=jax.vmap(lambda key,logit:jax.random.categorical(key,logit))(ak,logits)
            def move(p,a,key):
                delta=actions[a]
                nxt,_,_=_move(Point(*p),Point(*delta),key,env._noise,env.size,walls)
                return jnp.array((nxt.x,nxt.y))
            proposed=jax.vmap(move)(positions,selected,mk)
            positions=jnp.where(active[:,None],proposed,positions)
            reached=jnp.sum((positions-goals)**2,axis=1)<=learner.goal_threshold**2
            goal_hit=jnp.where(active & reached,t+1,goal_hit)
            active=active & ~reached
            return (positions,first,active,goal_hit,keys),None
        (_,first,active,goal_hit,_),_=jax.lax.scan(step,(starts,first,active,goal_hit,keys),jnp.arange(horizon))
        return first,goal_hit
    return estimate


def aggregate(numerator,denominator,eligible,mc,tolerance=1e-6):
    positive=np.isfinite(denominator)&(denominator>tolerance)
    with np.errstate(divide='ignore',invalid='ignore'):
        raw=numerator/denominator  # never clamp or substitute a denominator
    valid=eligible & positive & np.isfinite(numerator)&np.isfinite(raw)
    ratios=np.where(valid,raw,np.nan)
    counts=eligible.sum(0);usable=valid.sum(0)
    def mean(a,m):return np.divide(np.where(m,a,0.).sum(0),m.sum(0),out=np.full(a.shape[1],np.nan),where=m.sum(0)>0)
    fb=mean(raw,valid);all_mc=mean(mc,eligible);matched_mc=mean(mc,valid)
    coverage=np.divide(usable,counts,out=np.zeros_like(counts,dtype=float),where=counts>0)
    strict=fb.copy();strict[(usable!=counts)|(counts==0)]=np.nan
    se=np.sqrt(mean((mc-all_mc)**2,eligible))/np.sqrt(np.maximum(counts-1,1))
    return dict(ratios=ratios,valid=valid,fb_conditional=fb,fb_strict=strict,mc=all_mc,
                mc_matched=matched_mc,validity=coverage,eligible_counts=counts,valid_counts=usable,mc_se=se)


def comparison(a,b):
    valid=np.isfinite(a)&np.isfinite(b);x=a[valid];y=b[valid]
    corr=float(np.corrcoef(x,y)[0,1]) if len(x)>1 and np.std(x)>0 and np.std(y)>0 else None
    k=min(20,len(x));overlap=None
    if k:overlap=float(len(set(np.argsort(x)[-k:])&set(np.argsort(y)[-k:]))/k)
    return dict(cells=int(valid.sum()),correlation=corr,mae=float(np.mean(abs(x-y))) if len(x) else None,top20_overlap=overlap)


def region_summary(points,maps,size):
    p=points/size*11;x,y=p.T
    doors=dict(bottom=(x>=6)&(x<=7)&(y>=1)&(y<=2),top=(x>=6)&(x<=7)&(y>=8)&(y<=9),
               left=(x>=2)&(x<=3)&(y>=4)&(y<=5),right=(x>=9)&(x<=10)&(y>=4)&(y<=5))
    summary={}
    for name,selected in doors.items():
        row=dict(cells=int(selected.sum()))
        for label,values in maps.items():
            all_values=values[np.isfinite(values)]
            region_values=values[selected & np.isfinite(values)]
            row[label]=float(region_values.mean()) if len(region_values) else None
            row[label+'_mean_percentile']=float(np.mean(all_values<=region_values.mean())) if len(region_values) else None
        summary[name]=row
    return summary


def plot_maps(out,env,resolution,keep,maps):
    panels=[('fb_strict','FB: all eligible pairs valid'),('mc','MC: all eligible pairs'),
            ('validity','FB validity coverage'),('fb_conditional','FB: valid-pair mean (conditional)'),
            ('mc_matched','MC: same valid-pair subset'),('difference','Absolute difference on matched subset')]
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#bdbdbd')
    scores=np.concatenate((maps['fb_conditional'],maps['mc']))
    finite=scores[np.isfinite(scores)]
    maximum=max(.01,float(np.max(abs(finite))) if len(finite) else .01)
    norm=SymLogNorm(linthresh=.02,vmin=-maximum,vmax=maximum)
    def draw(ax,values,title,kind):
        data=np.full(resolution**2,np.nan);data[keep]=values
        kwargs=dict(norm=norm,cmap=cmap) if kind=='score' else dict(vmin=0,vmax=1 if kind=='coverage' else None,cmap='viridis' if kind=='coverage' else 'magma')
        im=ax.imshow(np.ma.masked_invalid(data.reshape(resolution,resolution)),origin='lower',extent=(0,env.size,0,env.size),**kwargs)
        for w in env.wall_blocks:ax.add_patch(Rectangle((w.xmin,w.ymin),w.xmax-w.xmin,w.ymax-w.ymin,color='black'))
        ax.set_title(title);ax.set_aspect('equal');fig.colorbar(im,ax=ax)
    for ax,(name,title) in zip(axes.ravel(),panels):
        kind='coverage' if name=='validity' else 'error' if name=='difference' else 'score'
        draw(ax,maps[name],title,kind)
    fig.suptitle('Goal-terminated bottleneck proxy: gray = missing, black = walls')
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5),constrained_layout=True)
    data=np.full(resolution**2,np.nan);data[keep]=maps['mc']
    im=ax.imshow(data.reshape(resolution,resolution),origin='lower',extent=(0,env.size,0,env.size),vmin=0,cmap='viridis')
    for w in env.wall_blocks:ax.add_patch(Rectangle((w.xmin,w.ymin),w.xmax-w.xmin,w.ymax-w.ymin,color='black'))
    ax.set_title('MC discounted first entry before goal');fig.colorbar(im,ax=ax)
    fig.savefig(out/'mc_heatmap.png',dpi=160);plt.close(fig)


def evaluate(args):
    if min(args.samples,args.grid,args.horizon,args.fb_batch,args.mc_batch)<1:raise ValueError('Counts must be positive')
    if not np.isfinite(args.denominator_tolerance) or args.denominator_tolerance<0:raise ValueError('Invalid denominator tolerance')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    learner=FBLearner.from_checkpoint(args.checkpoint,device=args.device,buffer_device='cpu')
    if not learner.goal_terminated or learner.z_sampling!='replay':raise ValueError('A goal-terminated replay-task checkpoint is required')
    if learner.n_actions not in (4,5):raise ValueError('Expected Four Rooms action space')
    with jax.default_device(learner.device):
        env=ContinuousFourRooms(include_stay_action=learner.n_actions==5)
        all_points,lower,upper,keep=grid_regions(env,args.grid)
        points=all_points[keep];ids=np.flatnonzero(keep)
        if not len(points):raise ValueError('Grid has no wall-free whole cells; increase resolution')
        starts,goals=sample_pairs(env,args.samples,learner.goal_threshold,args.seed)
        eligible=eligibility(goals,lower[keep],upper[keep],learner.goal_threshold)
        tasks=learner.embed_B(goals);backward=learner.embed_B(points)
        start_features=learner.featuriser.transform(jnp.asarray(starts))
        estimate_fb=make_fb_estimator(learner,points,backward)
        nums=[];dens=[]
        for first in range(0,args.samples,args.fb_batch):
            n,d=estimate_fb(learner.state['params']['f'],learner.state['target']['f'],start_features[first:first+args.fb_batch],tasks[first:first+args.fb_batch])
            nums.append(np.asarray(n));dens.append(np.asarray(d))
            if first%max(args.fb_batch,(args.samples//5//args.fb_batch)*args.fb_batch)==0:print(f'FB pairs {min(first+args.fb_batch,args.samples)}/{args.samples}',flush=True)
        numerator=np.concatenate(nums);denominator=np.concatenate(dens)
        estimate_mc=make_mc_estimator(learner,env,args.grid,args.horizon)
        keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(args.seed+1),i))(jnp.arange(args.samples))
        hits=[];goal_hits=[]
        for first in range(0,args.samples,args.mc_batch):
            end=first+args.mc_batch
            h,g=estimate_mc(learner.state['target']['f'],jnp.asarray(starts[first:end]),jnp.asarray(goals[first:end]),tasks[first:end],keys[first:end])
            hits.append(np.asarray(h)[:,ids]);goal_hits.append(np.asarray(g))
            print(f'MC pairs {min(end,args.samples)}/{args.samples}',flush=True)
        first_hits=np.concatenate(hits);goal_hits=np.concatenate(goal_hits)
        mc=np.where(first_hits<args.horizon,learner.discount_factor**first_hits,0.)
        maps=aggregate(numerator,denominator,eligible,mc,args.denominator_tolerance)
        maps['difference']=abs(maps['fb_conditional']-maps['mc_matched'])
        fraction=lambda condition:float(np.sum(condition & eligible)/np.maximum(eligible.sum(),1))
        metrics=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            updates=int(learner.state['optim'][0]),samples=args.samples,grid=args.grid,free_candidate_cells=len(points),
            seed=args.seed,horizon=args.horizon,gamma=learner.discount_factor,goal_threshold=learner.goal_threshold,
            temperature=learner.temperature,policy='frozen target masked softmax',prediction='online F/B, action averaged under the same target policy',
            denominator_tolerance=args.denominator_tolerance,eligible_pair_fraction=float(eligible.mean()),
            valid_eligible_pair_fraction=float(maps['valid'].sum()/max(eligible.sum(),1)),
            strict_cell_coverage=float(np.isfinite(maps['fb_strict']).mean()),any_valid_cell_coverage=float(np.isfinite(maps['fb_conditional']).mean()),
            nonpositive_denominator_fraction=fraction(denominator<=0),small_positive_denominator_fraction=fraction((denominator>0)&(denominator<=args.denominator_tolerance)),
            nonfinite_denominator_fraction=fraction(~np.isfinite(denominator)),
            valid_ratios_outside_probability_range_fraction=float(np.mean((maps['ratios'][maps['valid']]<0)|(maps['ratios'][maps['valid']]>1))) if maps['valid'].any() else None,
            strict_comparison=comparison(maps['fb_strict'],maps['mc']),matched_conditional_comparison=comparison(maps['fb_conditional'],maps['mc_matched']),
            conditional_fb_vs_full_mc=comparison(maps['fb_conditional'],maps['mc']),goal_reached_fraction=float(np.mean(goal_hits<=args.horizon)),
            discounted_horizon_bound=learner.discount_factor**args.horizon,
            doorway_cells=region_summary(points,dict(fb_strict=maps['fb_strict'],fb_conditional=maps['fb_conditional'],mc=maps['mc'],mc_matched=maps['mc_matched']),env.size),
            caveats=['Density ratio is a finite-region hitting-probability proxy, not an exact cell identity.',
                     'Starts/goals are jointly uniform in free space conditioned on starts outside the goal disk.',
                     'Per-candidate means condition on the whole candidate cell being disjoint from the goal disk.',
                     'Valid-pair conditional maps omit invalid ratios; matched MC uses exactly the same pair subset.',
                     'The unchanged immediate loss has no exact zero boundary for starts inside the goal. Such pairs and overlapping candidate diagonals are excluded.',
                     'The learned policy is not proven optimal. One rollout per independently sampled pair estimates endpoint and policy randomness jointly.',
                     'Grid resolution, MC noise, near-zero diagonals, and online/target lag may affect rankings. No density/ratio clipping is used.'])
        valid_ratios=maps['ratios'][maps['valid']]
        metrics.update(
            negative_ratio_fraction=float(np.mean(valid_ratios<0)) if len(valid_ratios) else None,
            above_one_ratio_fraction=float(np.mean(valid_ratios>1)) if len(valid_ratios) else None,
            mean_fb_score=float(np.nanmean(maps['fb_conditional'])) if np.isfinite(maps['fb_conditional']).any() else None,
            mean_mc_score=float(np.nanmean(maps['mc'])) if np.isfinite(maps['mc']).any() else None,
            mean_mc_standard_error=float(np.nanmean(maps['mc_se'])) if np.isfinite(maps['mc_se']).any() else None)
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
        np.savez_compressed(out/'intermediate.npz',starts=starts,goals=goals,tasks=np.asarray(tasks),candidate_points=points,
            candidate_lower=lower[keep],candidate_upper=upper[keep],candidate_mask=keep,grid_ids=ids,
            numerator=numerator,denominator=denominator,eligible=eligible,first_hits=first_hits,goal_hits=goal_hits,
            mc_pair_scores=mc,**maps)
        with (out/'cells.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['x','y','fb_strict','fb_conditional','mc','mc_matched','validity','eligible_pairs','valid_pairs','mc_standard_error'])
            writer.writerows(zip(points[:,0],points[:,1],maps['fb_strict'],maps['fb_conditional'],maps['mc'],maps['mc_matched'],maps['validity'],maps['eligible_counts'],maps['valid_counts'],maps['mc_se']))
        plot_maps(out,env,args.grid,keep,maps)
        print(json.dumps(metrics,indent=2),flush=True)
        report=['# Goal-terminated FB bottleneck evaluation','',f'Checkpoint: {args.checkpoint}; {args.samples} pairs; {args.grid}x{args.grid} grid; horizon {args.horizon}.',
                '',f'Valid eligible pair fraction: {metrics["valid_eligible_pair_fraction"]:.2%}. Strict cell coverage: {metrics["strict_cell_coverage"]:.2%}.',
                '',f'Strict comparison: {metrics["strict_comparison"]}',f'Matched valid-subset comparison: {metrics["matched_conditional_comparison"]}',
                '', 'Invalid ratios were not clamped. Partial FB averages condition on valid pairs and are explicitly paired with the same MC subset. These diagnostics cannot repair invalid density estimates.',
                '', 'Doorway cell means:', '', '| Door | Cells | FB conditional | MC all pairs | MC matched |','|---|---:|---:|---:|---:|']
        for name,row in metrics['doorway_cells'].items():report.append(f'| {name} | {row["cells"]} | {row["fb_conditional"]} | {row["mc"]} | {row["mc_matched"]} |')
        report+=['', 'Doorway ranking diagnostics (percentile of the doorway mean among all free-cell scores):']
        for name,row in metrics['doorway_cells'].items():
            report.append(f'{name}: MC percentile={row["mc_mean_percentile"]}; FB conditional percentile={row["fb_conditional_mean_percentile"]}.')
        report += ['', f'Mean FB score: {metrics["mean_fb_score"]}; mean MC score: {metrics["mean_mc_score"]}; mean MC standard error: {metrics["mean_mc_standard_error"]}.',
                   f'Negative individual ratios: {metrics["negative_ratio_fraction"]}; above-one ratios: {metrics["above_one_ratio_fraction"]}.',
                   'High global correlation measures spatial agreement, not calibrated hitting probabilities or doorway discovery. Check the doorway percentiles and top-cell maps: broad room-interior peaks are not doorway bottlenecks. Finite discount and a nonoptimal softmax policy can suppress long-range doorway traffic.']
        report+=['', 'This is a finite-discount, finite-region proxy. See FB_BOTTLENECK_EVALUATION.md for definitions and limitations. Full raw matrices and sampled pairs are in intermediate.npz. No training was performed.']
        (out/'RESULTS.md').write_text('\n'.join(report))
        return metrics


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default='models/fb_fourrooms_jax_goal.pkl');p.add_argument('--device',default='cpu')
    p.add_argument('--samples',type=int,default=1000);p.add_argument('--grid',type=int,default=20)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--horizon',type=int,default=1000)
    p.add_argument('--fb-batch',type=int,default=8);p.add_argument('--mc-batch',type=int,default=128)
    p.add_argument('--denominator-tolerance',type=float,default=1e-6)
    p.add_argument('--output',default='evaluation_fb_bottlenecks')
    return p


if __name__=='__main__':evaluate(parser().parse_args())
