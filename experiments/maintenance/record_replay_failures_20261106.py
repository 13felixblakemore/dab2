"""Record replay-continuing diagonal failures and quantify stationary wall loops."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import argparse
import csv
import hashlib
import json
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation,PillowWriter
from adjustable_four_rooms import AdjustableFourRooms
from fourrooms_reference_diagnostics import coordinate_room
from investigate_doorway_decline import trace_function
from forward_backward_jax import FBLearner

ACTIONS=('LEFT','RIGHT','UP','DOWN')

def wall_metrics(position,blocked,actions):
    """Exact location repeats plus a .02-radius primary collision neighborhood."""
    longest=longest_same=run=same=repeat=0
    for t,hit in enumerate(blocked):
        previous=t>0 and blocked[t-1]
        identical=previous and actions[t]==actions[t-1]
        run=run+1 if hit else 0
        same=same+1 if hit and identical else 1 if hit else 0
        repeat+=int(hit and identical);longest=max(longest,run);longest_same=max(longest_same,same)
    points=position[blocked]
    if len(points):
        unique,counts=np.unique(points,axis=0,return_counts=True);anchor=unique[counts.argmax()]
        dominant_exact=float(counts.max()/len(points))
        dominant_near=float(np.mean(np.linalg.norm(points-anchor,axis=1)<=.02))
    else:
        anchor=np.array([np.nan,np.nan]);dominant_exact=dominant_near=0.
    return dict(collisions=int(blocked.sum()),longest_stationary_collision_run=longest,
        longest_same_action_collision_run=longest_same,repeated_same_action_collisions=repeat,
        dominant_exact_fraction=dominant_exact,dominant_neighborhood_fraction=dominant_near,
        anchor=anchor.tolist())

def draw_world(ax,env):
    for w in env.wall_blocks:
        ax.add_patch(Rectangle((w.xmin,w.ymin),w.xmax-w.xmin,w.ymax-w.ymin,facecolor='#343b43',zorder=3))
    ax.add_patch(Rectangle((0,0),1,1,fill=False,color='#343b43',lw=1.2,zorder=4))
    ax.set(xlim=(-.02,1.02),ylim=(-.02,1.02),aspect='equal',xlabel='x',ylabel='y')
    ax.set_xticks([0,.25,.5,.75,1]);ax.set_yticks([0,.25,.5,.75,1]);ax.grid(alpha=.12)

def draw_path(ax,env,record,i,starts,goals,limit=150):
    draw_world(ax,env);steps=min(int(record['hits'][i]),limit)
    path=np.vstack([starts[i],record['following'][:steps,i]])
    segments=np.stack([path[:-1],path[1:]],axis=1)
    line=LineCollection(segments,cmap='viridis',norm=plt.Normalize(0,limit),linewidth=2,zorder=5)
    line.set_array(np.arange(steps));ax.add_collection(line)
    ax.scatter(*starts[i],s=35,color='#21a65b',edgecolors='white',zorder=8,label='Start')
    ax.scatter(*goals[i],s=110,marker='*',color='#ed9e18',edgecolors='black',lw=.5,zorder=9,label='Goal')
    tolerance=env.threshold_distance
    ax.add_patch(Rectangle(goals[i]-tolerance,2*tolerance,2*tolerance,fill=False,linestyle=':',edgecolor='#bd7d12',lw=1,zorder=4))
    points=record['position'][:steps,i][record['blocked'][:steps,i]]
    if len(points):
        unique,counts=np.unique(points,axis=0,return_counts=True)
        ax.scatter(*unique.T,s=10+np.minimum(counts,100)*.7,marker='x',color='#dd3535',lw=1,zorder=7,label='Rejected move')
    return path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'experiments/fourrooms_doorways/runs/width2_replay_continuing_seed0')
    parser.add_argument('--evaluation',type=Path,default=ROOT/'experiments/fourrooms_doorways/analyses/replay_continuing_comparison')
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/fourrooms_doorways/analyses/replay_continuing_failures')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--example-trials',type=int,nargs='+',
        help='Render additional failed diagonal-subset trial indices without replacing the original example')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    pairs=dict(np.load(args.evaluation/'pairs.npz'));settings=json.loads((args.evaluation/'config.json').read_text())
    config=json.loads((args.run/'config.json').read_text());env=AdjustableFourRooms(**config['environment'])
    if config['learner']['z_sampling']!='replay' or config['learner']['goal_terminated']:
        raise ValueError('This recording is for replay-continuing FB')
    repetitions=settings['rollouts'];pair_ids=np.flatnonzero(pairs['categories']==2)
    original=np.repeat(pair_ids,repetitions);replica=np.tile(np.arange(repetitions),len(pair_ids))
    starts=pairs['starts'][original];goals=pairs['goals'][original]
    random_ids=original*repetitions+replica
    checkpoint=args.run/'epoch_0800.pkl';digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    cache=args.output/'traces.npz'
    if cache.exists():
        record=dict(np.load(cache))
        if str(record.pop('checkpoint_sha256'))!=digest:raise ValueError('Checkpoint changed')
    else:
        learner=FBLearner.from_checkpoint(checkpoint,device=args.device);fn=trace_function(learner,env,150)
        batches=[]
        for first in range(0,len(starts),256):
            sl=slice(first,first+256)
            keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(settings['seed']+1),i))(jnp.asarray(random_ids[sl],jnp.uint32))
            batches.append(jax.device_get(fn(learner.state['params'],jnp.asarray(starts[sl]),jnp.asarray(goals[sl]),keys)))
        record={k:np.concatenate([b[k] for b in batches],axis=0 if k=='hits' else 1) for k in batches[0]}
        np.savez_compressed(cache,checkpoint_sha256=np.array(digest),**record)
    expected=np.load(args.evaluation/f'{args.run.name}_epsilon002.npz')['hits'][pair_ids].reshape(-1)
    if not np.array_equal(expected,record['hits']):raise ValueError('Trajectory replay does not reproduce saved MC outcomes')
    if not np.array_equal(record['position'][record['blocked']],record['following'][record['blocked']]):
        raise ValueError('Rejected moves changed position')
    if not np.array_equal(record['position'][0],starts):
        raise ValueError('Recorded starting positions do not match the evaluation pairs')
    if args.example_trials:
        for i in args.example_trials:
            if not 0 <= i < len(starts) or record['hits'][i] <= 150:
                raise ValueError(f'Trial {i} is not a recorded 150-step failure')
            metrics=wall_metrics(record['position'][:,i],record['blocked'][:,i],record['action'][:,i])
            stem=f'recorded_failure_pair{original[i]:04d}_rep{replica[i]:02d}'
            title=(f'Goal pair {original[i]}, rollout {replica[i]}\n'
                f'{metrics["collisions"]}/150 wall hits; longest stationary run {metrics["longest_stationary_collision_run"]}')
            record_example(args.output,env,record,starts,goals,i,stem=stem,title=title)
            metadata=dict(trial=i,pair=int(original[i]),replica=int(replica[i]),
                checkpoint_sha256=digest,epsilon=.02,horizon=150,start=starts[i].tolist(),
                goal=goals[i].tolist(),**metrics)
            (args.output/f'{stem}.json').write_text(json.dumps(metadata,indent=2))
            print(json.dumps(metadata,indent=2),flush=True)
        return
    n=len(starts);all_metrics={};summary=dict(checkpoint_sha256=digest,trajectories=n,diagonal_goal_pairs=len(pair_ids),
        repetitions=repetitions,epsilon=.02,exact_mc_replay=True,horizons={})
    for horizon in (30,150):
        rows=[]
        for i in range(n):
            steps=min(int(record['hits'][i]),horizon)
            row=wall_metrics(record['position'][:steps,i],record['blocked'][:steps,i],record['action'][:steps,i])
            row.update(trial=i,pair=int(original[i]),replica=int(replica[i]),steps=steps,success=bool(record['hits'][i]<=horizon))
            rows.append(row)
        all_metrics[horizon]=rows
        failed=[r for r in rows if not r['success']];colliding=[r for r in failed if r['collisions']>0]
        summary['horizons'][horizon]=dict(failures=len(failed),successes=n-len(failed),
            failed_with_any_wall_hit=len(colliding),
            failed_with_10_stationary_hits=sum(r['longest_stationary_collision_run']>=10 for r in failed),
            failed_with_20_stationary_hits=sum(r['longest_stationary_collision_run']>=20 for r in failed),
            failed_with_10_identical_action_wall_hits=sum(r['longest_same_action_collision_run']>=10 for r in failed),
            median_longest_stationary_run=float(np.median([r['longest_stationary_collision_run'] for r in failed])),
            median_dominant_neighborhood_fraction_among_colliding_failures=float(np.median([r['dominant_neighborhood_fraction'] for r in colliding])),
            failed_with_80pct_collisions_in_primary_neighborhood=sum(r['dominant_neighborhood_fraction']>=.8 for r in failed),
            wall_fraction_failed=sum(r['collisions'] for r in failed)/sum(r['steps'] for r in failed),
            repeat_wall_action_fraction_failed=sum(r['repeated_same_action_collisions'] for r in failed)/sum(r['steps'] for r in failed))
        (args.output/f'metrics_{horizon}.json').write_text(json.dumps(rows,indent=2))
    rooms=np.asarray(coordinate_room(jnp.asarray(record['following'].reshape(-1,2)))).reshape(150,n)
    committed=pairs['start_rooms'][original].copy();crossings=np.zeros(n,int)
    for t in range(150):
        valid=record['active'][t]&(rooms[t]>=0);crossings+=valid&(rooms[t]!=committed)
        committed=np.where(valid,rooms[t],committed)
    failed=np.flatnonzero(record['hits']>150)
    post_cross=[i for i in failed if crossings[i]>0 and all_metrics[150][i]['longest_stationary_collision_run']>=10]
    candidates=post_cross or failed.tolist()
    candidates=sorted(candidates,key=lambda i:all_metrics[150][i]['longest_stationary_collision_run'])
    example=candidates[len(candidates)//2]
    summary['recorded_example']=dict(index=int(example),room_crossings=int(crossings[example]),
        **all_metrics[150][example])
    # Diverse goal IDs at stationary-streak quantiles, not only the worst failures.
    selected=[];used=set()
    for quantile in (.1,.3,.5,.7,.85,.95):
        target=np.quantile([all_metrics[150][i]['longest_stationary_collision_run'] for i in failed],quantile)
        for i in sorted(failed,key=lambda i:abs(all_metrics[150][i]['longest_stationary_collision_run']-target)):
            if original[i] not in used:selected.append(int(i));used.add(original[i]);break
    gallery(args.output,env,record,starts,goals,all_metrics[150],selected,original)
    heatmaps(args.output,env,record,failed)
    record_example(args.output,env,record,starts,goals,example)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

def gallery(output,env,record,starts,goals,metrics,selected,original):
    fig,axes=plt.subplots(2,3,figsize=(13,9))
    for ax,i in zip(axes.flat,selected):
        draw_path(ax,env,record,i,starts,goals);m=metrics[i]
        ax.set_title(f'Goal pair {original[i]}: {m["collisions"]}/150 wall hits\nLongest stationary run: {m["longest_stationary_collision_run"]}',fontsize=10)
    axes.flat[0].legend(fontsize=7,loc='upper left');fig.suptitle('Replay continuing, epoch 800: diagonal failures (150 steps)')
    fig.tight_layout();fig.savefig(output/'failure_trajectories.png',dpi=160);plt.close(fig)

def heatmaps(output,env,record,failed):
    points=[];unique_trials=[]
    bins=np.linspace(0,1,51)
    for i in failed:
        p=record['position'][:,i][record['blocked'][:,i]];points.extend(p)
        if len(p):
            cell=np.clip(np.floor(p*50).astype(int),0,49);cell=np.unique(cell,axis=0)
            unique_trials.extend((cell+.5)/50)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,data,title in zip(axes,[points,unique_trials],['All rejected moves (repeat-weighted)','Failed trajectories hitting each cell (once per trial)']):
        h,_,_=np.histogram2d(*np.asarray(data).T,bins=[bins,bins])
        draw_world(ax,env);im=ax.imshow(np.log1p(h.T),origin='lower',extent=[0,1,0,1],cmap='magma',zorder=1)
        ax.set_title(title,fontsize=10);fig.colorbar(im,ax=ax,label='log(1 + count)',fraction=.046)
    fig.tight_layout();fig.savefig(output/'collision_hotspots.png',dpi=170);plt.close(fig)

def record_example(output,env,record,starts,goals,i,stem='recorded_failure',title='One recorded diagonal failure'):
    fig=plt.figure(figsize=(11,6));grid=fig.add_gridspec(2,2,width_ratios=[1,1.15]);ax=fig.add_subplot(grid[:,0])
    draw_path(ax,env,record,i,starts,goals);ax.legend(fontsize=7);ax.set_title(title,fontsize=11)
    action_ax=fig.add_subplot(grid[0,1]);t=np.arange(1,151);blocked=record['blocked'][:,i]
    action_ax.scatter(t,record['action'][:,i],c=np.where(blocked,'#dd3535','#277ca7'),s=12)
    action_ax.set(yticks=range(4),yticklabels=ACTIONS,xlim=(0,151),ylabel='Selected action',title='Red = rejected move; blue = accepted')
    coord_ax=fig.add_subplot(grid[1,1]);path=np.vstack([starts[i],record['following'][:,i]])
    coord_ax.plot(np.arange(151),path[:,0],label='x');coord_ax.plot(np.arange(151),path[:,1],label='y')
    coord_ax.set(xlabel='Step',ylabel='Position coordinate',ylim=(0,1));coord_ax.legend();coord_ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/f'{stem}.png',dpi=170);plt.close(fig)
    csv_name='recorded_trajectory.csv' if stem=='recorded_failure' else f'{stem}.csv'
    with (output/csv_name).open('w',newline='') as stream:
        fields=['step','x','y','next_x','next_y','action','wall_collision','Q_LEFT','Q_RIGHT','Q_UP','Q_DOWN','goal_x','goal_y']
        writer=csv.writer(stream);writer.writerow(fields)
        for t in range(150):writer.writerow([t+1,*record['position'][t,i],*record['following'][t,i],ACTIONS[record['action'][t,i]],bool(blocked[t]),*record['q'][t,i],*goals[i]])
    fig,ax=plt.subplots(figsize=(5,5));draw_world(ax,env)
    tolerance=env.threshold_distance
    ax.add_patch(Rectangle(goals[i]-tolerance,2*tolerance,2*tolerance,fill=False,linestyle=':',edgecolor='#bd7d12',lw=1))
    ax.scatter(*starts[i],color='#21a65b',s=40,label='Start');ax.scatter(*goals[i],marker='*',s=120,color='#ed9e18',edgecolors='black',label='Goal')
    line,=ax.plot([],[],color='#277ca7',lw=2);agent,=ax.plot([],[],'o',ms=7);text=ax.text(.01,1.02,'',transform=ax.transAxes,fontsize=9)
    ax.legend(fontsize=7,loc='lower left')
    def animate(frame):
        line.set_data(path[:frame+1,0],path[:frame+1,1]);agent.set_data([path[frame,0]],[path[frame,1]])
        agent.set_color('#dd3535' if frame>0 and blocked[frame-1] else '#21a65b')
        text.set_text(f'Step {frame}/150 | rejected moves {int(blocked[:frame].sum())}')
        return line,agent,text
    animation=FuncAnimation(fig,animate,frames=range(151),interval=100,blit=False)
    animation.save(output/f'{stem}.gif',writer=PillowWriter(fps=10));plt.close(fig)

if __name__=='__main__':main()
