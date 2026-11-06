"""Matched trajectory-stage ranking, collision and Monte Carlo diagnostics."""
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
from adjustable_four_rooms import AdjustableFourRooms
from continuous_four_rooms_jax import Point, _move
from fourrooms_reference_diagnostics import coordinate_room
from investigate_doorway_decline import door_rectangles, room_distance, nominal_transitions
from track_doorway_advantages import advantages, discounted_hits
from forward_backward_jax import FBLearner

STAGES=('first_door_approach','just_after_crossing','second_door_traversal')


def extract_probes(traces,pairs,scale,approach=.1,post_steps=3):
    rectangles=door_rectangles(scale)
    connections=[(0,2),(1,3),(0,1),(2,3)]
    rows=[]; availability={}
    old_success=traces[400]['hits']<=30;new_success=traces[800]['hits']<=30
    for epoch,trace in traces.items():
        shape=trace['following'].shape[:2]
        next_room=np.asarray(coordinate_room(jnp.asarray(trace['following'].reshape(-1,2)))).reshape(shape)
        counts={s:0 for s in STAGES}; reasons={}
        for i in np.flatnonzero(pairs['categories']==2):
            start,goal=int(pairs['start_rooms'][i]),int(pairs['goal_rooms'][i])
            steps=min(30,int(trace['hits'][i]))
            committed=start;first_cross=None;first_approach=None;committed_pre=[]
            for t in range(steps):
                committed_pre.append(committed)
                if first_cross is None and first_approach is None:
                    p=trace['position'][t,i]
                    for rectangle,(a,b) in zip(rectangles,connections):
                        if committed not in (a,b):continue
                        neighbor=b if committed==a else a
                        if room_distance(neighbor,goal)>=room_distance(committed,goal):continue
                        if np.linalg.norm(p-np.clip(p,rectangle[:2],rectangle[2:]))<=approach+1e-7:
                            first_approach=t;break
                current=int(next_room[t,i])
                if current>=0 and current!=committed:
                    if first_cross is None and room_distance(current,goal)<2:first_cross=t+1
                    committed=current
            times=dict(first_door_approach=first_approach,just_after_crossing=first_cross,
                       second_door_traversal=None if first_cross is None else first_cross+post_steps)
            for stage,time in times.items():
                if time is None:
                    reason='not_reached'
                elif time>=30:
                    reason='outside_original_budget'
                elif time>=int(trace['hits'][i]):
                    reason='already_successful'
                elif stage=='second_door_traversal' and committed_pre[time]==goal:
                    reason='already_in_goal_room'
                else:
                    reason=None
                if reason:
                    reasons[f'{stage}:{reason}']=reasons.get(f'{stage}:{reason}',0)+1;continue
                state=trace['position'][time,i]
                cohort=('lost' if old_success[i] and not new_success[i] else
                        'gained' if new_success[i] and not old_success[i] else
                        'kept_success' if old_success[i] else 'both_failed')
                rows.append(dict(stage=stage,source_epoch=epoch,trial=int(i),source_step=int(time),
                    remaining_budget=30-int(time),state=state.tolist(),goal=pairs['goals'][i].tolist(),
                    cohort=cohort,committed_room=int(committed_pre[time])))
                counts[stage]+=1
            # committed_pre includes every pre-action state before hit/horizon.
        availability[epoch]=dict(diagonal_trials=int(np.sum(pairs['categories']==2)),reached=counts,omitted=reasons)
    return rows,availability


def mc_rollout(learner,env,horizon=150,epsilon=.02):
    actions=jnp.array([(a.x,a.y) for a in env.actions]);walls=tuple(env.wall_blocks)
    @jax.jit
    def rollout(params,starts,goals,budgets,keys):
        tasks=learner.backward_network.apply(params['b'],learner.featuriser.transform(goals))
        n=len(starts);limits=jnp.stack([jnp.full(n,30),jnp.full(n,horizon),budgets],axis=1)
        def step(carry,t):
            position,hits,keys,collisions,repeats,longest,run,previous,previous_blocked=carry
            split=jax.vmap(lambda k:jax.random.split(k,4))(keys)
            keys,ek,ak,mk=[split[:,i] for i in range(4)]
            f=learner.forward_network.apply(params['f'],learner.featuriser.transform(position),tasks)
            greedy=jnp.argmax(jnp.einsum('bda,bd->ba',f,tasks),axis=1)
            explore=jax.vmap(lambda k:jax.random.uniform(k)<epsilon)(ek)
            random=jax.vmap(lambda k:jax.random.randint(k,(),0,4))(ak)
            selected=jnp.where(explore,random,greedy)
            def move(p,a,key):
                nxt,_,blocked=_move(Point(*p),Point(*actions[a]),key,env._noise,env.size,walls)
                return jnp.array([nxt.x,nxt.y]),blocked
            following,blocked=jax.vmap(move)(position,selected,mk)
            active=hits>horizon;blocked=blocked&active
            eligible=active[:,None]&(t<limits)
            repeated=blocked&previous_blocked&(selected==previous)
            run=jnp.where(repeated,run+1,jnp.where(blocked,1,0))
            collisions+=eligible&blocked[:,None];repeats+=eligible&repeated[:,None]
            longest=jnp.where(eligible,jnp.maximum(longest,run[:,None]),longest)
            position=jnp.where(active[:,None],following,position)
            reached=jnp.all(abs(position-goals)<=env.threshold_distance,axis=1)
            hits=jnp.where(active&reached,t+1,hits)
            return (position,hits,keys,collisions,repeats,longest,run,selected,blocked),None
        counters=jnp.zeros((n,3),jnp.int32)
        initial=(starts,jnp.full(n,horizon+1,jnp.int32),keys,counters,counters,counters,
                 jnp.zeros(n,jnp.int32),jnp.full(n,-1,jnp.int32),jnp.zeros(n,bool))
        final,_=jax.lax.scan(step,initial,jnp.arange(horizon))
        return dict(hits=final[1],collisions=final[3],repeats=final[4],longest=final[5])
    return rollout


def evaluate(learner,env,rows,reps,seed,batch):
    states=np.array([r['state'] for r in rows],np.float32);goals=np.array([r['goal'] for r in rows],np.float32)
    budget=np.array([r['remaining_budget'] for r in rows]);n=len(rows)
    qparts=[]
    for first in range(0,n,128):
        sl=slice(first,first+128);z=learner.embed_B(goals[sl]);f=learner.embed_F(states[sl],z)
        qparts.append(np.asarray(jnp.einsum('bda,bd->ba',f,z)))
    q=np.concatenate(qparts);fn=mc_rollout(learner,env)
    indices=np.repeat(np.arange(n),reps);all_records=[]
    for first in range(0,len(indices),batch):
        sl=slice(first,first+batch);ids=np.arange(len(indices))[sl]
        keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(seed),i))(jnp.asarray(ids,jnp.uint32))
        selected=indices[sl]
        record=fn(learner.state['params'],jnp.asarray(states[selected]),jnp.asarray(goals[selected]),
                  jnp.asarray(budget[selected]),keys)
        all_records.append({k:np.asarray(v) for k,v in record.items()})
    result={k:np.concatenate([r[k] for r in all_records]).reshape((n,reps)+all_records[0][k].shape[1:]) for k in all_records[0]}
    result.update(q=q,advantages=advantages(q))
    return result


def summary_for(record,rows,mask,blocked,column):
    chosen=record['q'].argmax(1)
    limits=np.array([30 if column==0 else 150 if column==1 else r['remaining_budget'] for r in rows])
    hits=record['hits'];success=hits<=limits[:,None];actions=np.minimum(hits,limits[:,None])
    value=discounted_hits(hits,100000)*success
    return dict(probes=int(mask.sum()),original_trials=len(set(rows[i]['trial'] for i in np.flatnonzero(mask))),
        blocked_greedy_fraction=float(blocked[np.flatnonzero(mask),chosen[mask]].mean()),
        median_preferred_advantage=float(np.median(record['advantages'][np.flatnonzero(mask),chosen[mask]])),
        success_rate=float(success[mask].mean()),discounted_return=float(value[mask].mean()),
        wall_action_fraction=float(record['collisions'][mask,:,column].sum()/actions[mask].sum()),
        repeated_wall_action_fraction=float(record['repeats'][mask,:,column].sum()/actions[mask].sum()),
        fraction_rollouts_wall_streak5=float((record['longest'][mask,:,column]>=5).mean()),
        mean_remaining_budget=float(limits[mask].mean()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('experiments/fourrooms_doorways/fourrooms_doorways/runs/width2_seed0'))
    parser.add_argument('--traces',type=Path,default=Path('experiments/fourrooms_doorways/decline_analysis'))
    parser.add_argument('--output',type=Path,default=Path('experiments/fourrooms_doorways/stage_analysis'))
    parser.add_argument('--rollouts',type=int,default=32)
    parser.add_argument('--batch-size',type=int,default=512)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if args.rollouts<2 or args.batch_size<1:parser.error('Invalid Monte Carlo counts')
    args.output.mkdir(parents=True,exist_ok=True)
    config=json.loads((args.run/'config.json').read_text());env=AdjustableFourRooms(**config['environment'])
    pairs=dict(np.load(args.run/'evaluation_pairs.npz'))
    traces={ep:dict(np.load(args.traces/f'trace_{ep}.npz')) for ep in (400,800)}
    rows,availability=extract_probes(traces,pairs,env.doorway_scale)
    (args.output/'probes.json').write_text(json.dumps(rows,indent=2))
    settings=dict(rollouts=args.rollouts,seed=84117,source_horizon=30,first_approach_distance=.1,post_cross_steps=3,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),jax=jax.__version__,numpy=np.__version__,
        trace_sha256={str(ep):hashlib.sha256((args.traces/f'trace_{ep}.npz').read_bytes()).hexdigest() for ep in (400,800)})
    manifest=args.output/'config.json'
    if manifest.exists() and json.loads(manifest.read_text())!=settings:raise ValueError('Settings changed; choose a fresh output')
    manifest.write_text(json.dumps(settings,indent=2))
    print(json.dumps(dict(probes=len(rows),availability=availability)),flush=True)
    states=np.array([r['state'] for r in rows],np.float32);_,blocked=nominal_transitions(env,states)
    np.savez_compressed(args.output/'probe_geometry.npz',states=states,blocked=blocked)
    records={};summary=dict(availability=availability,measurements={},changes={})
    for epoch in (400,800):
        path=args.run/f'epoch_{epoch:04d}.pkl';digest=hashlib.sha256(path.read_bytes()).hexdigest();cache=args.output/f'epoch_{epoch}.npz'
        if cache.exists():
            record=dict(np.load(cache))
            if str(record['checkpoint_sha256'])!=digest:raise ValueError('Checkpoint changed')
        else:
            learner=FBLearner.from_checkpoint(path,device=args.device)
            record=evaluate(learner,env,rows,args.rollouts,84117,args.batch_size)
            record['checkpoint_sha256']=np.array(digest);np.savez_compressed(cache,**record)
        records[epoch]=record
        print(f'Evaluated epoch {epoch}',flush=True)
        for stage in STAGES:
            for cohort in ('all','lost','kept_success','both_failed','gained'):
                for source in ('both',400,800):
                    mask=np.array([r['stage']==stage and (cohort=='all' or r['cohort']==cohort) and
                                   (source=='both' or r['source_epoch']==source) for r in rows])
                    if not mask.any():continue
                    for col,label in enumerate(('fresh30','fresh150','remaining30')):
                        key=f'{epoch}:{stage}:{cohort}:{source}:{label}'
                        summary['measurements'][key]=summary_for(record,rows,mask,blocked,col)
    for stage in STAGES:
        mask=np.array([r['stage']==stage for r in rows]);a=records[400]['q'][mask];b=records[800]['q'][mask]
        summary['changes'][stage]=dict(preferred_action_flip_fraction=float((a.argmax(1)!=b.argmax(1)).mean()),
            full_rank_change_fraction=float(np.any(a.argsort(1)!=b.argsort(1),axis=1).mean()),
            probes=int(mask.sum()))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    export_and_plot(args.output,rows,records,blocked,summary)
    print('Saved '+str(args.output/'summary.json'),flush=True)


def export_and_plot(output,rows,records,blocked,summary):
    exported=[]
    for epoch,record in records.items():
        for i,row in enumerate(rows):
            for action,name in enumerate(('LEFT','RIGHT','UP','DOWN')):
                exported.append(dict(epoch=epoch,probe=i,stage=row['stage'],source_epoch=row['source_epoch'],
                    trial=row['trial'],cohort=row['cohort'],action=name,q=float(record['q'][i,action]),
                    advantage=float(record['advantages'][i,action]),blocked=bool(blocked[i,action])))
    with (output/'action_rankings.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(exported[0]));writer.writeheader();writer.writerows(exported)
    fig,axes=plt.subplots(2,2,figsize=(12,8));x=np.arange(3)
    for epoch,shift in [(400,-.18),(800,.18)]:
        values=[summary['measurements'][f'{epoch}:{stage}:all:both:fresh30'] for stage in STAGES]
        for ax,key in zip(axes.flat,('success_rate','blocked_greedy_fraction','wall_action_fraction','repeated_wall_action_fraction')):
            ax.bar(x+shift,[v[key] for v in values],width=.35,label=str(epoch));ax.set(ylabel=key,ylim=(0,1))
            ax.set_xticks(x,['First approach','Just crossed','Second-door attempt'],rotation=12)
    axes[0,0].legend();fig.tight_layout();fig.savefig(output/'stages.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
