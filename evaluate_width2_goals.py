"""Fresh matched goal-reaching test for width-2 Cauchy and replay-stopped models."""
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
from adjustable_four_rooms import AdjustableFourRooms,fixed_pairs
from compare_navigation_stages import mc_rollout
from forward_backward_jax import FBLearner
from audit_doorway_goals import point_status

CATEGORIES=('same_room','adjacent_rooms','diagonal_rooms')

def summarize(record,pairs,label,policy,horizon):
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

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('experiments/fourrooms_doorways'))
    parser.add_argument('--output',type=Path,default=Path('experiments/fourrooms_doorways/analyses/goal_reaching'))
    parser.add_argument('--pairs-per-category',type=int,default=200)
    parser.add_argument('--rollouts',type=int,default=32)
    parser.add_argument('--batch-size',type=int,default=512)
    parser.add_argument('--seed',type=int,default=20261107)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if min(args.pairs_per_category,args.rollouts,args.batch_size)<1:parser.error('Counts must be positive')
    env=AdjustableFourRooms(doorway_scale=2,include_stay_action=False)
    pairs=fixed_pairs(env,args.pairs_per_category,args.seed)
    validity=dict(starts=point_status(pairs['starts'],env),goals=point_status(pairs['goals'],env))
    if validity['starts']['bad_indices'] or validity['goals']['bad_indices']:raise ValueError('Invalid endpoints')
    settings=dict(seed=args.seed,rollouts=args.rollouts,pairs_per_category=args.pairs_per_category,
        jax=jax.__version__,device=args.device,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),validity=validity)
    manifest=args.output/'config.json'
    if manifest.exists() and json.loads(manifest.read_text())!=settings:raise ValueError('Use a fresh output directory')
    manifest.write_text(json.dumps(settings,indent=2));np.savez_compressed(args.output/'pairs.npz',**pairs)
    results=[];n=len(pairs['starts']);ids=np.repeat(np.arange(n),args.rollouts)
    for label in ('width2_seed0','width2_replay_stopped_seed0'):
        path=args.root/'runs'/label/'epoch_0800.pkl'
        learner=FBLearner.from_checkpoint(path,device=args.device)
        if learner.n_actions!=4 or int(learner.state['optim'][0])!=800000:raise ValueError('Unexpected checkpoint')
        for policy,epsilon in [('epsilon002',.02),('greedy',0.),('random',1.)]:
            if policy=='random' and label!='width2_seed0':continue
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
            for horizon in (30,150):results.extend(summarize(record,pairs,label,policy,horizon))
            print(json.dumps(dict(model=label,policy=policy,success30=[r['success_rate'] for r in results[-6:-3]])),flush=True)
    (args.output/'results.json').write_text(json.dumps(results,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,h in zip(axes,(30,150)):
        for k,label in enumerate(('width2_seed0','width2_replay_stopped_seed0','random')):
            rows=[r for r in results if r['horizon']==h and ((r['model']==label and r['policy']=='epsilon002') or (label=='random' and r['policy']=='random'))]
            ax.bar(np.arange(3)+(k-1)*.25,[r['success_rate'] for r in rows],width=.25,label=label)
        ax.set_xticks(range(3),CATEGORIES,rotation=12);ax.set(ylim=(0,1),ylabel='Goal success',title=f'{h} steps');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(args.output/'success.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
