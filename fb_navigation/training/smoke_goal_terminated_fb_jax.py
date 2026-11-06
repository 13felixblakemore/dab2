"""Small, reproducible stopped-FB training/Monte Carlo smoke comparison.

Writes only to --output; never loads or overwrites the user's trained models.
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from fb_navigation.environments.continuous_four_rooms_jax import ContinuousFourRooms, Point, _move
from fb_navigation.agents.forward_backward_jax import FBLearner, RadialBasisFunction2D, mask_action_scores
from fb_navigation.training.run_fb_jax import make_collector


def make_stopped_mc(learner,env,rollouts=128,horizon=700,bins=20):
    """Current-state visitation, with goal-entry states excluded (L2 radius).

    Only outside-goal starts are supported, matching the documented loss scope.
    The initial action is forced; later actions use the frozen target softmax.
    """
    actions=jnp.asarray([(a.x,a.y) for a in env.actions])
    walls=tuple(env.wall_blocks)
    mask=jnp.zeros(learner.n_actions,bool).at[jnp.array(env.get_available_actions(None))].set(True)
    @jax.jit
    def rollout(target,start,action,z,goal,key):
        positions=jnp.broadcast_to(start,(rollouts,2))
        histogram=jnp.zeros((rollouts,bins*bins))
        active=jnp.ones(rollouts,bool)
        def step(carry,t):
            positions,histogram,active,key=carry
            active=active & (jnp.sum((positions-goal)**2,axis=1)>learner.goal_threshold**2)
            xy=jnp.clip(jnp.floor(positions/env.size*bins).astype(jnp.int32),0,bins-1)
            ids=xy[:,1]*bins+xy[:,0]
            histogram=histogram.at[jnp.arange(rollouts),ids].add(active*learner.discount_factor**t)
            key,ak,mk=jax.random.split(key,3)
            def choose(_):
                f=learner.forward_network.apply(target,learner.featuriser.transform(positions),jnp.broadcast_to(z,(rollouts,learner.embed_dim)))
                logits=mask_action_scores(jnp.einsum('nda,d->na',f,z)/learner.temperature,mask)
                return jax.random.categorical(ak,logits,axis=-1).astype(jnp.int32)
            selected=jax.lax.cond(t==0,lambda _:jnp.full(rollouts,action,jnp.int32),choose,None)
            def move(p,a,k):
                delta=actions[a]
                nxt,_,_=_move(Point(*p),Point(*delta),k,env._noise,env.size,walls)
                return jnp.array((nxt.x,nxt.y))
            proposed=jax.vmap(move)(positions,selected,jax.random.split(mk,rollouts))
            positions=jnp.where(active[:,None],proposed,positions)
            active=active & (jnp.sum((positions-goal)**2,axis=1)>learner.goal_threshold**2)
            return (positions,histogram,active,key),None
        (_,hist,active,_),_=jax.lax.scan(step,(positions,histogram,active,key),jnp.arange(horizon))
        return hist,active
    def checked(target,start,action,z,goal,key):
        if float(jnp.sum((start-goal)**2))<=learner.goal_threshold**2:
            raise ValueError('Start-in-goal boundary is unsupported by the unchanged immediate loss term')
        return rollout(target,start,action,z,goal,key)
    return checked


def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    env=ContinuousFourRooms(include_stay_action=False,seed=args.seed)
    l=FBLearner(2,4,16,hidden_dim=64,num_layers=1,batch_size=64,buffer_size=15360,
        learning_rate=5e-4,device='cpu',buffer_device='cpu',z_sampling='replay',
        goal_terminated=True,goal_threshold=.1,zero_z_p=.25,seed=args.seed,
        featuriser=RadialBasisFunction2D((0,1),(0,1),7,.08))
    _,batch=jax.jit(make_collector(env,episodes=512,steps=30))(
        env._rng,jax.random.PRNGKey(args.seed+1),jax.random.PRNGKey(args.seed+2))
    l.buffer.add_batch(batch)
    l.buffer.storage=jax.device_put(l.buffer.storage);l.buffer._host=False
    # Independent reference draws from exactly the replay used in training.
    reference=np.asarray(l.buffer.obs).copy()
    losses=[]
    for first in range(0,args.updates,40):
        losses.append(l.update_networks(min(40,args.updates-first)))
        l.soft_update_target_networks()
    losses=np.asarray(jnp.concatenate(losses));assert np.isfinite(losses).all()
    l.save(out/'small_model.pkl');np.save(out/'training_losses.npy',losses)
    cases=[([.2,.2],[.35,.2],1),([.2,.7],[.8,.2],3),([.8,.8],[.2,.7],0)]
    mc=make_stopped_mc(l,env,args.rollouts,args.horizon)
    b=np.concatenate([np.asarray(l.embed_B(reference[i:i+2048])) for i in range(0,len(reference),2048)])
    xy=np.clip(np.floor(reference*20).astype(int),0,19);ids=xy[:,1]*20+xy[:,0]
    rows=[];maps={}
    for i,(start,goal,action) in enumerate(cases):
        start=jnp.array(start);goal=jnp.array(goal);z=l.embed_B(goal[None])[0]
        hist,active=mc(l.state['target']['f'],start,action,z,goal,jax.random.PRNGKey(args.seed+100+i))
        hist=np.asarray(hist);mean=hist.mean(axis=0)
        f=np.asarray(l.embed_F(start,z[None],action))[0]
        score=b@f;prediction=np.bincount(ids,weights=score,minlength=400)/len(reference)
        masses=hist.sum(axis=1)
        rows.append(dict(start=np.asarray(start).tolist(),goal=np.asarray(goal).tolist(),initial_action=action,
            mc_mass=float(mean.sum()),mc_mass_standard_error=float(masses.std(ddof=1)/np.sqrt(args.rollouts)),
            fb_mass=float(prediction.sum()),normalised_l1=float(abs(prediction-mean).sum()/mean.sum()),
            negative_fb_mass=float(-np.minimum(prediction,0).sum()),goal_reached_fraction=float(1-np.asarray(active).mean())))
        maps[f'case{i}_mc']=mean;maps[f'case{i}_fb']=prediction
    result=dict(updates=args.updates,seed=args.seed,rollouts=args.rollouts,horizon=args.horizon,
        gamma=l.discount_factor,goal_threshold=l.goal_threshold,temperature=l.temperature,
        policy='frozen target forward masked softmax, z=B_online(goal)',
        convention='current state at t=0; force initial action; exclude goal entry; no resets; reject starts in goal',
        reference='all 15360 raw observations in the fixed training replay',
        final_loss=float(losses[-1]),finite_losses=bool(np.isfinite(losses).all()),cases=rows,
        limitation='Small unvalidated model and only three tasks. This compares stopped estimates, not a convergence or accuracy proof.')
    (out/'metrics.json').write_text(json.dumps(result,indent=2));np.savez_compressed(out/'maps.npz',**maps)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--updates',type=int,default=3000);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--rollouts',type=int,default=128);p.add_argument('--horizon',type=int,default=700)
    p.add_argument('--output',default='evaluation_goal_terminated_smoke')
    args=p.parse_args()
    if min(args.updates,args.rollouts,args.horizon)<2:p.error('Counts must be >=2')
    main(args)
