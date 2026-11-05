"""Compare FB successor measures with independent continuing Monte Carlo rollouts.
See EVALUATION.md for conventions, commands, and reference-distribution caveats.
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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from continuous_four_rooms_jax import ContinuousFourRooms, Point, _move
from forward_backward_jax import FBLearner, _cauchy_ball
from run_fb_jax import make_collector

ROOM_NAMES = ['bottom_left', 'top_left', 'bottom_right', 'top_right']
REWARD_NAMES = ROOM_NAMES + ['corridors', 'door_bottom', 'door_top', 'door_left', 'door_right']


def rewards(points, size=1.):
    """Exact continuous regions, independent of the plotting grid."""
    x, y = (points / size * 11).T
    rooms = jnp.stack(((x < 6) & (y < 4), (x < 6) & (y > 5),
                       (x > 7) & (y < 4), (x > 7) & (y > 5)), axis=1)
    doors = jnp.stack(((x >= 6) & (x <= 7) & (y >= 1) & (y <= 2),
                       (x >= 6) & (x <= 7) & (y >= 8) & (y <= 9),
                       (y >= 4) & (y <= 5) & (x >= 2) & (x <= 3),
                       (y >= 4) & (y <= 5) & (x >= 9) & (x <= 10)), axis=1)
    return jnp.concatenate((rooms, ~jnp.any(rooms, axis=1, keepdims=True), doors), axis=1).astype(jnp.float32)


def bin_ids(points, bins, size):
    xy = jnp.clip(jnp.floor(points / size * bins).astype(jnp.int32), 0, bins-1)
    return xy[:, 1]*bins + xy[:, 0]


def make_mc(learner, env, rollouts, horizon, bins):
    actions = jnp.array([(a.x, a.y) for a in env.actions])
    walls = tuple(env.wall_blocks)
    # Every action is available everywhere in this environment, including moves
    # rejected by walls. This is the same mask stored by the training collector.
    mask = jnp.zeros(learner.n_actions, dtype=bool).at[jnp.array(env.get_available_actions(None))].set(True)
    gamma = learner.discount_factor

    def run(params, start, initial_action, z, key):
        positions = jnp.broadcast_to(start, (rollouts, 2))
        histogram = jnp.zeros((rollouts, bins*bins))
        returns = jnp.zeros((rollouts, len(REWARD_NAMES)))
        def step(carry, t):
            positions, histogram, returns, key = carry
            # t=0 is the current state, not the post-action state.
            weight = gamma**t
            histogram = histogram.at[jnp.arange(rollouts), bin_ids(positions, bins, env.size)].add(weight)
            returns = returns + weight*rewards(positions, env.size)
            key, policy_key, dynamics_key = jax.random.split(key, 3)
            def sample_policy(_):
                features = learner.featuriser.transform(positions)
                f = learner.forward_network.apply(params, features, jnp.broadcast_to(z, (rollouts, learner.embed_dim)))
                logits = jnp.einsum('kda,d->ka', f, z) / learner.temperature
                logits = jnp.where(mask[None], logits, -jnp.inf)
                return jax.random.categorical(policy_key, logits, axis=-1).astype(jnp.int32)
            selected = jax.lax.cond(t == 0, lambda _: jnp.full((rollouts,), initial_action, jnp.int32), sample_policy, None)
            def move(p, a, k):
                delta = actions[a]
                result, _, _ = _move(Point(p[0], p[1]), Point(delta[0], delta[1]), k, env._noise, env.size, walls)
                return jnp.stack((result.x, result.y))
            positions = jax.vmap(move)(positions, selected, jax.random.split(dynamics_key, rollouts))
            return (positions, histogram, returns, key), None
        (_, hist, ret, _), _ = jax.lax.scan(step, (positions, histogram, returns, key), jnp.arange(horizon))
        return hist, ret
    return jax.jit(run)


def reconstruct_reference(env, cycles, capacity, samples, seed, path, updates):
    """Replay the actual epsilon=1 collector, sampling the final replay window.

    This is conditional reconstruction: original runtime/backend and collection
    settings must match. Never label it as observations saved during training.
    """
    total = cycles*120
    rng = np.random.default_rng(seed+104729)
    times = np.sort(rng.integers(max(0,total-capacity), total, size=samples))
    result = np.empty((samples,2), np.float32)
    # Chunk compilation once; the final smaller chunk is separately compiled.
    keys = (env._rng, jax.random.PRNGKey(seed), jax.random.PRNGKey(0))
    offset = 0
    for first in range(0, cycles, 100):
        count = min(100, cycles-first)
        collect = jax.jit(make_collector(env, episodes=count*4, steps=30)) if first in (0, cycles-cycles%100) else collect
        keys, batch = collect(*keys)
        obs = np.asarray(batch['obs'])
        lo, hi = np.searchsorted(times, [offset, offset+len(obs)])
        result[lo:hi] = obs[times[lo:hi]-offset]
        offset += len(obs)
        if first % 1000 == 0:
            print(f'Reconstructing replay: {min(first+count,cycles)}/{cycles} cycles', flush=True)
    np.savez_compressed(path, observations=result, source='reconstructed_final_replay',
                        seed=seed, cycles=cycles, training_updates=updates,
                        size=env.size, movement_noise=env._noise,
                        actions=np.asarray([(a.x,a.y) for a in env.actions]))
    return result, 'reconstructed_final_replay'


def load_reference(path, env, updates, checkpoint=None):
    with np.load(path, allow_pickle=False) as data:
        obs = data['observations'].copy()
        source = str(data['source'])
        if checkpoint is not None and 'checkpoint_sha256' in data:
            if str(data['checkpoint_sha256']) != hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
                raise ValueError('Reference belongs to a different checkpoint')
        if 'wall_blocks' in data:
            np.testing.assert_allclose(data['wall_blocks'],[(b.xmin,b.ymin,b.xmax,b.ymax) for b in env.wall_blocks])
        if 'training_updates' in data and int(data['training_updates']) != updates:
            raise ValueError('Reference and checkpoint training update counts differ')
        if 'size' in data and not np.isclose(float(data['size']),env.size):
            raise ValueError('Reference environment size mismatch')
        if 'movement_noise' in data and not np.isclose(float(data['movement_noise']),env._noise):
            raise ValueError('Reference environment noise mismatch')
        if 'actions' in data:
            np.testing.assert_allclose(data['actions'],[(a.x,a.y) for a in env.actions])
    if obs.ndim != 2 or obs.shape[1] != 2 or not np.isfinite(obs).all() or len(obs)<2:
        raise ValueError('Reference must contain finite (N,2) observations')
    if np.any((obs<0)|(obs>env.size)):
        raise ValueError('Reference observations outside environment')
    return obs, source


def configurations(learner, env, key):
    centers = np.array([[3,2],[3,8],[9,2],[9,8]],np.float32)/11*env.size
    tasks = [('goal_'+name, np.asarray(learner.embed_B(centers[i:i+1]))[0]) for i,name in enumerate(ROOM_NAMES)]
    random_z = np.asarray(_cauchy_ball(key, 4, learner.embed_dim))
    configs=[]
    for i,(name,z) in enumerate(tasks):
        for j in range(2):
            configs.append((f'{name}_start{(i+j+1)%4}_a{(i+j)%learner.n_actions}', centers[(i+j+1)%4], (i+j)%learner.n_actions,z))
    for i,z in enumerate(random_z):
        configs.append((f'random_{i}',centers[i],i%learner.n_actions,z))
    configs.append(('zero_task',centers[0],0,np.zeros(learner.embed_dim,np.float32)))
    return configs


def predict(learner, reference, start, action, z, bins, size, backward=None, batch=4096):
    f = np.asarray(learner.embed_F(start,z[None],action=action))[0]
    # Batching avoids creating N x RBF-dimension activations for all replay.
    if backward is None:
        backward=np.concatenate([np.asarray(learner.embed_B(reference[first:first+batch]))
                                 for first in range(0,len(reference),batch)])
    scores=backward @ f
    ids=np.asarray(bin_ids(jnp.asarray(reference),bins,size))
    hist=np.bincount(ids,weights=scores,minlength=bins*bins)/len(reference)
    region=np.asarray(rewards(jnp.asarray(reference),size))
    ret=(scores[:,None]*region).mean(axis=0)
    # Independent-reference-sample SE, not a correction or renormalisation.
    ret_se=(scores[:,None]*region).std(axis=0,ddof=1)/np.sqrt(len(reference))
    return hist,ret,ret_se,scores


def plot_maps(path, env, bins, mc, fb, title):
    fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    bound=max(float(np.max(mc)),float(np.max(np.abs(fb))),1e-8)
    norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound)
    for i,(values,label) in enumerate(((mc,'Monte Carlo'),(fb,'FB (signed)'),(abs(mc-fb),'Absolute error'))):
        image=axs[i].imshow(values.reshape(bins,bins),origin='lower',extent=(0,env.size,0,env.size),
                            cmap='RdBu_r' if i<2 else 'magma',norm=norm if i<2 else None)
        for b in env.wall_blocks:
            axs[i].add_patch(Rectangle((b.xmin,b.ymin),b.xmax-b.xmin,b.ymax-b.ymin,color='black'))
        axs[i].set_title(label);axs[i].set_aspect('equal')
        fig.colorbar(image,ax=axs[i],label='Discounted mass per bin')
    fig.suptitle(title);fig.savefig(path,dpi=150);plt.close(fig)


def smoke_test(learner,env,bins,seed):
    # Horizon one is exactly delta_start and is independent of forced action.
    fn=make_mc(learner,env,8,1,bins)
    start=jnp.array([.2,.2]); z=jnp.zeros(learner.embed_dim)
    hist,ret=fn(learner.state['target']['f'],start,jnp.array(1),z,jax.random.PRNGKey(seed))
    expected=np.zeros(bins*bins);expected[int(bin_ids(start[None],bins,env.size)[0])]=1
    np.testing.assert_array_equal(np.asarray(hist),np.broadcast_to(expected,hist.shape))
    np.testing.assert_allclose(ret,np.broadcast_to(np.asarray(rewards(start[None],env.size)),ret.shape))
    fn=make_mc(learner,env,8,8,bins)
    hist,ret=fn(learner.state['target']['f'],start,jnp.array(1),z,jax.random.PRNGKey(seed))
    np.testing.assert_allclose(np.asarray(hist).sum(axis=1),(1-learner.discount_factor**8)/(1-learner.discount_factor),rtol=2e-6)
    assert np.isfinite(hist).all() and np.isfinite(ret).all()
    print('Smoke tests passed: immediate-state indexing, reward regions, continuing mass.',flush=True)


def evaluate(args):
    if min(args.rollouts,args.horizon,args.bins,args.reference_samples)<2:
        raise ValueError('Rollouts, horizon, bins, and reference samples must be >=2')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    learner=FBLearner.from_checkpoint(args.checkpoint,device=args.device,buffer_device='cpu')
    with jax.default_device(learner.device):
        env=ContinuousFourRooms(include_stay_action=learner.n_actions==5, seed=learner.config['seed'])
    if learner.n_actions not in (4,5):
        raise ValueError('Expected the Four Rooms action set')
    updates=int(learner.state['optim'][0])
    reference_path=Path(args.reference)
    with jax.default_device(learner.device):
        smoke_test(learner,env,args.bins,args.seed)
        if args.smoke_only:
            return
        if reference_path.exists():
            reference,source=load_reference(reference_path,env,updates,args.checkpoint)
        elif args.reconstruct_reference_cycles:
            if updates != args.reconstruct_reference_cycles*40:
                raise ValueError('Reconstruction assumes 40 updates per collection cycle; counts disagree')
            reference,source=reconstruct_reference(env,args.reconstruct_reference_cycles,learner.buffer.max_size,
                         args.reference_samples,learner.config['seed'],reference_path,updates)
        else:
            raise FileNotFoundError('No replay reference. See EVALUATION.md; reconstruction requires explicit --reconstruct-reference-cycles.')
        if len(reference)>args.reference_samples:
            reference=reference[np.random.default_rng(args.seed).choice(len(reference),args.reference_samples,replace=False)]
        gamma=learner.discount_factor
        infinite=1/(1-gamma);finite=(1-gamma**args.horizon)/(1-gamma)
        mc=make_mc(learner,env,args.rollouts,args.horizon,args.bins)
        cases=configurations(learner,env,jax.random.PRNGKey(args.seed+1))
        if args.max_configs: cases=cases[:args.max_configs]
        print('Embedding replay reference once for all configurations',flush=True)
        backward=np.concatenate([np.asarray(learner.embed_B(reference[first:first+4096]))
                                 for first in range(0,len(reference),4096)])
        target_backward=np.concatenate([np.asarray(learner._embed_b(learner.state['target']['b'],
                                    jnp.asarray(reference[first:first+4096])))
                                    for first in range(0,len(reference),4096)])
        reference_bins=np.asarray(bin_ids(jnp.asarray(reference),args.bins,env.size))
        metrics=[];reward_rows=[];saved={};case_metadata=[]
        for i,(name,start,action,z) in enumerate(cases):
            print(f'Evaluating {i+1}/{len(cases)}: {name}',flush=True)
            hist,returns=mc(learner.state['target']['f'],jnp.asarray(start),jnp.array(action),jnp.asarray(z),jax.random.fold_in(jax.random.PRNGKey(args.seed),i))
            hist,returns=np.asarray(hist),np.asarray(returns)
            empirical=hist.mean(axis=0);mc_se=hist.std(axis=0,ddof=1)/np.sqrt(args.rollouts)
            fb,fb_ret,fb_ret_se,scores=predict(learner,reference,start,action,z,args.bins,env.size,backward)
            target_f=np.asarray(learner._embed_f(learner.state['target']['f'],jnp.asarray(start[None]),jnp.asarray(z[None])))[0,:,action]
            target_scores=target_backward @ target_f
            target_map=np.bincount(reference_bins,weights=target_scores,minlength=args.bins**2)/len(reference)
            ret_mean=returns.mean(axis=0);ret_se=returns.std(axis=0,ddof=1)/np.sqrt(args.rollouts)
            corr=np.corrcoef(empirical,fb)[0,1] if np.std(fb)>0 else None
            row=dict(configuration=name,normalised_l1=float(abs(empirical-fb).sum()/empirical.sum()),
                mc_mass=float(empirical.sum()),fb_mass=float(fb.sum()),theoretical_mass=infinite,
                finite_horizon_mass=finite,tail_mass_bound=infinite-finite,
                mass_error_vs_mc=float(fb.sum()-empirical.sum()),mass_error_vs_theory=float(fb.sum()-infinite),
                relative_mass_error_vs_theory=float(abs(fb.sum()-infinite)/infinite),
                negative_bin_fraction=float(np.mean(fb<0)),negative_mass=float(-fb[fb<0].sum()),
                negative_score_fraction=float(np.mean(scores<0)),negative_score_mass=float(-np.minimum(scores,0).mean()),
                correlation=None if corr is None else float(corr),
                mc_split_half_l1=float(abs(hist[::2].mean(0)-hist[1::2].mean(0)).sum()/finite),
                mc_pointwise_se_l1=float(mc_se.sum()),reference_source=source)
            row['target_normalised_l1']=float(abs(empirical-target_map).sum()/empirical.sum())
            row['target_mass']=float(target_map.sum())
            for r,region in enumerate(REWARD_NAMES):
                reward_rows.append(dict(configuration=name,reward=region,mc_return=float(ret_mean[r]),
                    fb_return=float(fb_ret[r]),signed_error=float(fb_ret[r]-ret_mean[r]),
                    mc_standard_error=float(ret_se[r]),reference_standard_error=float(fb_ret_se[r])))
                if r<5:
                    row[region+'_mass_error']=float(fb_ret[r]-ret_mean[r])
            metrics.append(row)
            case_metadata.append(dict(name=name,start=start.tolist(),initial_action=int(action),z=z.tolist()))
            saved[name+'_mc']=empirical;saved[name+'_fb']=fb;saved[name+'_mc_se']=mc_se
            saved[name+'_target_fb']=target_map
            plot_maps(out/(name+'.png'),env,args.bins,empirical,fb,name)
            print(f"  normalized L1={row['normalised_l1']:.3f}; FB mass={row['fb_mass']:.2f}; MC mass={row['mc_mass']:.2f}",flush=True)
        for filename,rows in [('metrics.csv',metrics),('reward_predictions.csv',reward_rows)]:
            with (out/filename).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        np.savez_compressed(out/'maps.npz',**saved)
        metadata=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            reference=str(reference_path),reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            reference_source=source,reference_count=len(reference),seed=args.seed,rollouts=args.rollouts,horizon=args.horizon,
            bins=args.bins,gamma=gamma,temperature=learner.temperature,policy='frozen target forward masked softmax',
            prediction='online F and B, signed and unnormalised',convention='current state at t=0, continuing dynamics, no resets',
            device=str(learner.device),configurations=case_metadata,metrics=metrics,
            caveat='Reference reconstructed with assumed original runner settings; original replay was not saved. Cross-backend numerical differences may change reset trajectories.' if source.startswith('reconstructed') else 'Final replay sample approximates the time-varying training reference; sampling uncertainty remains.')
        (out/'metrics.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))
        fig,ax=plt.subplots(figsize=(12,5));ax.bar(range(len(metrics)),[r['normalised_l1'] for r in metrics]);ax.set_xticks(range(len(metrics)),[r['configuration'] for r in metrics],rotation=75,ha='right');ax.set_ylabel('L1 error / MC total mass');fig.tight_layout();fig.savefig(out/'summary.png',dpi=150);plt.close(fig)
        mean=np.mean([r['normalised_l1'] for r in metrics]);mass=np.mean([r['relative_mass_error_vs_theory'] for r in metrics])
        report=f'''Mean normalized L1: {mean:.4f}\nMean relative total-mass error: {mass:.4f}\nFinite-horizon missing mass: {infinite-finite:.6f}\nReference: {source}\n{metadata['caveat']}\n\nThe empirical maps and arbitrary-reward returns, not training loss or goal reaching, are the validation evidence. Errors substantially larger than MC variability and the finite-horizon tail indicate a failure of the evaluated factorization/reference combination. These results alone cannot separate representation error from reference-distribution error. Frozen target-policy evaluation removes greedy/softmax mismatch; online/target lag and a time-varying replay reference can still contribute. No clipping or normalization is applied.\n'''
        (out/'interpretation.txt').write_text(report);print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='fb_fourrooms_jax.pkl')
    parser.add_argument('--reference',default='fb_fourrooms_jax_reference.npz')
    parser.add_argument('--reconstruct-reference-cycles',type=int,default=0)
    parser.add_argument('--reference-samples',type=int,default=100000)
    parser.add_argument('--rollouts',type=int,default=500)
    parser.add_argument('--horizon',type=int,default=700)
    parser.add_argument('--bins',type=int,default=20)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',default='evaluation_fb')
    parser.add_argument('--max-configs',type=int,default=0)
    parser.add_argument('--smoke-only',action='store_true')
    evaluate(parser.parse_args())
