"""Train unchanged Four Rooms with the released Continuous Maze FB configuration."""
from fb_navigation.paths import PROJECT_ROOT, source_path
import argparse
import hashlib
import json
import pickle
import platform
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from fb_navigation.environments.continuous_four_rooms_jax import ContinuousFourRooms
from fb_navigation.agents.forward_backward_jax import FBLearner, RadialBasisFunction2D
from fb_navigation.training.run_fb_jax import make_epoch
from fb_navigation.evaluation.fourrooms import make_pairs, evaluate_checkpoint, plot_history

ROOT = PROJECT_ROOT
CONFIG = ROOT/'experiments/fourrooms_reference/config.json'


def create_learner(config, seed, device):
    return FBLearner(**config['learner'], seed=seed, device=device, buffer_device=device,
                     featuriser=RadialBasisFunction2D(**config['rbf'], device=device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/fourrooms_reference/runs/seed0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'config.json'
    if manifest.exists() and not args.resume:
        raise FileExistsError('Use a fresh output directory or --resume')
    if args.resume and json.loads(manifest.read_text()) != config:
        raise ValueError('Resume config mismatch')
    sources = ['run_fourrooms_reference.py', 'fourrooms_reference_diagnostics.py', 'run_fb_jax.py',
               'continuous_four_rooms_jax.py', 'continuous_four_rooms.py',
               'forward_backward_jax.py', 'control_bfs_bottlenecks.py',
               'estimate_fb_bottlenecks_jax.py', 'paths.py']
    hashes = {s: hashlib.sha256((source_path(s)).read_bytes()).hexdigest() for s in sources}
    if args.resume and json.loads((output/'metadata.json').read_text())['sha256'] != hashes:
        raise ValueError('Resume source mismatch; restore archived sources')
    if args.resume and json.loads((output/'metadata.json').read_text())['seed'] != args.seed:
        raise ValueError('Resume seed mismatch')
    manifest.write_text(json.dumps(config, indent=2))
    if not args.resume:
        (output/'source').mkdir()
        for name in sources:
            (output/'source'/name).write_bytes((source_path(name)).read_bytes())
        (output/'metadata.json').write_text(json.dumps(dict(seed=args.seed, jax=jax.__version__,
            source_paths={s: source_path(s).relative_to(ROOT).as_posix() for s in sources},
            numpy=np.__version__, python=platform.python_version(), sha256=hashes,
            schedule=dict(updates=config['epochs']*config['cycles']*config['updates'],
                          transitions=config['epochs']*config['cycles']*config['episodes']*config['steps'])), indent=2))
    env = ContinuousFourRooms(seed=args.seed, **config['environment'])
    learner = create_learner(config, args.seed, args.device)
    pairs_file = output/'evaluation_pairs.npz'
    if pairs_file.exists():
        with np.load(pairs_file) as data:
            pairs = {k: data[k] for k in data.files}
    else:
        pairs = make_pairs(env, config['evaluation_per_category'], config['evaluation_seed'])
        np.savez_compressed(pairs_file, **pairs)
    history, train_rows, start = [], [], 0
    losses = np.full((config['epochs'], config['cycles'], config['updates']), np.nan, np.float32)
    with jax.default_device(learner.device):
        if args.resume:
            with (output/'resume.pkl').open('rb') as stream:
                saved = pickle.load(stream)
            learner = FBLearner.from_checkpoint(output/saved['checkpoint'], device=args.device, buffer_device=args.device)
            start, history, train_rows = saved['epoch'], saved['history'], saved['training']
            losses[:start] = np.load(output/'losses.npy')[:start]
            carry = (learner.state, jax.device_put(saved['storage'], learner.device),
                     jnp.asarray(saved['ptr'], jnp.int32), jnp.asarray(saved['size'], jnp.int32),
                     jnp.asarray(saved['env_key']), jnp.asarray(saved['action_key']))
        else:
            carry = (learner.state, jax.device_put(learner.buffer.storage, learner.device),
                     jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32),
                     env._rng, jax.random.PRNGKey(args.seed))
        epoch_fn = make_epoch(env, learner, config['cycles'], config['episodes'], config['steps'], config['updates'])
        count = int(learner.state['optim'][0])
        for epoch in range(start, config['epochs']):
            before = time.perf_counter()
            carry, epoch_losses = epoch_fn(*carry)
            losses[epoch] = np.asarray(epoch_losses)
            if not np.isfinite(losses[epoch]).all():
                raise FloatingPointError('Nonfinite loss; previous epoch remains resumable')
            learner.state = carry[0]
            learner._sync_networks()
            count = int(learner.state['optim'][0])
            assert count == (epoch+1)*config['cycles']*config['updates']
            checkpoint = f'epoch_{epoch+1:04d}.pkl'
            learner.save(output/checkpoint)
            if epoch+1 in config['evaluation_epochs'] or epoch+1 == config['epochs']:
                rows = evaluate_checkpoint(learner, env, pairs, output/'evaluation', epoch+1,
                                            horizons=config['evaluation_horizons'])
                history.extend(rows)
                plot_history(history, output/'evaluation')
                for row in rows:
                    print(json.dumps(dict(evaluation=row)), flush=True)
            train_rows.append(dict(epoch=epoch+1, updates=count,
                transitions=(epoch+1)*config['cycles']*config['episodes']*config['steps'],
                loss_mean=float(losses[epoch].mean()), loss_last=float(losses[epoch, -1, -1]),
                seconds=time.perf_counter()-before))
            np.save(output/'losses.npy', losses)
            (output/'training.json').write_text(json.dumps(train_rows, indent=2))
            saved = dict(epoch=epoch+1, checkpoint=checkpoint, history=history, training=train_rows,
                         storage=jax.device_get(carry[1]), ptr=int(carry[2]), size=int(carry[3]),
                         env_key=np.asarray(carry[4]), action_key=np.asarray(carry[5]))
            with (output/'resume.tmp').open('wb') as stream:
                pickle.dump(saved, stream, protocol=pickle.HIGHEST_PROTOCOL)
            (output/'resume.tmp').replace(output/'resume.pkl')
            print(json.dumps(train_rows[-1]), flush=True)
    (output/'complete.json').write_text(json.dumps(dict(epochs=config['epochs'], updates=count, evaluation=history), indent=2))


if __name__ == '__main__':
    main()
