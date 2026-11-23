"""Doorway-width experiment; periodic retained models and atomic rolling recovery."""
from fb_navigation.paths import PROJECT_ROOT as ROOT, source_path
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
from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms, fixed_pairs
from fb_navigation.agents.forward_backward_jax import FBLearner, RadialBasisFunction2D
from fb_navigation.training.run_fb_jax import make_epoch
from fb_navigation.evaluation.fourrooms import evaluate_checkpoint, plot_history


def create_learner(config, seed, device):
    """Construct FB directly from the supplied config, without retired launchers."""
    learner_config = dict(config['learner'])
    environment = config.get('environment', {})
    if environment.get('action_space') == 'continuous':
        learner_config.update(n_actions=None, action_space='continuous',
                              action_limit=environment.get('step_size', 0.05))
        learner_config.setdefault('goal_threshold', environment.get('threshold_distance', 0.1))
    return FBLearner(**learner_config, seed=seed, device=device, buffer_device=device,
                     featuriser=RadialBasisFunction2D(**config['rbf'], device=device))


def retained_checkpoint(epoch, total, interval):
    return epoch % interval == 0 or epoch == total


def resume_sources_match(previous, current, source_dir, *, discrete):
    if previous == current:
        return True
    if not discrete or previous.keys() != current.keys():
        return False

    # Recognize only the verified pre-continuous source snapshot. Its discrete
    # optimizer, replay, and RNG streams can resume with this source upgrade.
    def fingerprint(contents):
        return hashlib.sha256(contents.replace(b'\r\n', b'\n')).hexdigest()

    def bundle(contents):
        return hashlib.sha256(''.join(map(fingerprint, contents)).encode('ascii')).hexdigest()

    runner = 'run_fourrooms_doorways.py'
    try:
        archived = {name: (source_dir/name).read_bytes() for name in current}
    except OSError:
        return False
    if any(hashlib.sha256(data).hexdigest() != previous[name] for name, data in archived.items()):
        return False
    return (
        fingerprint(archived[runner]) == '42b3949c0f4804bb8ae549d963b511451dcc49f9dc0d1ee56772bcf53b45a8e2'
        and bundle(data for name, data in archived.items() if name != runner)
        == 'e3ca50384f6f5dc51b55115d5243ea7d8c54dc6a8bc715ac18dfcd1f38599309'
        and bundle(source_path(name).read_bytes() for name in current if name != runner)
        == '645a301aa14e85fead2e07c95f046cb8cae9466cbeba819af1fb2ebb46f8bf32'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--doorway-scale', type=float)
    parser.add_argument('--checkpoint-every', type=int, default=200)
    parser.add_argument('--recovery-every', type=int, default=1,
                        help='Overwrite full replay/RNG recovery snapshot every N epochs')
    parser.add_argument('--log-every', type=int, default=10,
                        help='Print progress every N epochs; 0 disables progress printing')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error('--checkpoint-every must be positive')
    if args.recovery_every < 1 or args.log_every < 0:
        parser.error('--recovery-every must be positive and --log-every nonnegative')
    config = json.loads(args.config.read_text())
    if args.doorway_scale is None:
        args.doorway_scale = (config['environment'].get('doorway_scale', 1.0)
                              if config['environment'].get('action_space') == 'continuous' else 1.0)
    config['environment']['doorway_scale'] = args.doorway_scale
    config['checkpoint_every'] = args.checkpoint_every
    config['evaluation_epochs'] = sorted({1, config['epochs'], *range(args.checkpoint_every, config['epochs']+1, args.checkpoint_every)})
    env = AdjustableFourRooms(seed=args.seed, **config['environment'])
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'config.json'
    if manifest.exists() and not args.resume:
        raise FileExistsError('Choose a new output directory or use --resume')
    if args.resume and json.loads(manifest.read_text()) != config:
        raise ValueError('Resume config mismatch: supply the original doorway scale and options')
    sources = ['run_fourrooms_doorways.py', 'adjustable_four_rooms.py',
               'fourrooms_reference_diagnostics.py', 'run_fb_jax.py', 'continuous_four_rooms_jax.py',
               'continuous_four_rooms.py',
               'forward_backward_jax.py', 'control_bfs_bottlenecks.py', 'estimate_fb_bottlenecks_jax.py', 'paths.py']
    hashes = {s: hashlib.sha256((source_path(s)).read_bytes()).hexdigest() for s in sources}
    if args.resume:
        previous = json.loads((output/'metadata.json').read_text())
        if previous['seed'] != args.seed or not resume_sources_match(
                previous['sha256'], hashes, output/'source', discrete=env.action_space == 'discrete'):
            raise ValueError('Resume source/seed mismatch')
    else:
        manifest.write_text(json.dumps(config, indent=2))
        (output/'source').mkdir()
        for name in sources:
            (output/'source'/name).write_bytes((source_path(name)).read_bytes())
        (output/'metadata.json').write_text(json.dumps(dict(seed=args.seed, sha256=hashes,
            source_paths={s: source_path(s).relative_to(ROOT).as_posix() for s in sources},
            recovery_every=args.recovery_every, log_every=args.log_every,
            python=platform.python_version(), jax=jax.__version__, numpy=np.__version__,
            doorway_width=env.size*args.doorway_scale/11,
            wall_blocks=[[b.xmin, b.ymin, b.xmax, b.ymax] for b in env.wall_blocks]), indent=2))
    learner = create_learner(config, args.seed, args.device)
    pairs = fixed_pairs(env, config['evaluation_per_category'], config['evaluation_seed'])
    if not args.resume:
        np.savez_compressed(output/'evaluation_pairs.npz', **pairs)
        from fb_navigation.environments.continuous_four_rooms_jax import visualize_environment
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        visualize_environment(env, ax, draw_initial_position=False, draw_goal=False)
        ax.set_title(f'Doorway scale {args.doorway_scale:g}; width {env.size*args.doorway_scale/11:.4f}')
        fig.savefig(output/'environment.png', dpi=150); plt.close(fig)
    start, history, training = 0, [], []
    losses = np.full((config['epochs'], config['cycles'], config['updates']), np.nan, np.float32)
    with jax.default_device(learner.device):
        if args.resume:
            with (output/'resume.pkl').open('rb') as stream:
                saved = pickle.load(stream)
            start, history, training = saved['epoch'], saved['history'], saved['training']
            if (output/'training.json').exists():
                recorded = json.loads((output/'training.json').read_text())
                if len(recorded) >= start and recorded[start-1]['epoch'] == start:
                    training = recorded[:start]
            losses[:start] = saved['losses']
            carry = jax.device_put(saved['carry'], learner.device)
            learner.state = carry[0]
        else:
            carry = (learner.state, jax.device_put(learner.buffer.storage, learner.device),
                     jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32), env._rng, jax.random.PRNGKey(args.seed))
        epoch_fn = make_epoch(env, learner, config['cycles'], config['episodes'], config['steps'], config['updates'])
        count = int(learner.state['optim'][0])
        for epoch in range(start, config['epochs']):
            before = time.perf_counter()
            carry, new_losses = epoch_fn(*carry)
            losses[epoch] = np.asarray(new_losses)
            train_seconds = time.perf_counter()-before
            if not np.isfinite(losses[epoch]).all():
                raise FloatingPointError('Nonfinite training loss')
            learner.state = carry[0]; learner._sync_networks()
            count = int(learner.state['optim'][0])
            assert count == (epoch+1)*config['cycles']*config['updates']
            save_started = time.perf_counter()
            checkpoint_due = retained_checkpoint(epoch+1, config['epochs'], args.checkpoint_every)
            if checkpoint_due:
                path = output/f'epoch_{epoch+1:04d}.pkl'
                learner.save(path.with_suffix('.tmp'))
                path.with_suffix('.tmp').replace(path)
            checkpoint_seconds = time.perf_counter()-save_started
            evaluation_started = time.perf_counter()
            if epoch+1 in config['evaluation_epochs']:
                history.extend(evaluate_checkpoint(learner, env, pairs, output/'evaluation', epoch+1,
                                                  horizons=config['evaluation_horizons']))
                plot_history(history, output/'evaluation')
            evaluation_seconds = time.perf_counter()-evaluation_started
            training.append(dict(epoch=epoch+1, updates=count,
                transitions=(epoch+1)*config['cycles']*config['episodes']*config['steps'],
                loss_mean=float(losses[epoch].mean()), train_seconds=train_seconds,
                checkpoint_seconds=checkpoint_seconds, evaluation_seconds=evaluation_seconds,
                recovery_seconds=0., seconds=0.))
            recovery_started = time.perf_counter()
            recovery_due = checkpoint_due or retained_checkpoint(epoch+1, config['epochs'], args.recovery_every)
            if recovery_due:
                # A single atomic recovery snapshot includes matching model,
                # replay, optimizer, RNG streams and losses. Avoid copying the
                # replay to host on epochs that do not need a recovery save.
                saved = dict(epoch=epoch+1, history=history, training=training,
                             carry=jax.device_get(carry), losses=losses[:epoch+1])
                with (output/'resume.tmp').open('wb') as stream:
                    pickle.dump(saved, stream, protocol=pickle.HIGHEST_PROTOCOL)
                (output/'resume.tmp').replace(output/'resume.pkl')
            training[-1]['recovery_seconds'] = time.perf_counter()-recovery_started
            training[-1]['seconds'] = time.perf_counter()-before
            # Keep the durable snapshot's timing metadata accurate without
            # rewriting its large payload: merge matching JSON rows on resume.
            log_due = args.log_every > 0 and (epoch == 0 or (epoch+1) % args.log_every == 0 or epoch+1 == config['epochs'])
            if log_due or recovery_due:
                np.save(output/'losses.npy', losses)
                (output/'training.json').write_text(json.dumps(training, indent=2))
            if log_due:
                print(json.dumps(training[-1]), flush=True)
    (output/'complete.json').write_text(json.dumps(dict(epochs=config['epochs'], updates=count, evaluation=history), indent=2))


if __name__ == '__main__':
    main()
