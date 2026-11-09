"""Reevaluate the orthonormal final model on its saved monitor goals."""
import hashlib
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from fb_navigation.agents.forward_backward_jax import FBLearner
from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms
from fb_navigation.evaluation.fourrooms import CATEGORIES, make_rollout, summarize


def main():
    run = ROOT / 'experiments/fourrooms_doorways/runs/width2_replay_continuing_orthonormal_seed0_20261109_084240'
    checkpoint = run / 'epoch_0800.pkl'
    config = json.loads((run / 'config.json').read_text())
    pairs = dict(np.load(run / 'evaluation_pairs.npz'))
    learner = FBLearner.from_checkpoint(checkpoint, device='cpu')
    assert int(learner.state['optim'][0]) == 800000
    env = AdjustableFourRooms(seed=0, **config['environment'])
    manifest = dict(run=str(run.relative_to(ROOT)), epoch=800, device='cpu',
        jax=jax.__version__, numpy=np.__version__, epsilons=[.02, .1],
        pairs=600, rollouts_per_pair=1, horizons=[30, 150],
        endpoint_and_rollout_seed=int(pairs['trial_seed']),
        goal_threshold=env.threshold_distance, parameters='online F/B',
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        pair_file_sha256=hashlib.sha256((run / 'evaluation_pairs.npz').read_bytes()).hexdigest(),
        evaluator_sha256=hashlib.sha256((ROOT / 'fb_navigation/evaluation/fourrooms.py').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target = OUT / 'config.json'
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError('Evaluation inputs changed; use a fresh output folder')
    target.write_text(json.dumps(manifest, indent=2) + '\n')
    np.savez_compressed(OUT / 'pairs.npz', **pairs)
    rows, records = [], {}
    for epsilon in (.02, .1):
        tag = f'epsilon{round(epsilon*100):03d}'
        record = {}
        for horizon in (30, 150):
            fn = make_rollout(learner, env, horizon, epsilon=epsilon)
            batches = []
            for first in range(0, len(pairs['starts']), 100):
                stop = min(first + 100, len(pairs['starts']))
                keys = jax.vmap(lambda i: jax.random.fold_in(
                    jax.random.PRNGKey(int(pairs['trial_seed'])), i))(
                        jnp.arange(first, stop, dtype=jnp.uint32))
                hit, collisions, doors, _ = fn(learner.state['params'],
                    jnp.asarray(pairs['starts'][first:stop]),
                    jnp.asarray(pairs['goals'][first:stop]), keys)
                batches.append(tuple(np.asarray(v) for v in (hit, collisions, doors)))
            hit, collisions, doors = (np.concatenate([b[i] for b in batches]) for i in range(3))
            assert np.all((hit >= 1) & (hit <= horizon+1))
            assert np.all(collisions <= np.minimum(hit, horizon))
            rows.extend(dict(epoch=800, epsilon=epsilon, horizon=horizon, **r)
                        for r in summarize(hit, collisions, doors, pairs, horizon))
            record.update({f'hits_{horizon}': hit, f'collisions_{horizon}': collisions,
                           f'doors_{horizon}': doors})
            print(json.dumps(dict(epsilon=epsilon, horizon=horizon,
                success=[r['success_rate'] for r in rows[-3:]])), flush=True)
        assert np.array_equal(record['hits_30'] <= 30, record['hits_150'] <= 30)
        np.savez_compressed(OUT / f'{tag}_trials.npz', **record)
        records[epsilon] = record
    old = dict(np.load(run / 'evaluation/epoch_0800_trials.npz'))
    agreement = {k: dict(exact=bool(np.array_equal(v, old[k])),
                         differing_elements=int(np.sum(v != old[k])))
                 for k, v in records[.02].items()}
    rng = np.random.default_rng(20261109)
    comparisons = []
    for h in (30, 150):
        for cat, name in enumerate(CATEGORIES):
            mask = pairs['categories'] == cat
            a, b = (records[e][f'hits_{h}'][mask] <= h for e in (.1, .02))
            d = a.astype(float) - b.astype(float)
            boot = d[rng.integers(len(d), size=(20000, len(d)))].mean(1)
            comparisons.append(dict(category=name, horizon=h,
                delta_success_pp=float(100*d.mean()), gained=int((a & ~b).sum()),
                lost=int((~a & b).sum()),
                paired_bootstrap95_pp=(100*np.quantile(boot, [.025, .975])).tolist()))
    result = dict(rows=rows, comparisons=comparisons, saved_baseline_agreement=agreement)
    (OUT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Orthonormal final model: evaluation epsilon 0.1', '',
        'The epoch-800 checkpoint was evaluated on its 600 saved monitor pairs '
        '(200 per category), with one noisy rollout per pair, online F/B, '
        'doubled doorways and coordinate-wise goal tolerance 0.1. '
        'Epsilon 0.02 was rerun on the same local CPU backend as epsilon 0.1 '
        'to control for backend/version differences. Both policies use identical '
        'per-trial random seeds. No training or checkpoint changes were made.', '',
        '| Category | Horizon | Success eps=.02 | Success eps=.10 | Change (pp) | Paired 95% interval (pp) | Wall actions eps=.02 | Wall actions eps=.10 |',
        '| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |']
    for c in comparisons:
        a, b = [next(r for r in rows if r['epsilon']==e and r['horizon']==c['horizon']
                     and r['category']==c['category']) for e in (.02, .1)]
        lo, hi = c['paired_bootstrap95_pp']
        lines.append(f"| {c['category']} | {c['horizon']} | {100*a['success_rate']:.1f}% | "
            f"{100*b['success_rate']:.1f}% | {c['delta_success_pp']:+.1f} | [{lo:+.1f}, {hi:+.1f}] | "
            f"{a['wall_action_percent']:.2f}% | {b['wall_action_percent']:.2f}% |")
    lines += ['', 'Intervals use 20,000 paired endpoint bootstrap samples (seed 20261109). '
        'They do not capture training-seed uncertainty. Each pair has only one '
        'noise realization, so repeated rollouts would better establish robustness. '
        'Wall-action percentages pool executed actions and exclude actions after success.', '',
        'The policies differ only in their random-action probability: 2% versus 10%; '
        'a random choice may select the greedy action too. Longer-horizon gains '
        'are consistent with exploration helping escape traps, but this evaluation '
        'does not directly classify stationary trap escapes.', '',
        '## Saved baseline reproduction', '',
        'Exact array agreement with the original cluster epsilon-0.02 evaluation:', '',
        '```json', json.dumps(agreement, indent=2), '```', '',
        'Full settings and input hashes: [config.json](config.json). '
        'Metrics, paired gains/losses and baseline checks: [results.json](results.json).', '',
        'Reproduce from the repository root:', '', '```bash',
        'python experiments/fourrooms_doorways/analyses/orthonormal_epsilon010/evaluate.py', '```', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
