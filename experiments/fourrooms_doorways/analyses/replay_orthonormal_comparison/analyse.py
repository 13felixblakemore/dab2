"""Compare saved monitor trials; no training or policy rollouts are rerun."""
import ast
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
RUNS = OUT.parents[1] / 'runs'
NAMES = {
    'orthonormal': 'width2_replay_continuing_orthonormal_seed0_20261109_084240',
    'released': 'width2_replay_continuing_seed0',
    'cauchy': 'width2_seed0',
}
CATEGORIES = ('same_room', 'adjacent_rooms', 'diagonal_rooms')
EPOCHS = (200, 400, 600, 800)


def read(path):
    return json.loads(path.read_text())


def functions(path):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    return {n.name: ast.dump(n, include_attributes=False) for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def main():
    paths = {k: RUNS / v for k, v in NAMES.items()}
    pairs = {k: dict(np.load(p / 'evaluation_pairs.npz')) for k, p in paths.items()}
    reference = pairs['released']
    for data in pairs.values():
        assert data.keys() == reference.keys()
        assert all(np.array_equal(data[k], reference[k]) for k in reference)
    configs = {k: read(p / 'config.json') for k, p in paths.items()}
    old, new = json.loads(json.dumps(configs['released'])), configs['orthonormal']
    old['learner']['b_regularization'] = 'orthonormal'
    assert old == new
    integrity = {}
    for label, p in paths.items():
        metadata = read(p / 'metadata.json')
        for name, digest in metadata['sha256'].items():
            assert hashlib.sha256((p / 'source' / name).read_bytes()).hexdigest() == digest
        complete = read(p / 'complete.json')
        training = read(p / 'training.json')
        losses = np.load(p / 'losses.npy')
        assert complete['epochs'] == 800 and complete['updates'] == 800000
        assert len(training) == 800 and losses.shape == (800, 25, 40)
        assert np.isfinite(losses).all()
        integrity[label] = dict(epochs=800, updates=800000,
            transitions=training[-1]['transitions'], finite_losses=True,
            recorded_seconds=sum(t['seconds'] for t in training),
            versions={k: metadata[k] for k in ('seed', 'python', 'jax', 'numpy')})
    old_source = paths['released'] / 'source'
    new_source = paths['orthonormal'] / 'source'
    assert (old_source / 'forward_backward_jax.py').read_bytes() == (new_source / 'forward_backward_jax.py').read_bytes()
    source_checks = {}
    for filename, names in {
        'fourrooms_reference_diagnostics.py': ('make_pairs', 'coordinate_room', 'make_rollout', 'summarize', 'evaluate_checkpoint'),
        'continuous_four_rooms_jax.py': ('Point', 'WallBlock', '_move', '_sample_point', 'ContinuousWorld', 'ContinuousFourRooms'),
        'run_fb_jax.py': ('make_epoch',),
    }.items():
        a, b = functions(old_source / filename), functions(new_source / filename)
        for name in names:
            assert a[name] == b[name], (filename, name)
            source_checks[f'{filename}:{name}'] = True
    rows, trials = {}, {}
    for label, p in paths.items():
        rows[label] = {e: read(p / f'evaluation/epoch_{e:04d}.json') for e in EPOCHS}
        trials[label] = {e: dict(np.load(p / f'evaluation/epoch_{e:04d}_trials.npz')) for e in EPOCHS}
        for e in EPOCHS:
            for r in rows[label][e]:
                selected = reference['categories'] == CATEGORIES.index(r['category'])
                h = r['horizon']
                hits = trials[label][e][f'hits_{h}'][selected]
                blocked = trials[label][e][f'collisions_{h}'][selected]
                assert np.isclose((hits <= h).mean(), r['success_rate'])
                assert np.isclose(100 * blocked.sum() / np.minimum(hits, h).sum(), r['wall_action_percent'])
    rng = np.random.default_rng(20261109)
    comparisons = []
    for comparator in ('released', 'cauchy'):
        for h in (30, 150):
            for cat, name in enumerate(CATEGORIES):
                selected = reference['categories'] == cat
                a = trials['orthonormal'][800][f'hits_{h}'][selected] <= h
                b = trials[comparator][800][f'hits_{h}'][selected] <= h
                d = a.astype(float) - b.astype(float)
                boot = d[rng.integers(len(d), size=(20000, len(d)))].mean(axis=1)
                comparisons.append(dict(comparator=comparator, horizon=h, category=name,
                    pairs=len(d), orthonormal_success_rate=float(a.mean()),
                    comparator_success_rate=float(b.mean()), delta_pp=float(100*d.mean()),
                    paired_bootstrap95_pp=(100*np.quantile(boot, [.025, .975])).tolist(),
                    gained=int((a & ~b).sum()), lost=int((~a & b).sum()),
                    both_succeed=int((a & b).sum()), both_fail=int((~a & ~b).sum())))
    changes = []
    for h in (30, 150):
        for cat, name in enumerate(CATEGORIES):
            mask = reference['categories'] == cat
            a = trials['orthonormal'][800][f'hits_{h}'][mask] <= h
            b = trials['orthonormal'][400][f'hits_{h}'][mask] <= h
            changes.append(dict(horizon=h, category=name, gained=int((a & ~b).sum()), lost=int((~a & b).sum())))
    result = dict(protocol=dict(pairs=600, per_category=200, rollouts_per_pair=1,
        endpoint_seed=int(reference['trial_seed']), epsilon=.02, goal_tolerance=.1,
        doorway_scale=2, checkpoint_epoch=800, parameters='online F/B',
        bootstrap_repetitions=20000, bootstrap_seed=20261109), runs=NAMES,
        integrity=integrity, matched_pair_arrays=True, learner_source_identical=True,
        unchanged_source_functions=source_checks, final_comparisons=comparisons,
        orthonormal_400_to_800=changes, monitor_rows=rows)
    (OUT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    for i, h in enumerate((30, 150)):
        for j, cat in enumerate(CATEGORIES):
            ax = axes[i, j]
            for label in NAMES:
                values = [100*next(r['success_rate'] for r in rows[label][e]
                          if r['horizon'] == h and r['category'] == cat) for e in EPOCHS]
                ax.plot(EPOCHS, values, marker='o', label=label)
            ax.set(title=f'{cat.replace("_", " ")}, H={h}', ylim=(0, 105), xlabel='Epoch', ylabel='Success (%)')
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Matched monitor goals: 200 pairs/category, one noisy rollout/pair')
    fig.tight_layout()
    fig.savefig(OUT / 'learning_curves.png', dpi=160)
    plt.close(fig)
    report = [
        '# Replay continuing: orthonormal versus released B regularization', '',
        'The orthonormal run completed 800 epochs / 800,000 updates / 2.4 million transitions. '
        'All 800,000 stored losses are finite. It modestly improves final diagonal success on the saved monitor, '
        'but does not establish an overall navigation improvement over replay with the released regularizer.', '',
        '## Protocol and validation', '',
        'This analysis reads saved artifacts only; no training or policy rollouts were rerun. '
        'All three runs have identical endpoint arrays, category assignments, BFS lengths and rollout seed. '
        'There are 200 pairs per category and one noisy rollout per pair: online F/B, epsilon 0.02, '
        'coordinate-wise goal tolerance 0.1, doubled doorways, horizons 30 and 150. '
        'These are the training monitor endpoints (seed 271828), not the different endpoints used in the older 32-rollout reports.', '',
        'The two replay configurations differ only in b_regularization: orthonormal versus touati_continuous. '
        'Both use replay goals, the continuing objective, coefficient 1, training seed 0 and identical recorded '
        'Python/JAX/NumPy versions. Archived source hashes were verified. Learner source is byte-identical; '
        'AST comparisons confirm unchanged rollout/evaluation, movement, environment classes and compiled epoch function. '
        'Other archived changes include package import relocation, shared geometry/plotting helpers, threshold validation '
        'with the same default 0.1, and source archival paths. This is not a byte-identical whole-repository comparison.', '',
        '## Final checkpoint', '',
        '| Category | Horizon | Released replay | Orthonormal replay | Change (pp) | Paired 95% interval (pp) | Gained / lost pairs |',
        '| --- | ---: | ---: | ---: | ---: | --- | --- |',
    ]
    for r in comparisons:
        if r['comparator'] != 'released':
            continue
        lo, hi = r['paired_bootstrap95_pp']
        report.append(f"| {r['category']} | {r['horizon']} | {100*r['comparator_success_rate']:.1f}% | "
                      f"{100*r['orthonormal_success_rate']:.1f}% | {r['delta_pp']:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {r['gained']} / {r['lost']} |")
    report += ['', 'Intervals use 20,000 paired resamples of endpoint pairs. They describe uncertainty across these '
        'task/noise realizations, not training seeds. All six replay-versus-replay intervals include zero. '
        'The horizons share rollout randomness and should not be treated as independent replications.', '',
        '## Learning curves and wall collisions', '',
        '| Epoch | Orthonormal diagonal H=30 | Released diagonal H=30 | Orthonormal diagonal H=150 | Released diagonal H=150 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for e in EPOCHS:
        vals = [100*next(r['success_rate'] for r in rows[k][e] if r['horizon']==h and r['category']=='diagonal_rooms')
                for h in (30, 150) for k in ('orthonormal', 'released')]
        report.append(f'| {e} | ' + ' | '.join(f'{v:.1f}%' for v in vals) + ' |')
    report += ['', 'The orthonormal diagonal monitor improves from epoch 400 to 800 at both horizons, '
        'with no observed late diagonal deterioration at the saved checkpoints. It initially trails the released '
        'regularizer at epoch 200. Adjacent success peaks at epoch 600 (65% / 82%) and falls to 64% / 79.5% at epoch 800. '
        'Epoch 800 is the final comparison; no checkpoint was retrospectively selected.', '',
        '| Category | Horizon | Released wall actions | Orthonormal wall actions |',
        '| --- | ---: | ---: | ---: |']
    for h in (30, 150):
        for cat in ('adjacent_rooms', 'diagonal_rooms'):
            vals = [next(r['wall_action_percent'] for r in rows[k][800] if r['horizon']==h and r['category']==cat)
                    for k in ('released', 'orthonormal')]
            report.append(f'| {cat} | {h} | {vals[0]:.2f}% | {vals[1]:.2f}% |')
    report += ['', 'Wall-action percentages pool executed actions, not failed trials. '
        'The small diagonal collision reductions do not resolve wall trapping: 75.31% of diagonal H=150 actions '
        'are wall-rejected and 46.5% of diagonal trials fail. Saved aggregates do not identify repeated stationary '
        'collision streaks or establish the cause of failure.', '',
        '## Interpretation', '',
        '- Relative to released replay: diagonal success increases by 5.0 pp / 3.5 pp, adjacent success decreases '
        'by 2.5 pp / 1.5 pp, and same-room success is unchanged at 99.5% / 99.5%. '
        'Equal-category overall success rises by only 0.83 pp / 0.67 pp.',
        '- Relative to Cauchy continuing: diagonal success rises from 12% / 26.5% to 34% / 53.5%. '
        'This contrast changes both task sampling and regularization, so it cannot isolate the orthonormality effect.',
        '- Losses are finite but total loss values across the two regularizers are different objectives; '
        'their magnitudes cannot rank navigation quality. No feature-covariance diagnostic was run, '
        'so selecting orthonormal does not demonstrate that learned features are empirically orthonormal.',
        '- The supported conclusion is a modest diagonal point-estimate gain in one training seed, '
        'with mixed results across categories. A matched repeated-rollout evaluation would test rollout robustness; '
        'additional training seeds are needed to establish a reliable regularizer benefit.', '',
        '![Learning curves](learning_curves.png)', '',
        'Reproduce artifact analysis from the repository root:', '',
        '```bash', 'python experiments/fourrooms_doorways/analyses/replay_orthonormal_comparison/analyse.py', '```', '',
        'Machine-readable validation and comparisons: [results.json](results.json).', '',
        'New run: `runs/' + NAMES['orthonormal'] + '/`.', '',
    ]
    (OUT / 'REPORT.md').write_text('\n'.join(report))
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
