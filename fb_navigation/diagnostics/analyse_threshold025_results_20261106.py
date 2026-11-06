"""Summarize matched strict-threshold evaluation and diagonal wall traps."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms
from fb_navigation.evaluation.fourrooms_reference_diagnostics import evaluate_checkpoint
from fb_navigation.agents.forward_backward_jax import FBLearner

EXPERIMENT = ROOT / 'experiments/fourrooms_doorways'
OUTPUT = EXPERIMENT / 'analyses/replay_threshold025_comparison'
MODELS = ('width2_replay_continuing_seed0', 'width2_replay_continuing_threshold025_seed0')
CATEGORIES = ('same_room', 'adjacent_rooms', 'diagonal_rooms')
LABELS = ('Old: training tolerance 0.1', 'New: training tolerance 0.025')
COLORS = ('#557ca4', '#d18435')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    settings = json.loads((OUTPUT / 'config.json').read_text())
    assert settings['goal_threshold'] == .025 and settings['models'] == list(MODELS)
    pairs = dict(np.load(OUTPUT / 'pairs.npz'))
    rows = json.loads((OUTPUT / 'results.json').read_text())
    records = [dict(np.load(OUTPUT / f'{name}_epsilon002.npz')) for name in MODELS]
    comparisons = []
    for horizon, column in ((30, 0), (150, 1)):
        for category, name in enumerate(CATEGORIES):
            chosen = pairs['categories'] == category
            draws = np.random.default_rng(20261109).integers(chosen.sum(), size=(5000, chosen.sum()))
            success = [(r['hits'][chosen] <= horizon).mean(1) for r in records]
            delta = success[1] - success[0]
            result = dict(horizon=horizon, category=name,
                success_delta_pp=float(100 * delta.mean()),
                success_delta_interval95_pp=(100 * np.quantile(delta[draws].mean(1), [.025, .975])).tolist())
            actions = [np.minimum(r['hits'][chosen], horizon).sum(1) for r in records]
            for metric, key in [('wall_action', 'collisions'), ('repeated_wall_action', 'repeats')]:
                counts = [r[key][chosen, :, column].sum(1) for r in records]
                estimates = [100 * c[draws].sum(1) / a[draws].sum(1) for c, a in zip(counts, actions)]
                result[f'{metric}_delta_pp'] = float(100 * (counts[1].sum()/actions[1].sum() - counts[0].sum()/actions[0].sum()))
                result[f'{metric}_delta_interval95_pp'] = np.quantile(estimates[1]-estimates[0], [.025, .975]).tolist()
            comparisons.append(result)
    (OUTPUT / 'paired_differences.json').write_text(json.dumps(comparisons, indent=2))

    def select(model, horizon, category):
        return next(r for r in rows if r['model'] == model and r['policy'] == 'epsilon002'
                    and r['horizon'] == horizon and r['category'] == category)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, horizon in enumerate((30, 150)):
        for i, model in enumerate(MODELS):
            selected = [select(model, horizon, category) for category in CATEGORIES]
            x = np.arange(3) + (i-.5)*.35
            success = np.array([100*r['success_rate'] for r in selected])
            intervals = np.array([r['success_interval95'] for r in selected])*100
            bars = axes[row, 0].bar(x, success, .35, label=LABELS[i], color=COLORS[i],
                yerr=np.stack([success-intervals[:, 0], intervals[:, 1]-success]), capsize=3)
            axes[row, 0].bar_label(bars, fmt='%.1f', padding=4, fontsize=8)
            bars = axes[row, 1].bar(x, [r['wall_action_percent'] for r in selected], .35, color=COLORS[i])
            axes[row, 1].bar_label(bars, fmt='%.1f', padding=4, fontsize=8)
        for ax, metric in zip(axes[row], ('Goal success (%)', 'Rejected wall actions (%)')):
            ax.set(xticks=range(3), xticklabels=['Same room', 'Adjacent', 'Diagonal'], ylim=(0, 100),
                   ylabel=metric, title=f'{horizon}-step horizon')
            ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Both models evaluated at tolerance 0.025, epsilon 0.02\n200 pairs/category, 32 matched rollouts/pair')
    fig.tight_layout(); fig.savefig(OUTPUT / 'goal_and_wall_comparison.png', dpi=170); plt.close(fig)

    traps = [json.loads((EXPERIMENT / 'analyses' / folder / 'summary.json').read_text())
             for folder in ('replay_threshold025_old_failures', 'replay_threshold025_failures')]
    assert all(t['goal_threshold'] == .025 and t['exact_mc_replay'] for t in traps)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, summary in enumerate(traps):
        stats = [summary['horizons'][str(h)] for h in (30, 150)]
        x = np.arange(2) + (i-.5)*.35
        bars = axes[0].bar(x, [100*s['failed_with_10_stationary_hits']/s['failures'] for s in stats], .35,
                          color=COLORS[i], label=LABELS[i])
        axes[0].bar_label(bars, fmt='%.1f', padding=3, fontsize=8)
        bars = axes[1].bar(x, [s['median_longest_stationary_run'] for s in stats], .35, color=COLORS[i])
        axes[1].bar_label(bars, fmt='%.0f', padding=3, fontsize=8)
    axes[0].set(ylabel='Failed rollouts (%)', ylim=(0, 100), title='At least 10 consecutive stationary wall hits')
    axes[1].set(ylabel='Steps', ylim=(0, 110), title='Median longest stationary wall-hit run')
    for ax in axes: ax.set_xticks([0, 1], ['30-step horizon', '150-step horizon']); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    axes[0].legend(fontsize=7)
    fig.suptitle('Diagonal failures: both models tested at tolerance 0.025')
    fig.tight_layout(); fig.savefig(OUTPUT / 'wall_trap_comparison.png', dpi=170); plt.close(fig)

    env = AdjustableFourRooms(doorway_scale=2, include_stay_action=False, threshold_distance=.025)
    monitor_pairs = dict(np.load(EXPERIMENT / 'runs' / MODELS[1] / 'evaluation_pairs.npz'))
    monitor_folder = OUTPUT / 'old_strict_monitor'; monitor_folder.mkdir(exist_ok=True)
    new_monitor_folder = OUTPUT / 'new_strict_monitor'; new_monitor_folder.mkdir(exist_ok=True)
    old_history, new_history = [], []
    for epoch in (200, 400, 600, 800):
        path = monitor_folder / f'epoch_{epoch:04d}.json'
        if path.exists():
            old_rows = json.loads(path.read_text())
        else:
            learner = FBLearner.from_checkpoint(EXPERIMENT / 'runs' / MODELS[0] / f'epoch_{epoch:04d}.pkl', device=args.device)
            old_rows = evaluate_checkpoint(learner, env, monitor_pairs, monitor_folder, epoch)
        old_history.extend(old_rows)
        new_path = new_monitor_folder / f'epoch_{epoch:04d}.json'
        if new_path.exists():
            new_rows = json.loads(new_path.read_text())
        else:
            learner = FBLearner.from_checkpoint(EXPERIMENT / 'runs' / MODELS[1] / f'epoch_{epoch:04d}.pkl', device=args.device)
            new_rows = evaluate_checkpoint(learner, env, monitor_pairs, new_monitor_folder, epoch)
        new_history.extend(new_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, history in enumerate((old_history, new_history)):
        for ax, horizon in zip(axes, (30, 150)):
            chosen = [r for r in history if r['category'] == 'diagonal_rooms' and r['horizon'] == horizon]
            ax.plot([r['epoch'] for r in chosen], [100*r['success_rate'] for r in chosen], 'o-',
                    label=LABELS[i], color=COLORS[i])
            ax.set(xlabel='Epoch', ylabel='Diagonal success (%)', ylim=(0, 100), title=f'{horizon} steps')
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=8); fig.suptitle('Matched checkpoint monitor at tolerance 0.025: 200 diagonal pairs, one rollout/pair')
    fig.tight_layout(); fig.savefig(OUTPUT / 'diagonal_learning_curves.png', dpi=170); plt.close(fig)
    (OUTPUT / 'strict_monitor.json').write_text(json.dumps(dict(old=old_history, new=new_history), indent=2))

    lines = ['# Smaller-threshold experiment: matched comparison', '',
        'Both 800-epoch models were evaluated at coordinate-wise tolerance **0.025**, online F/B, '
        'z=B(goal), epsilon 0.02, 200 start-goal pairs per category and 32 matched stochastic rollouts per pair. '
        'One training seed (0) per condition. All 800,000 losses are finite in each run; learner and RBF '
        'settings match, geometry/actions/noise match, and both collected 2.4 million transitions. '
        'Archived source files match except the adjustable environment parameter that exposes goal tolerance. '
        'See validation.json for checkpoint hashes and source parity.', '',
        'The new model modestly improves diagonal point estimates and reduces diagonal wall actions. '
        'Same-room and adjacent-room goal success decrease, so goal reaching does not improve overall. '
        'Repeated wall trapping remains common.', '',
        '| Horizon | Task | Old success | New success | New - old (pp; paired 95% interval) | Old wall actions | New wall actions |',
        '|---|---|---:|---:|---:|---:|---:|']
    for result in comparisons:
        h, category = result['horizon'], result['category']
        a, b = [select(model, h, category) for model in MODELS]
        lo, hi = result['success_delta_interval95_pp']
        lines.append(f'| {h} | {category} | {100*a["success_rate"]:.2f}% | {100*b["success_rate"]:.2f}% | '
                     f'{result["success_delta_pp"]:+.2f} [{lo:+.2f}, {hi:+.2f}] | {a["wall_action_percent"]:.2f}% | {b["wall_action_percent"]:.2f}% |')
    lines += ['', 'Intervals resample the 200 goal pairs with all 32 matched rollouts kept together '
        '(5,000 paired bootstrap samples). They describe uncertainty across these tasks; they do not '
        'measure variation across training seeds. Full paired wall/repeated-wall differences are in paired_differences.json.', '',
        '![Goal success and wall actions](goal_and_wall_comparison.png)', '',
        '| Failed diagonal rollouts | Old, 30 steps | New, 30 steps | Old, 150 steps | New, 150 steps |',
        '|---|---:|---:|---:|---:|']
    for label, key, percent in [('At least 10 consecutive wall hits at the same exact position', 'failed_with_10_stationary_hits', True),
                                ('Median longest stationary wall-hit run (steps)', 'median_longest_stationary_run', False)]:
        values = []
        for horizon in (30, 150):
            for t in traps:
                s = t['horizons'][str(horizon)]
                values.append(f'{100*s[key]/s["failures"]:.2f}%' if percent else str(s[key]))
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    example = traps[1]['recorded_example']
    lines += ['', 'Failure-conditioned percentages compare different sets of failed trajectories. '
        'The all-action collision percentages above provide a common-policy comparison.', '',
        '![Wall traps](wall_trap_comparison.png)', '',
        f'The new-model example (pair {example["pair"]}, replica {example["replica"]}) crosses one doorway, '
        f'then repeatedly selects LEFT into the central vertical wall. {example["collisions"]}/150 rejected moves; '
        f'{example["longest_same_action_collision_run"]} consecutive rejected actions at one exact position. '
        'The recording was selected at the median stationary wall-hit run among failures that completed a room crossing; '
        'it is an illustrative conditional example, not a random draw.', '',
        '![New-model diagonal failure](../replay_threshold025_failures/recorded_failure.png)', '',
        '[Animated trajectory](../replay_threshold025_failures/recorded_failure.gif) · '
        '[Collision heatmaps](../replay_threshold025_failures/collision_hotspots.png) · '
        '[Six trajectories](../replay_threshold025_failures/failure_trajectories.png)', '',
        '![Diagonal checkpoint curves](diagonal_learning_curves.png)', '',
        'Checkpoint curves re-evaluate both models on the same local backend with the original monitor seed '
        '271828 and one rollout per pair. '
        'The final comparison above uses the separate evaluation seed 20261107 and 32 rollouts per pair. '
        'The new model improves throughout its retained checkpoints on this monitor. '
        'Do not compare strict results directly with the earlier scores evaluated at tolerance 0.1.', '',
        'The continuing objective is unchanged. Goal tolerance affects collection resets and stopping criteria; '
        'it introduces no direct wall-action penalty. These results show that shrinking the acceptance region '
        'does not resolve the learned preference for blocked actions. No training or hyperparameter changes '
        'were made during analysis.', '',
        'Reproduce the matched evaluation with evaluate_width2_goals.py as documented in THRESHOLD025.md. '
        'Record both models using the strict comparison directory; then run '
        '`python experiments/maintenance/analyse_threshold025_results_20261106.py`.']
    (OUTPUT / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == '__main__':
    main()
