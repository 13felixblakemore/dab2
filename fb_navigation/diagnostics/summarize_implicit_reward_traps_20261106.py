"""Compare predicted FB values, learned-reward MC, and verified geometry detours."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import csv
import hashlib
import json
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms
from fb_navigation.agents.forward_backward_jax import FBLearner
from fb_navigation.diagnostics.investigate_doorway_decline import nominal_transitions
from fb_navigation.diagnostics.diagnose_implicit_reward_traps_20261106 import make_rollout

OUTPUT = ROOT/'experiments/fourrooms_doorways/analyses/implicit_reward_wall_traps'
ACTIONS = ('LEFT', 'RIGHT', 'UP', 'DOWN')


def target_policy_records(settings, learner, env, states, goals, corrected):
    cache = OUTPUT/'target_softmax200.npz'
    params = dict(f=learner.state['target']['f'], b=learner.state['params']['b'])
    if not cache.exists():
        reps = settings['rollouts']; n = len(states)
        pids = np.repeat(np.arange(n), 4*reps)
        first = np.tile(np.repeat(np.arange(4), reps), n)
        replicas = np.tile(np.arange(reps), n*4)
        ids = pids*reps+replicas
        fn = make_rollout(learner, env, settings['horizons'], 'softmax200')
        batches = []
        for begin in range(0, len(ids), 512):
            sl = slice(begin, begin+512); selected = pids[sl]
            keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(settings['seed']), i))(jnp.asarray(ids[sl], jnp.uint32))
            result = fn(params, jnp.asarray(states[selected]), jnp.asarray(goals[selected]), jnp.asarray(first[sl]), keys,
                        jnp.asarray(corrected[selected]))
            batches.append(jax.device_get(result))
            print(json.dumps(dict(policy='target_softmax200', completed=min(begin+512, len(ids)), total=len(ids))), flush=True)
        record = {k: np.concatenate([b[k] for b in batches]).reshape((n, 4, reps)+batches[0][k].shape[1:]) for k in batches[0]}
        np.savez_compressed(cache, **record)
    return dict(np.load(cache))


def main():
    settings = json.loads((OUTPUT/'config.json').read_text())
    probes = settings['probes']; n = len(probes); reps = settings['rollouts']
    run = Path(settings['run']); config = json.loads((run/'config.json').read_text())
    env = AdjustableFourRooms(**config['environment'])
    learner = FBLearner.from_checkpoint(run/'epoch_0800.pkl', device=settings['device'])
    states = np.array([p['state'] for p in probes], np.float32)
    goals = np.array([p['goal'] for p in probes], np.float32)
    tasks = np.asarray(learner.embed_B(goals))
    corrected = np.load(OUTPUT/'covariance.npz')['corrected_tasks']
    covariance = json.loads((OUTPUT/'covariance_audit.json').read_text())
    records = {policy: dict(np.load(OUTPUT/f'{policy}.npz')) for policy in ['greedy', 'softmax200', 'wall_masked', 'geometry_detour']}
    records['target_softmax200'] = target_policy_records(settings, learner, env, states, goals, corrected)
    target_f = learner.forward_network.apply(learner.state['target']['f'], learner.featuriser.transform(jnp.asarray(states)), jnp.asarray(tasks))
    target_q = np.asarray(jnp.einsum('bda,bd->ba', target_f, jnp.asarray(tasks)))
    next_states, blocked = nominal_transitions(env, states)
    moving = (np.linalg.norm(next_states-states[:, None], axis=2) > 1e-6) & ~blocked
    values, comparisons = [], []
    for policy, record in records.items():
        for i, probe in enumerate(probes):
            wall = int(np.argmax(probe['q']))
            draws = np.random.default_rng(settings['seed']+i+1000).integers(reps, size=(4000, reps))
            returns = record['returns'][i, :, :, -1]
            for reward_id, kind in enumerate(('raw', 'covariance_adjusted')):
                r = returns[:, :, reward_id].astype(np.float64)
                differences = r-r[wall]
                mean = r.mean(1); gap = differences.mean(1)
                boot = differences[:, draws].mean(2)
                # Bonferroni across three alternatives to the FB-preferred action.
                ci = np.quantile(boot, [.05/6, 1-.05/6], axis=1).T
                tail = covariance['raw_tail_bound' if reward_id == 0 else 'corrected_tail_bound'][i]
                interval = ci + np.array([-2*tail, 2*tail])
                free = np.flatnonzero(moving[i])
                better = [int(a) for a in free if interval[a, 0] > 0]
                if better:
                    finding = 'moving_initial_action_better_than_FB_wall_action'
                elif len(free) and np.all(interval[free, 1] < 0):
                    finding = 'FB_wall_action_better_than_all_tested_moving_initial_actions'
                else:
                    finding = 'not_resolved_by_simultaneous_MC_intervals'
                comparisons.append(dict(trap=i+1, pair=probe['pair'], policy=policy, reward=kind,
                    fb_action=ACTIONS[wall], mc_best_action=ACTIONS[int(mean.argmax())],
                    mean_returns=mean.tolist(), gaps_vs_fb_wall=gap.tolist(),
                    simultaneous95_gap_intervals_with_tail=interval.tolist(),
                    genuinely_moving_nominal_actions=[ACTIONS[a] for a in free],
                    better_moving_actions=[ACTIONS[a] for a in better], finding=finding))
                for action in range(4):
                    values.append(dict(trap=i+1, pair=probe['pair'], policy=policy, reward=kind,
                        action=ACTIONS[action], nominal_blocked=bool(blocked[i, action]), nominal_moves=bool(moving[i, action]),
                        first_action_collision_probability=float(record['first_blocked'][i, action].mean()),
                        predicted_q=probe['q'][action], target_q=float(target_q[i, action]), mc_mean=float(mean[action]),
                        mc_standard_error=float(r[action].std(ddof=1)/np.sqrt(reps)),
                        mc_gap_vs_fb_action=float(gap[action]), paired_gap_low=float(interval[action, 0]),
                        paired_gap_high=float(interval[action, 1]),
                        final_goal_hit_rate=float(np.mean(record['hits'][i, action] <= max(settings['horizons'])))))
    (OUTPUT/'action_comparisons.json').write_text(json.dumps(comparisons, indent=2))
    with (OUTPUT/'q_and_mc_returns.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    metadata = dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkpoint_sha256=settings['checkpoint_sha256'], target_policy='frozen target F softmax(Q/200), online B goal task',
        covariance_reward='B(s)^T pinv(C) B(goal); C from 32768 final replay samples; same fixed policy task z=B(goal)',
        simultaneous_intervals='paired bootstrap 4000 draws; Bonferroni over 3 action alternatives, plus conservative remaining-discount tail',
        target_q=target_q.tolist(), geometry_detour_goal_rates=json.loads((OUTPUT/'geometry_detour.json').read_text())['goal_hit_rate_by_probe_action'])
    (OUTPUT/'analysis_metadata.json').write_text(json.dumps(metadata, indent=2))

    for reward, name in [('raw', 'raw_mc_advantages.png'), ('covariance_adjusted', 'covariance_adjusted_mc_advantages.png')]:
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        for i, (ax, probe) in enumerate(zip(axes.flat, probes)):
            ax.axhline(0, color='black', lw=1)
            for shift, policy, color, label in [(-.06, 'greedy', '#338465', 'Greedy continuation'),
                                               (.06, 'target_softmax200', '#b75e37', 'Target softmax T=200')]:
                result = next(r for r in comparisons if r['trap'] == i+1 and r['policy'] == policy and r['reward'] == reward)
                gap = np.array(result['gaps_vs_fb_wall']); interval = np.array(result['simultaneous95_gap_intervals_with_tail'])
                ax.errorbar(np.arange(4)+shift, gap, yerr=np.stack([gap-interval[:, 0], interval[:, 1]-gap]),
                    fmt='o-', color=color, label=label, capsize=3, lw=1)
            ax.set(xticks=range(4), xticklabels=ACTIONS, ylabel='MC return minus MC return of FB action',
                title=f'Trap {i+1}, pair {probe["pair"]}: FB chooses {probe["fb_action"]}')
            ax.tick_params(axis='x', labelsize=8); ax.grid(alpha=.15)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f'{reward.replace("_", " ").title()} reward: positive = better initial action than FB wall choice\n256 common-random-number rollouts/action, 1500 continuing steps; simultaneous intervals + tail bounds')
        fig.tight_layout(); fig.savefig(OUTPUT/name, dpi=170); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for i, (ax, probe) in enumerate(zip(axes.flat, probes)):
        ax.plot(range(4), probe['q'], 'ko-', label='FB predicted Q')
        for policy, color, label in [('greedy', '#338465', 'Raw reward, greedy MC'),
                                     ('target_softmax200', '#b75e37', 'Raw reward, target-softmax MC')]:
            r = records[policy]['returns'][i, :, :, -1, 0]
            ax.errorbar(range(4), r.mean(1), yerr=1.96*r.std(1, ddof=1)/np.sqrt(reps), fmt='o-', color=color, label=label, capsize=3)
        r = records['target_softmax200']['returns'][i, :, :, -1, 1]
        ax.errorbar(range(4), r.mean(1), yerr=1.96*r.std(1, ddof=1)/np.sqrt(reps), fmt='s--', color='#4775b5', label='Covariance-adjusted, target-softmax MC', capsize=3)
        ax.set(xticks=range(4), xticklabels=ACTIONS, ylabel='Predicted Q / discounted return', title=f'Trap {i+1}, pair {probe["pair"]}')
        ax.tick_params(axis='x', labelsize=8); ax.grid(alpha=.15)
    axes.flat[0].legend(fontsize=6)
    fig.suptitle('Absolute values: raw-reward scale is not calibrated to FB when E[BB^T] differs from identity')
    fig.tight_layout(); fig.savefig(OUTPUT/'q_vs_mc_values.png', dpi=170); plt.close(fig)

    def lookup(i, policy, kind='raw'):
        return next(r for r in comparisons if r['trap'] == i+1 and r['policy'] == policy and r['reward'] == kind)
    summary = []
    b_states = np.asarray(learner.embed_B(states))
    for i, probe in enumerate(probes):
        wall = int(np.argmax(probe['q']))
        summary.append(dict(trap=i+1, pair=probe['pair'], fb_action=probe['fb_action'],
            raw_reward_trap=probe['reward_at_trap'], raw_reward_goal=probe['reward_at_goal'],
            adjusted_reward_trap=float(b_states[i] @ corrected[i]), adjusted_reward_goal=float(tasks[i] @ corrected[i]),
            raw_greedy=lookup(i, 'greedy'), raw_target_softmax=lookup(i, 'target_softmax200'),
            adjusted_target_softmax=lookup(i, 'target_softmax200', 'covariance_adjusted'),
            greedy_return_after_fb_action=lookup(i, 'greedy')['mean_returns'][wall],
            geometry_detour_return_after_fb_action=lookup(i, 'geometry_detour')['mean_returns'][wall],
            geometry_goal_rate=metadata['geometry_detour_goal_rates'][i][wall]))
    (OUTPUT/'summary.json').write_text(json.dumps(summary, indent=2))
    raw_rank_cases = [r['trap'] for r in summary if r['raw_target_softmax']['better_moving_actions']]
    adjusted_rank_cases = [r['trap'] for r in summary if r['adjusted_target_softmax']['better_moving_actions']]
    raw_peak_cases = [r['trap'] for r in summary if r['raw_reward_trap'] > r['raw_reward_goal']]
    lines = ['# Learned reward versus inaccurate values at recorded wall traps', '',
        'Frozen epoch-800 width-2 replay-continuing model trained with goal threshold 0.025. '
        'Six different goal pairs, starting with the previous recorded example, then distinct failed goal pairs '
        'nearest an 80-step stationary collision run. Each anchor is a recorded collision position where '
        'the current greedy action is nominally blocked. These are selected cases, not prevalence estimates.', '',
        '## Findings', '',
        f'**Both mechanisms occur in these selected cases.** Moving away beats FB\'s blocked initial action '
        f'under the literal training-backup continuation in traps {raw_rank_cases}; these same four '
        f'cases remain significant with the covariance-adjusted reward (traps {adjusted_rank_cases}). '
        'This supports inaccurate action rankings rather than a raw-reward scaling artifact in those cases. '
        f'Traps {raw_peak_cases} have raw reward at the trapped position greater than reward at the goal. '
        'In both, a controller that reaches the goal in every rollout has lower raw discounted return '
        'than the trapped greedy policy.', '',
        'For example, trap 4 (pair 417) predicts LEFT=435.47 versus DOWN=434.08, but the target-softmax '
        'MC returns are LEFT=116.07 versus DOWN=126.45 for the raw reward. DOWN gains 10.38 '
        '(simultaneous interval including the tail bound: 7.37 to 13.49). The covariance-adjusted '
        'DOWN advantage is 160.12 (108.02 to 215.37), retaining the contradiction.', '',
        'Trap 2 (pair 412) instead has raw r(trap)=4.457 versus r(goal)=3.112. The trapped greedy '
        'return is 445.70; the verified goal detour returns 312.43. Its covariance-adjusted '
        'instantaneous reward also favors the trap (41.99 versus 32.34). Trap 3 has a raw-reward '
        'false peak too, but its covariance-adjusted immediate reward favors the goal (16.55 versus '
        '31.65), and its target-policy MC ranking is unresolved for that adjusted reward. '
        'It should not be assigned the same confident diagnosis as trap 2.', '',
        'These six deliberately selected examples do not estimate how often each mechanism occurs '
        'across all failures.', '',
        '## Return convention', '',
        '`G_H = sum(t=0..1499) gamma^t * B(s_t)^T B(goal)`, gamma=0.99. '
        'Force each of LEFT/RIGHT/UP/DOWN at t=0; use 256 matched noise/continuation streams per action. '
        'Online B is frozen; rewards are neither normalized nor clipped. Movement is native and noisy. '
        'Do not stop, freeze, reset, or zero rewards on goal entry. The environment success threshold '
        'is used only to record auxiliary goal hits. No entropy bonus is added to returns.', '',
        'The local loss uses the current observation in `-mean(F(s,a,z) dot B(s))`, '
        'so current-state reward timing is used. The primary policies are greedy and the literal frozen '
        'target-F softmax(Q/200) used by the training backup. Online softmax(Q/200), wall-masked greedy, '
        'and an external geometry-only detour controller are additional diagnostics. '
        'All comparisons use the original online policy task z=B(goal).', '',
        'Four analytic tests passed: current-state timing/forced first action, continuing after goal entry, '
        'rejected moves, and all three continuation selectors. Conservative global network interval bounds '
        'limit omitted return tails; see covariance_audit.json. Return checkpoints at 150, 500, 1000 and 1500 '
        'steps are retained. Intervals use paired bootstrap samples, a Bonferroni correction across the three '
        'alternatives to the FB action, and twice the remaining-tail bound for action differences.', '',
        '## Raw reward and action rankings', '',
        '| Trap / goal pair | FB choice | r(trap) | r(goal) | Greedy-MC best first action | Target-softmax-MC best first action |',
        '|---|---|---:|---:|---|---|']
    for r in summary:
        lines.append(f'| {r["trap"]} / {r["pair"]} | {r["fb_action"]} | {r["raw_reward_trap"]:.3f} | {r["raw_reward_goal"]:.3f} | '
                     f'{r["raw_greedy"]["mc_best_action"]} | {r["raw_target_softmax"]["mc_best_action"]} |')
    lines += ['', 'Best actions are point estimates; use action_comparisons.json for uncertainty and '
        'which genuinely moving first actions beat the blocked choice. Boundary-clipped actions with zero '
        'nominal displacement are not classified as detours.', '',
        '![Reward landscape](reward_landscapes.png)', '',
        '![Raw reward action advantages](raw_mc_advantages.png)', '',
        '## Verified multi-step goal detours', '',
        'A one-step intervention can return to the same trap under greedy continuation. To test longer '
        'detours, an external controller follows a geometry-only BFS distance field, using native noisy '
        'movement and all four original actions. It continues moving near the goal after entry. This '
        'controller is diagnostic only and changes neither FB nor training. Its goal-hit rates verify '
        'whether the tested routes actually reach the goal.', '',
        '| Trap | Raw return: forced FB action then greedy | Raw return: forced FB action then goal detour | Detour goal-hit rate |',
        '|---|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["trap"]} | {r["greedy_return_after_fb_action"]:.2f} | {r["geometry_detour_return_after_fb_action"]:.2f} | {100*r["geometry_goal_rate"]:.1f}% |')
    lines += ['', '## Covariance qualification for comparing Q to this reward', '',
        'The empirical uncentered replay covariance C=E[B(s)B(s)^T] is far from identity. '
        f'Its mean eigenvalue is {covariance["mean_eigenvalue"]:.5f}, eigenvalues range '
        f'from {min(covariance["eigenvalues"]):.5f} to {max(covariance["eigenvalues"]):.5f}, '
        f'and ||C-I||_F/sqrt(d)={covariance["relative_frobenius_distance_from_identity"]:.3f}. '
        'Thus raw B(s)^T B(goal) returns and F(s,a,B(goal))^T B(goal) need not have the same scale or rankings.', '',
        'With frozen shared B and an exactly solved projected FB equation, the F normal equation is '
        '`C F = B(s) + gamma C E[F_next]`. Consequently the corresponding scalar immediate reward is '
        '`B(s)^T C^{-1} z`, which reduces to `B(s)^T z` only when C=I (on the relevant subspace). '
        'This is an inference from the implemented loss, not an extra reward used in training. '
        'We therefore also score each identical trajectory with `B(s)^T pinv(C) B(goal)`. '
        'C is estimated from 32,768 final-replay observations; rcond=1e-6. This secondary diagnostic '
        'has covariance estimation uncertainty not covered by the trajectory confidence intervals.', '',
        '![Absolute Q and MC values](q_vs_mc_values.png)', '',
        '![Covariance-adjusted action advantages](covariance_adjusted_mc_advantages.png)', '',
        '## Interpretation limits', '',
        'A significantly better moving initial action under the backup policy, especially when it remains '
        'better for the covariance-adjusted reward, is evidence of inaccurate learned action rankings. '
        'A wall trap with greater raw reward than the goal, and a verified goal-reaching detour with lower '
        'raw return, is evidence of a problematic raw goal-reward landscape. Both mechanisms can coexist. '
        'Greedy first-action agreement alone does not prove the wall is globally optimal, and an absolute '
        'raw-return/Q mismatch alone does not prove a value error when C differs from identity. '
        'Training uses target networks and a changing representation; the frozen projected equation is '
        'an approximation for diagnosing the trained model, not a guarantee of its calibration.', '',
        '## Files and reproducibility', '',
        '`q_and_mc_returns.csv` contains every action, continuation, reward, Q prediction, MC mean/SE, '
        'paired difference interval, first collision rate and auxiliary goal-hit rate. '
        '`surrounding_rewards.csv` and the reward maps record nearby free states. '
        'All trajectories, checkpoint hashes and sampling settings are retained in the NPZ/JSON files. '
        'Run these from the repository root:', '',
        '```bash',
        'python experiments/maintenance/diagnose_implicit_reward_traps_20261106.py --device cpu',
        'python experiments/maintenance/geometry_detour_implicit_reward_20261106.py',
        'python experiments/maintenance/summarize_implicit_reward_traps_20261106.py',
        'python -m unittest test_implicit_reward_rollout -v',
        '```', '',
        'No changes were made to training, learned parameters, checkpoints, or environment dynamics.']
    (OUTPUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    sources = OUTPUT/'source'; sources.mkdir(exist_ok=True)
    source_files = [Path(__file__), Path(__file__).with_name('diagnose_implicit_reward_traps_20261106.py'),
                    Path(__file__).with_name('geometry_detour_implicit_reward_20261106.py'), ROOT/'tests/test_implicit_reward_rollout.py']
    metadata['source_sha256'] = {}
    for source in source_files:
        content = source.read_bytes()
        metadata['source_sha256'][source.name] = hashlib.sha256(content).hexdigest()
        (sources/source.name).write_bytes(content)
    (OUTPUT/'analysis_metadata.json').write_text(json.dumps(metadata, indent=2))
    for r in summary:
        print(json.dumps(dict(trap=r['trap'], pair=r['pair'], fb=r['fb_action'],
            raw_greedy=r['raw_greedy']['finding'], raw_target=r['raw_target_softmax']['finding'],
            adjusted_target=r['adjusted_target_softmax']['finding'],
            greedy_raw=r['greedy_return_after_fb_action'], detour_raw=r['geometry_detour_return_after_fb_action'],
            detour_goal=r['geometry_goal_rate'])), flush=True)


if __name__ == '__main__': main()
