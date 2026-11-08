# Experiments and results

This is the results overview for the experiments retained in this repository.
Numbers below come from saved evaluation JSON files and analysis reports; no
training or evaluation was rerun to produce this summary.

The strongest observed width-2 result is **replay-goal sampling with the continuing
FB objective**. It improves cross-room navigation over Cauchy task sampling on the
matched evaluation. Adding the tested stopped-bootstrap modification performs
poorly. Reducing goal tolerance from 0.1 to 0.025 addresses an overly permissive
success region, but does not resolve wall trapping. Selected wall-trap studies
find evidence of both inaccurate action rankings and problematic learned rewards.

These are **single-training-seed findings**, not established rankings across
training seeds. Repeated rollouts measure task/noise variability for fixed models.

## Completed training runs

All five retained full runs completed **800 epochs, 800,000 gradient updates and
2.4 million collected transitions**. Each uses training seed 0. The shared setup
uses four movement actions, step 0.05, movement noise 0.005, RBF features on a
21-by-21 grid, embedding dimension 50, learning rate 0.0005, discount 0.99,
temperature 200, Polyak coefficient 0.95, the released B regularizer, no gradient
clipping and no zero-task mixing. Collection uses uniformly random actions.

| Saved run | Doorway scale | Task sampling | FB objective | Coordinate-wise environment goal tolerance |
| --- | ---: | --- | --- | ---: |
| [Original-width reference](fourrooms_reference/runs/seed0/config.json) | 1 | Cauchy | Continuing | 0.1 |
| [Width-2 Cauchy](fourrooms_doorways/runs/width2_seed0/config.json) | 2 | Cauchy | Continuing | 0.1 |
| [Width-2 replay continuing](fourrooms_doorways/runs/width2_replay_continuing_seed0/config.json) | 2 | Independent replay goals, z = B(goal) | Continuing | 0.1 |
| [Width-2 replay stopped](fourrooms_doorways/runs/width2_replay_stopped_seed0/config.json) | 2 | Independent replay goals, z = B(goal) | Bootstrap stops on next-state goal entry | 0.1 |
| [Width-2 replay continuing, strict tolerance](fourrooms_doorways/runs/width2_replay_continuing_threshold025_seed0/config.json) | 2 | Independent replay goals, z = B(goal) | Continuing | 0.025 |

Scale 1 has doorway width 1/11; scale 2 doubles all four openings to 2/11.
The width-2 stopped run explicitly uses the square goal predicate (`goal_norm=linf`).
It should not be confused with older standalone stopped-measure studies using a
Euclidean goal disk.

## 1. Does widening the doorways help?

The original-width and doubled-width Cauchy models were evaluated on the same
600 monitor endpoint pairs: 200 same-room, 200 adjacent-room and 200 diagonal-room
pairs, with one noisy rollout per pair. Online F/B, epsilon 0.02 and goal tolerance
0.1 were used. H is the maximum number of actions.

| Model, epoch 800 | Same H=30 | Adjacent H=30 | Diagonal H=30 | Same H=150 | Adjacent H=150 | Diagonal H=150 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original-width Cauchy | 90.5% | 36.5% | 9.0% | 95.5% | 54.5% | 20.0% |
| Width-2 Cauchy | 94.0% | 47.5% | 12.0% | 96.5% | 64.0% | 26.5% |

Wider openings improve the observed scores, but cross-room success remains low.
The original run used a different backend/JAX version from the width-2 run, so
this comparison does not isolate doorway geometry as the sole cause.

Source: [width comparison report](fourrooms_doorways/RESULTS.md), with final
monitor JSON for [original width](fourrooms_reference/runs/seed0/evaluation/epoch_0800.json)
and [width 2](fourrooms_doorways/runs/width2_seed0/evaluation/epoch_0800.json).

## 2. Task sampling and bootstrapping: evaluation tolerance 0.1

This separate final-checkpoint evaluation uses **600 matched pairs and 32 noisy
rollouts per pair**, with endpoint seed 20261107, online F/B, epsilon 0.02, native
movement and coordinate-wise tolerance 0.1. These are different tasks and more
repetitions than the monitor table above; their absolute scores are not directly
interchangeable with that table.

This section asks whether replay tasks help compared with Cauchy tasks, and what
happens when the replay model also uses stopped bootstrapping. Every model here
is trained and evaluated with environment tolerance **0.1**. Section 5 asks a
different question: whether changing training tolerance helps when both models
are evaluated with the stricter **0.025** success rule.

| Model/policy | Same H=30 | Adjacent H=30 | Diagonal H=30 | Same H=150 | Adjacent H=150 | Diagonal H=150 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cauchy continuing | 94.89% | 57.36% | 21.78% | 96.83% | 72.16% | 35.47% |
| Replay continuing | **99.06%** | **72.14%** | **36.83%** | **99.67%** | **84.19%** | **55.39%** |
| Replay stopped | 13.33% | 3.45% | 1.83% | 18.03% | 7.52% | 4.41% |
| Random actions | 33.56% | 3.06% | 0.03% | 65.41% | 15.16% | 2.08% |

Replay continuing gains **15.05 percentage points** of diagonal success at H=30
and **19.92 points** at H=150 over Cauchy. Pair-bootstrap 95% intervals are
[10.05, 20.14] and [14.06, 25.63] points respectively; these do not measure
training-seed uncertainty. Its configuration differs from the width-2 Cauchy run
only in task sampling. The replay-stopped result therefore cannot be explained
by replay sampling alone, although the exact cause of its failure is unresolved.

Replay continuing still hits walls frequently: diagonal wall actions are 46.73%
at H=30 and 73.55% at H=150. At H=150 its diagonal success is 55.39% with epsilon
exploration, versus 39.94% when greedy. Occasional random actions appear useful
for escaping some traps, but do not solve navigation.

Sources: [three-model report](fourrooms_doorways/analyses/replay_continuing_comparison/REPORT.md),
[numerical results](fourrooms_doorways/analyses/replay_continuing_comparison/results.json),
and [original two-model/random study](fourrooms_doorways/analyses/goal_reaching/REPORT.md).

## 3. Learning curves and late deterioration

On the fixed monitor suite, width-2 Cauchy diagonal H=30 success is 12.5%, 24.0%,
17.0% and 12.0% at epochs 200, 400, 600 and 800. The 400-to-800 decline includes
38 lost successes and 14 newly successful tasks. A repeated 32-rollout evaluation
on the original diagonal monitor pairs also finds deterioration: **23.78% to
15.08%** at H=30 and **43.59% to 29.80%** at H=150.

Replay continuing does not show the same late decline in this seed. On its
matched repeated evaluation, diagonal success improves from **31.91% to 36.83%**
at H=30 and **48.38% to 55.39%** at H=150 between epochs 400 and 800.

The Cauchy diagnostics refine where the decline occurs:

| Study | Main finding |
| --- | --- |
| [Trajectory and value analysis](fourrooms_doorways/analyses/decline_analysis/REPORT.md) | Lost successes develop blocked-action preferences and later wall loops. Ranking changes track deterioration better than global Q scale or normalized bootstrap residuals. |
| [Fixed doorway probes](fourrooms_doorways/ADVANTAGE_STABILITY.md) | On 160 independently defined local probes, diagonal H=30 success stays approximately 25.5% to 25.8% from epoch 400 to 800. The decline is not uniform across doorway states. |
| [Matched navigation stages](fourrooms_doorways/analyses/stage_analysis/REPORT.md) | On 203 fixed post-crossing probes, fresh-30 success falls from 46.5% to 32.6%. Neither model initially chooses a blocked action there; later continuation and wall encounters matter. |

The evidence supports a route/state-dependent policy deficit. It does not
establish general representation collapse, exploding Bellman error or a specific
optimizer fault. Epoch 800 remains the final reported checkpoint; the earlier
peak is a learning-curve observation, not a retrospectively selected final score.

## 4. Are invalid goals or insufficient time causing the failures?

The [geometry and reachability audit](fourrooms_doorways/GOAL_VALIDITY.md) found
no wall/out-of-world endpoints among the 600 monitor pairs, 160 doorway probes
or 759 stage probes. All 79,822 inspected active trajectory positions were valid.
The 352-node free-space lattice is connected, and all 1,276 directed graph edges
agree with native noiseless movement.

One original diagonal pair has a 31-step lattice BFS reference; 28 stage probes
exceed their remaining original reference budget. Removing reference-budget-limited
cases preserves the performance gap. A noiseless lattice path is a diagnostic
reference, not an exact optimal noisy continuous trajectory.

All goal centers are valid, but **252/600 monitor success squares overlap a wall**
at tolerance 0.1. That is an acceptance-region issue, rather than invalid sampling.
An audit of 6,400 replay-continuing diagonal trajectories found wall intersection
between the stopping position and goal in 42/2,357 H=30 successes and 62/3,545
H=150 successes. These include corner occlusion, not just opposite-full-wall cases.
See [the threshold rationale](fourrooms_doorways/THRESHOLD025.md).

## 5. Training tolerance comparison: evaluation tolerance 0.025

The strict run changes the collection environment tolerance from 0.1 to 0.025;
the FB loss remains continuing. This changes collection resets and replay data,
as well as evaluation stopping. Both final models below are evaluated at the
**same strict tolerance 0.025**, on 200 pairs per category and 32 matched rollouts.

The first row below is the **same replay-continuing checkpoint as section 2**,
evaluated against a smaller goal region. Its diagonal H=150 success is therefore
45.36% here, compared with 55.39% under section 2's looser rule. The second row
is a separately trained strict-tolerance checkpoint. Both rows use replay tasks
and a continuing objective; this table does not compare replay against Cauchy.

| Model | Trained with tolerance | Evaluated with tolerance | Same H=30 | Adjacent H=30 | Diagonal H=30 | Same H=150 | Adjacent H=150 | Diagonal H=150 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original replay continuing | 0.1 | **0.025** | 78.47% | 57.12% | 25.20% | 84.56% | 71.81% | 45.36% |
| Strict replay continuing | 0.025 | **0.025** | 76.81% | 53.50% | 26.62% | 83.20% | 68.11% | 48.14% |

Diagonal point estimates increase by 1.42 and 2.78 percentage points, but paired
95% intervals include zero: [-2.78, 5.55] and [-2.03, 7.64]. Same-room and adjacent
success decrease. Diagonal wall-action fractions fall from 45.12% to 41.89% at
H=30 and 69.68% to 66.46% at H=150. There is no clear overall navigation gain.

The two findings are compatible: **replay sampling helps relative to Cauchy at
tolerance 0.1; training with a smaller tolerance does not clearly improve overall
performance when tested at tolerance 0.025.**

Stationary wall trapping remains common. Among strict-model diagonal failures,
65.89% at H=30 and 84.21% at H=150 contain at least ten consecutive rejected moves
at one exact position. These percentages condition on failure and are not the
fraction of all trajectories that fail. A smaller goal region introduces no
direct penalty for selecting a blocked action.

Sources: [strict comparison](fourrooms_doorways/analyses/replay_threshold025_comparison/REPORT.md),
[results JSON](fourrooms_doorways/analyses/replay_threshold025_comparison/results.json),
and [training/source validation](fourrooms_doorways/analyses/replay_threshold025_comparison/validation.json).

## 6. What causes the remaining wall traps?

The [original replay-continuing recordings](fourrooms_doorways/analyses/replay_continuing_failures/REPORT.md)
show repeated rejection at the same exact position across many different goals.
Among failed diagonal trials at tolerance 0.1, 83.48% at H=30 and 90.12% at H=150
have stationary wall-hit streaks of at least ten steps.

The subsequent [learned-reward study](fourrooms_doorways/analyses/implicit_reward_wall_traps/REPORT.md)
examines six deliberately selected traps in the strict model. It uses 256 matched
rollouts per action, horizons up to 1,500, discount 0.99, current-state learned
rewards and continuing returns that do not stop at goal entry.

- In four cases, a moving first action beats FB's blocked choice under the frozen
  target-softmax backup policy. The contradiction survives covariance-adjusted
  reward scoring, supporting inaccurate action rankings in those cases.
- Two traps have higher raw learned reward at the trapped position than at the
  goal. Verified goal-reaching detours have lower raw return than remaining
  trapped. The covariance-adjusted diagnosis is less conclusive for one of them.
- All six external geometry-only detour controllers reach their goals in every
  tested rollout. These are diagnostic controllers, not trained FB improvements.

The raw B(state)-dot-B(goal) return is not automatically calibrated to Q because
the empirical backward-feature covariance differs substantially from identity.
Absolute Q/return scale mismatch alone is insufficient evidence of value error.
These selected cases establish possible mechanisms, not their prevalence across
all failures. [Per-action results](fourrooms_doorways/analyses/implicit_reward_wall_traps/q_and_mc_returns.csv)
and [comparison intervals](fourrooms_doorways/analyses/implicit_reward_wall_traps/action_comparisons.json)
are retained alongside reward maps and trajectories.

## Earlier studies and work without a retained full result set

These studies are separate from the five-run comparison above. Their surviving
notes document results or methods, but the historical `evaluation_*` output
directories and Continuous Maze experiment directory are absent from this checkout.

| Study | What can be concluded from the retained material |
| --- | --- |
| [Goal-terminated measure smoke study](../docs/GOAL_TERMINATED_FB.md) | A 3,000-update small model ran with finite losses, but FB stopped masses were 2.94/3.37/3.28 versus MC 12.11/98.71/95.22 on three tasks. This validated execution, not convergence. The next-state-only mask leaves a start-inside-goal boundary limitation. |
| [FB bottleneck-density evaluation](../docs/FB_BOTTLENECK_EVALUATION.md) | Defines matched FB/MC estimators, exclusions and validity checks. Density ratios are finite-region hitting-probability proxies; no retained aggregate result here establishes calibration or reliable doorway discovery. |
| [Temperature and graph-policy study](../docs/TEMPERATURE_DIAGNOSIS.md) | On an older stopped checkpoint, target-softmax T=200 was nearly uniform. On a deterministic graph with a 10,000-step cap, success was 95.7% at T=200, 97.3% at T=100 and 2.1% when greedy. Colder policies became trapped. These are historical graph results, not native-dynamics scores or a retraining recommendation. |
| BFS, discount and epsilon-policy controls | Reusable tools remain in `diagnostics.control_bfs_bottlenecks` and `diagnostics.bottleneck_policies`; their old standalone output directories are not retained. Do not infer new results from tool availability. |
| Continuous Maze reproduction | Environment, runner and diagnostics remain in the package, but the original experiment config, reference-code bundle and run outputs were removed. No numerical reproduction claim is made from the current files. |
| Learning-rate / Polyak / dimension sweep | A sweep runner remains, but no complete sweep result directory is retained. There is no supported hyperparameter ranking to report. |
| Standalone PyTorch/JAX checkpoints | Four checkpoints remain in `../models/`. They are distinct from the five named runs above; the current tree has no corresponding complete comparative evaluation archive. |

## Files, interpretation and reproduction

Each full run keeps its resolved config, training records, evaluation arrays,
checkpoints and archived source snapshots. The [doorway artifact index](fourrooms_doorways/INDEX.md)
locates the analysis folders. Smoke-run outputs, redundant import-conflict archives
and dated one-off report generators were removed during cleanup. Historical
reports retain their original command descriptions; some no longer run unchanged.

Current commands are documented in [the evaluation guide](../docs/EVALUATION_TOOLS.md).
For example, rerun the three-model comparison into a **fresh** directory:

```bash
python -m fb_navigation.evaluation.fourrooms width2 \
  --models width2_seed0 width2_replay_continuing_seed0 width2_replay_stopped_seed0 \
  --pairs-per-category 200 --rollouts 32 --seed 20261107 \
  --output experiments/fourrooms_doorways/analyses/NEW_COMPARISON
```

For a strict comparison, add `--goal-threshold 0.025` and select
`width2_replay_continuing_seed0` and `width2_replay_continuing_threshold025_seed0`.
The standalone `fourrooms reference` command assumes original-width geometry;
use `fourrooms width2` for doubled-doorway models. New code/source hashes can
require fresh analysis outputs, and resuming old training requires its matching
archived sources.

Compare models within a common protocol. Goal tolerance, goal seed, noisy rollout
count, policy, horizon and dynamics all change the estimand. Successful path lengths
exclude failures; collision rates pool executed actions; failure-conditioned
statistics use different cohorts. None substitutes for overall success. Additional
training seeds and genuinely held-out tasks are needed before generalizing these
findings or claiming that one intervention caused an improvement.
