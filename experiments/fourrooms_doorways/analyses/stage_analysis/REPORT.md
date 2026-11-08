# Matched navigation-stage comparison: checkpoints 400 and 800

Subsequent goal-validity audit: all saved goals/states are outside walls and
geometrically connected. All stage probes fit the fresh 30-step BFS reference;
28 exceed their remaining original reference budget. Excluding those leaves the
post-crossing remaining-budget gap at 42.5% -> 30.1%. See `../../GOAL_VALIDITY.md`.

The stage-conditioned study identifies a strong continuation-policy deficit on
the actual diagonal task contexts, especially after the first crossing. A good
initial action at a doorway does not ensure the later route is executed well.
No checkpoint or training configuration was changed.

## Primary measurements

Extracted fixed state/goal/time-budget probes from **both** original checkpoint
trajectories, then evaluated **both models on every saved probe**. Each probe
uses 32 paired noisy rollouts, online B and F, epsilon=.02 including the first
action, native movement and goal detection. All goals are the original diagonal
goals, not resampled goals near a generic door.

The pooled table uses a fresh 30-step budget from each fixed state:

| Stage | Fixed probes / distinct original tasks | Top-action flips | Initially blocked greedy, 400 -> 800 | Success, 400 -> 800 | Wall-hit actions over rollout, 400 -> 800 |
| --- | --- | ---: | --- | --- | --- |
| First doorway approach | 354 / 190 | 25.7% | 9.0% -> 9.3% | 28.0% -> 18.4% | 56.2% -> 61.0% |
| Immediately after first crossing | 203 / 128 | 48.3% | 0% -> 0% | 46.5% -> 32.6% | 47.9% -> 57.6% |
| Second-door traversal attempt | 202 / 128 | 33.7% | 2.0% -> 6.4% | 49.0% -> 37.9% | 55.1% -> 62.4% |

Complete four-action rank order changes on 61.9%, 60.1% and 60.4% of the three
stage probe sets, respectively. Raw Q and all four signed action advantages are
in `action_rankings.csv`.

The strongest preferred-action churn is immediately after crossing, but **none
of those initial greedy actions hits a wall in either model**. The decline is
therefore not explained by immediate wall-hit choices at that instant. Indeed,
under the coarse noiseless BFS progress proxy, immediate post-crossing actions
are almost always locally progress-making (nonprogress 1.5% -> 0%). This does
not establish that either action has the best long-run return: later turns and
wall encounters still matter.

Repeated identical wall-hit actions occupy 44.0% -> 53.1% of actions after
post-crossing restarts. The fraction of those restarted rollouts containing a
wall-hit streak of at least five identical actions grows 49.5% -> 64.1%.
Later collision loops, rather than an initially blocked move, distinguish the
policies in these contexts.

## Time-budget control

Both models receive exactly the same budget for a given probe. The deficit
persists with either the original remaining budget or a fresh 150 steps:

| Stage | Mean original budget left | Success with original budget, 400 -> 800 | Fresh 150-step success, 400 -> 800 |
| --- | ---: | --- | --- |
| First approach | 27.5 | 26.2% -> 17.2% | 45.9% -> 33.3% |
| Just crossed | 22.0 | 40.4% -> 28.6% | 61.9% -> 46.9% |
| Second-door attempt | 19.1 | 42.6% -> 32.8% | 63.3% -> 53.2% |

Hence elapsed time before reaching the first doorway is not a sufficient
explanation. The stage cohorts differ; comparing their absolute success rates
against one another is not a causal stage-by-stage decomposition. The valid
controlled comparison is **400 versus 800 within each identical probe set**.

## Source-trajectory and task-selection controls

The result is not caused solely by extracting states from epoch 400:

| Probe source | Post-crossing success under 400 | Under 800 |
| --- | ---: | ---: |
| Source trajectory 400, 106 states | 49.8% | 32.1% |
| Source trajectory 800, 97 states | 42.8% | 33.1% |

The direction of the success deficit holds for both source sets at every stage.
Source-specific collision rates and all other metrics are retained in the full
summary. Identical state/goal contexts can occur twice if both source traces
visit them; they are retained to preserve the source-specific comparisons.

For the **38 lost-success tasks**, using only their original epoch-400 stage
states provides a particularly direct counterfactual:

| Restart stage, 38 probes each | Success under 400 | Under 800 | Wall-hit fraction under 400 / 800 |
| --- | ---: | ---: | --- |
| First approach | 69.2% | 15.5% | 28.9% / 55.7% |
| Just crossed | 80.3% | 32.4% | 29.3% / 62.3% |
| Second-door attempt | 88.1% | 59.0% | 27.3% / 51.2% |

Neither model initially prefers a nominal wall-hit action on these selected
stage states. Their later policies nevertheless differ sharply. This lost-task
subset was selected by the original single-rollout outcomes, so the broader
all-task comparisons should be read alongside it to avoid selection bias.

## Full-start Monte Carlo control

Independently restarted all 200 original diagonal tasks, 32 repetitions each,
using the same seeds for both models:

| Horizon | Epoch 400 | Epoch 800 |
| --- | ---: | ---: |
| 30 steps | 23.78% | 15.08% |
| 150 steps | 43.59% | 29.80% |

The earlier 24% -> 12% figures were one rollout per goal. The repeated estimate
still shows deterioration, but puts the 30-step difference near 8.7 percentage
points for this Monte Carlo sample, rather than exactly 12 points. It would be
misleading to attribute every point of the original difference to training.

## Relationship between rank changes and return changes

On post-crossing probes, fresh-30 success decreases by 17.4 percentage points on
probes whose initial preferred action changes, and by 10.6 points where it does
not. At second-door attempts those changes are -16.3 and -8.5 points. Both first
action choices and later continuation can therefore contribute.

At first-approach probes the pattern reverses: preferred-action changes improve
success by 4.9 points on average, while unchanged preferences lose 14.6 points.
Thus a changed first action is neither necessary nor sufficient for failure;
calling every rank change an error would be incorrect.

Descriptive paired bootstrap intervals, clustering both source states by original
task, put the mean success differences at:

- First approach: -9.6 points, interval [-15.2, -4.0].
- Just crossed: -13.9 points, interval [-21.9, -5.8].
- Second-door attempt: -11.1 points, interval [-18.1, -4.4].

These describe variation over the sampled tasks, not independent training seeds.
The samples are conditioned on reaching each stage; they do not estimate an
unconditional causal contribution of that stage to the total failure rate.

## Why generic fixed-doorway probes appeared stable

The earlier synthetic doorway suite used a regular geometry-defined grid and
different goals. Its diagonal 30-step success was 25.5% -> 25.8%. That is compatible
with deterioration on the actual route-conditioned state/goal pairs here.
Generic starts can bypass the approach history and omit the particular exit
positions, orientations and goal contexts where the policy is fragile.

This experiment demonstrates the distribution dependence of the earlier finding.
It does **not** separately identify whether state geometry, goal selection or
their interaction explains that distribution difference. The policy is Markov:
the past history itself is not an input. "Route context" here means the current
position and goal, plus the remaining evaluation budget (which the policy does
not observe), not a hidden dependence on history.

The evidence most strongly supports weaker continuation through the actual
post-crossing/second-door task contexts, with later wall loops. It does not prove
a global representation collapse or specify an optimizer-level cause. The next
targeted diagnostic is the first later turn/wall-facing disagreement after
crossing, with alternate-first-action Monte Carlo returns at that exact state.

## Exact stage definitions and availability

Only original diagonal trajectories within their first 30 steps are used:

1. First approach: earliest pre-action state within .1 of a doorway rectangle
   whose adjacent room reduces the remaining room-transition count. Both routes
   are accepted for diagonal goals.
2. Just crossed: pre-action state immediately after the first completed
   room-distance-reducing crossing, outside the wall strip.
3. Second-door traversal attempt: three actions after that crossing, if not
   already successful/in the goal room or outside the original budget. This is
   an *attempt* to continue, not a claim that the position is moving toward the
   correct door; stalled or backtracking states are allowed.

Availability out of 200 trajectories per source checkpoint:

| Source | First approach | Just crossed | Second-door attempt |
| --- | ---: | ---: | ---: |
| 400 | 177 | 106 | 105 |
| 800 | 177 | 97 | 97 |

Stage omissions are recorded rather than replaced with another trajectory's
state. The resulting 759 probe rows correspond to fewer distinct tasks. Each
checkpoint is evaluated at all 759 points, with matched rollout keys. Collisions
are divided by executed actions before goal/horizon; unsuccessful rollouts often
contribute more actions. Accordingly, the collision percentages characterize
experienced behavior, not a state-distribution-independent collision probability.

## Reproduction and artifacts

```powershell
& ./.venv/Scripts/python.exe -m unittest tests.test_navigation_diagnostics -v
& ./.venv/Scripts/python.exe compare_navigation_stages.py
```

Three tests passed: exact extraction times/budgets, collision/repeat/streak
windows, and remaining-budget success/discount/denominator handling.

Primary outputs: `probes.json`, `probe_geometry.npz`, `epoch_400.npz`,
`epoch_800.npz`, `summary.json`, `action_rankings.csv` and `stages.png`.
All raw first-hit times and three-window collision/repeat/streak counts are
retained. Config records the analysis script hash, source trace hashes and
Monte Carlo seed. Use a fresh output directory when changing suite settings.

Additional derived artifacts are `ranking_and_cluster_intervals.json` and the
32-repetition original-start control (`full_start_400.npz`, `full_start_800.npz`,
`full_start_summary.json`). They use the same rollout evaluator/seed, with the
original diagonal starts substituted as probe states and a budget of 30.
