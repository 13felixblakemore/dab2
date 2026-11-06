# New replay experiment: continuing objective

The new uploaded folder is `runs/width2_replay_continuing_seed0`, whose saved
configuration has **`z_sampling="replay"`, `goal_terminated=false`**. Despite the
request saying "non continuing", this is the recently requested experiment with
the continuing FB objective unchanged. The older non-continuing/stopped variant
is `runs/width2_replay_stopped_seed0` and is included separately below.

## Configuration and completion audit

The new run completed 800 epochs, 800,000 updates and 2.4 million collected
transitions. All 800×25×40 recorded losses are finite. Every training-row update
and transition counter matches the configured schedule.

Compared resolved configurations: **the only difference from the original
Cauchy-continuing width-2 run is `learner.z_sampling`**. Both use d=50,
lr=.0005, Polyak=.95, gamma=.99, T=200, the released B regulariser, zero-task
mixing 0 and no clipping. Walls, actions, goal tolerance and the monitor pairs
match. Both cluster runs record JAX .5.2, Python 3.12.3 and NumPy 2.2.3.
The learner-source difference adds a configurable goal norm inside the disabled
termination branch; it does not change the continuing objective. Newer runner
differences concern checkpoint/recovery/logging cadence, not training equations.

This is one training seed. Matching a seed across different task distributions
does not force identical minibatch/task RNG sequences or establish population
performance across training seeds.

## Independent final-checkpoint evaluation

Used the same **600 fresh matched endpoint pairs**, balanced across three task
categories, and **32 noisy rollouts per pair** as the previous independent test.
All endpoints pass geometry validation. Online B(goal) and F, epsilon=.02,
native dynamics and coordinate-wise .1 success tolerance are unchanged.
Baseline Monte Carlo arrays were reused after checkpoint-hash checks; only new
conditions were simulated. Goal pairs are independent of the training-monitor
suite, but had been used for earlier model comparisons, so a final selected-model
claim should still use another held-out suite and additional training seeds.

| Model | Same room, H=30 | Adjacent, H=30 | Diagonal, H=30 | Same room, H=150 | Adjacent, H=150 | Diagonal, H=150 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cauchy continuing | 94.89% | 57.36% | 21.78% | 96.83% | 72.16% | 35.47% |
| **Replay continuing (new)** | **99.06%** | **72.14%** | **36.83%** | **99.67%** | **84.19%** | **55.39%** |
| Replay stopped | 13.33% | 3.45% | 1.83% | 18.03% | 7.52% | 4.41% |

Replay continuing gains **15.05 percentage points** of diagonal success at 30
steps and **19.92 points** at 150, relative to Cauchy. Paired bootstrap intervals
over the 200 task pairs are [+10.05,+20.14] and [+14.06,+25.63] points. These
measure task-sampling variation for these checkpoints, not training-seed
uncertainty. The local success gain is 4.17 points; adjacent gain is 14.78 points
at 30 steps.

Replay-derived tasks can therefore work well here. The poor result of the
earlier replay-plus-stopping experiment cannot be attributed to replay task
sampling alone. The stopped-bootstrap modification is the important remaining
difference to investigate, including its treatment of starts inside a goal and
the unchanged immediate term. These results identify a candidate issue, not a
proof of its exact mathematical or optimization mechanism.

The replay task distribution aligns training with the online B(g) tasks used at
evaluation. That is a plausible explanation for improvement, but task-coverage
or gradient-distribution measurements would be needed to establish it directly.

## Does late deterioration remain?

On the original fixed monitor pairs (one noisy rollout per pair):

| Epoch | Same H=30 | Adjacent H=30 | Diagonal H=30 | Adjacent H=150 | Diagonal H=150 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 200 | 98.0% | 59.5% | 23.0% | 78.0% | 37.5% |
| 400 | 98.0% | 63.5% | 22.5% | 80.0% | 40.5% |
| 600 | 98.5% | 63.5% | 24.0% | 77.5% | 43.0% |
| 800 | 99.5% | 66.5% | 29.0% | 81.0% | 50.0% |

Repeated checkpoint 400 on the **same independent 600 pairs and 32 rollout keys**
as checkpoint 800 to check that trend:

| Replay continuing | Epoch 400 | Epoch 800 | Change |
| --- | ---: | ---: | ---: |
| Same room H=30 | 99.09% | 99.06% | -0.03 points |
| Adjacent H=30 | 68.58% | 72.14% | +3.56 points |
| Diagonal H=30 | 31.91% | 36.83% | **+4.92 points** |
| Diagonal H=150 | 48.38% | 55.39% | **+7.02 points** |

Paired task-bootstrap intervals for diagonal changes are [+0.59,+9.28] points
at H=30 and [+2.53,+11.44] at H=150. This seed does **not** reproduce the earlier
late diagonal decline under Cauchy sampling. That does not prove replay always
stabilizes training; replicate seeds before claiming a general effect.

## Remaining wall and doorway problems

With replay continuing, at H=30:

- Mean successful paths: 4.91 / 13.19 / 21.21 steps for same/adjacent/diagonal.
- Wall-hit actions: 1.25% / 39.71% / 46.73%.
- Consecutive identical wall-hit actions: 1.13% / 36.69% / 43.09%.

At H=150, adjacent and diagonal wall-hit actions remain **69.32% and 73.55%**.
Higher success therefore coexists with persistent long wall traps on difficult
trials; the bottleneck problem is improved, not solved. Collision fractions pool
all actions before stopping, so failures contribute more actions.

Greedy replay-continuing success is 98.98% / 69.78% / 35.13% at H=30, and
99.27% / 72.56% / 39.94% at H=150. The epsilon policy's long-horizon diagonal
55.39% versus greedy 39.94% suggests occasional random actions help escape traps.
This contrast is diagnostic, not proof of a unique escape mechanism. Short-horizon
successes average fewer steps under greedy, partly because fewer hard trials
succeed; successful-length metrics alone should not be used to rank policies.

## Artifacts and reproduction

```bash
python evaluate_width2_goals.py \
  --models width2_seed0 width2_replay_continuing_seed0 width2_replay_stopped_seed0 \
  --output experiments/fourrooms_doorways/analyses/replay_continuing_comparison
```

`results.json` and `success.png` contain the independent final comparison.
`learning_curves.png` and `monitor_comparison.json` show all three monitor curves.
`paired_comparisons.json` contains paired sampling/late-change estimates and
intervals. `replay_continuing_epoch0400_epsilon002.npz` is the extra 400-epoch
test using the same Monte Carlo evaluator and seed 20261108 as the final test.
Raw per-policy hit times, collisions and checkpoint hashes are retained. No
training settings, weights or original results were overwritten.
