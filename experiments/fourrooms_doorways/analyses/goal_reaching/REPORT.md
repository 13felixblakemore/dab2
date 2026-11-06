# Fresh width-2 seed-0 goal-reaching test

Evaluated final epoch-800 checkpoints of both available width-2 seed-0 models.
Both contain 800,000 optimizer updates. No model was retrained or modified.

Used **600 fresh matched pairs**, balanced as 200 same-room, 200 adjacent-room
and 200 diagonal-room pairs; 32 independent noisy repetitions per pair. Endpoint
seed is 20261107, independent of the original training-monitor evaluation seed.
All saved endpoints passed wall/outside checks. Endpoints are sampled from the
baseline free room-interior center lattice, with geometry-specific BFS lengths
for doubled doors. This is a stratified diagnostic, not a uniform continuous-goal
estimate. Both models and all policies share endpoints and noise/action keys.

Online B(goal), online F, coordinate-wise .1 goal tolerance, four original
actions, noise .005 and native thick-wall rejection are used. Epsilon=.02 is
applied to every action, including the first. Success stops the trial. Horizons
30 and 150 use the same first-hit times; the long horizon is an extra diagnostic.

## Goal success

| Model / policy | Same, H=30 | Adjacent, H=30 | Diagonal, H=30 | Same, H=150 | Adjacent, H=150 | Diagonal, H=150 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cauchy continuing, epsilon=.02 | 94.89% | 57.36% | 21.78% | 96.83% | 72.16% | 35.47% |
| Replay + stopped bootstrap, epsilon=.02 | 13.33% | 3.45% | 1.83% | 18.03% | 7.52% | 4.41% |
| Random | 33.56% | 3.06% | 0.03% | 65.41% | 15.16% | 2.08% |

Greedy Cauchy success at H=30 is 94.61% / 54.58% / 21.27%; greedy replay/stopped
is 12.72% / 3.36% / 1.77%. Evaluation exploration does not explain the weak
replay/stopped performance.

Cauchy continuing maintains strong local navigation but still has cross-room
bottleneck failures. Replay + goal stopping is substantially worse on this run,
including local navigation below random. This changes sampling and bootstrapping
jointly, so it does not isolate which change causes the deficit. No parameters
were tuned in response to the results.

These fresh-task estimates differ from the earlier fixed 200-pair single-rollout
numbers, including original diagonal 12%. They do not imply the checkpoint has
improved: goal seed and repetition counts changed. This test supports comparison
between the two models under a common independent evaluation suite.

## Paths and collisions, epsilon=.02

| Model | Task | Mean successful steps, H=30 | Wall-hit actions, H=30 | Wall-hit actions, H=150 |
| --- | --- | ---: | ---: | ---: |
| Cauchy continuing | Same | 4.99 | 7.04% | 18.52% |
| Cauchy continuing | Adjacent | 13.18 | 44.42% | 68.77% |
| Cauchy continuing | Diagonal | 21.80 | 50.96% | 72.99% |
| Replay + stopping | Same | 7.30 | 27.69% | 34.39% |
| Replay + stopping | Adjacent | 13.21 | 31.27% | 44.47% |
| Replay + stopping | Diagonal | 18.92 | 31.80% | 42.24% |

Successful lengths exclude failures and cannot be used alone to rank policies.
The replay policy's lower cross-room wall-hit fraction does not imply better
control: it reaches far fewer goals. Rates pool executed actions before success
or the horizon. Bootstrap intervals across pairs are stored in `results.json`;
they describe sampled-task variation, not uncertainty across training seeds.

## Artifacts and reproduction

`results.json` contains all policy/horizon metrics, intervals, successful lengths,
collision/repeated-action fractions and BFS feasibility. `pairs.npz` preserves
the fixed tasks. Per-condition NPZ files preserve first-hit times and collision
counters, plus the model checkpoint hash. `success.png` visualizes the comparison.
Random is evaluated once because it is independent of model parameters.

```bash
python evaluate_width2_goals.py
```

Seven tests passed for reporting, censoring, remaining-budget/collision/streak
handling, deliberately invalid goals and graph disconnection.
