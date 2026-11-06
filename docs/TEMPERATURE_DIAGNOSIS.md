# Diagnosis: softmax temperature and FB navigation

## Setup

No learner, checkpoint parameter, training algorithm or stored training temperature was modified. The sweep uses the exact same 1,000 saved cross-room pairs and 20x20 candidate cells as the preceding control. Both checkpoints are evaluated with **target F** and their own unnormalised online B(goal), on the same deterministic 44x44 graph, Euclidean goal radius .1, horizon 10,000 and four independent rollouts per pair (4,000 trials per condition). Every trial/time receives the same uniform random number across conditions, even when other trials have terminated.

The goal-terminated checkpoint SHA256 starts `3bbe973...`; the older continuing checkpoint starts `bdb1d3...`. Both were available and evaluated. The evaluation uses graph spacing 1/44 and no movement noise, whereas training used continuous step .05 with noise .005. This is a controlled graph diagnostic; absolute success percentages and temperature recommendations require native-environment validation before deployment or retraining.

## Is T=200 approximately uniform?

**Yes, especially for the goal-terminated model.**

| Q / policy diagnostic | Goal-terminated | Older continuing |
|---|---:|---:|
| Raw Q median | 18.77 | 714.20 |
| Median max-min action span | 7.42 | 70.53 |
| Median best-second action gap | 2.21 | 17.19 |
| Median gap at the sampled initial states | 1.44 | 10.80 |
| Mean entropy at T=200 / log(4) | 99.943% | 97.782% |
| Mean maximum action probability at T=200 | .259 | .312 |
| Fraction with entropy >99% of maximum | 99.61% | 56.78% |

The global Q offset does not determine softmax randomness; **action differences relative to T** do. A gap of 2.21 at T=200 produces a best/second odds ratio near exp(2.21/200)=1.011. Histograms and full Q tables are saved. These statistics weight sampled unique goals and all free graph states equally; they are not visitation-weighted statistics.

## Navigation effects

The full comparison table is `evaluation_temperature_sweep/RESULTS.md` and `comparison.csv`. The main goal-terminated results are:

| Policy | Goal success | Mean successful steps | Median successful steps | Wall actions | Mean first doorway* | Door emphasis, gamma=.99 |
|---|---:|---:|---:|---:|---:|---:|
| T=200 | 95.7% | 2813 | 2185 | 5.1% | 527 | .97x |
| T=100 | 97.3% | 2036 | 1415 | 6.9% | 445 | 1.10x |
| T=50 | 91.6% | 1576 | 951 | 14.7% | 347 | 1.40x |
| T=20 | 68.7% | 1477 | 689 | 32.4% | 254 | 2.00x |
| T=10 | 42.2% | 1555 | 585 | 49.5% | 223 | 2.47x |
| T=2 | 9.6% | 1462 | 336 | 71.1% | 432 | 2.86x |
| T=.2 | 2.2% | 67 | 22 | 70.2% | 239 | 3.05x |
| Greedy | 2.1% | 22 | 21 | 62.8% | 12 | 3.13x |
| Epsilon=.02 | 14.8% | 1943 | 946 | 68.6% | 448 | 3.32x |

*Doorway times condition on arriving before the goal/cap; arrival rates are in the CSV. Successful lengths exclude failures. These conditional numbers become misleading alone for cold/greedy policies: greedy reaches a doorway in only 26.4% of trials. Mean capped steps, which include failures, rise from 3,122 at T=200 to 9,790 for greedy. Strong doorway emphasis does not imply successful navigation.

T=100 provides the clearest improvement for the goal-terminated model: success increases by 1.575 percentage points and mean capped length drops from 3,122 to 2,253 steps. A paired bootstrap over the 1,000 endpoint-pair means gives a descriptive 95% interval of +.70 to +2.425 percentage points for the success difference (2,000 resamples, seed 903). This interval covers endpoint/trial variability, not different training seeds or generalisation to continuous dynamics.

## What is wrong besides temperature?

Reducing T further is counterproductive. The greedy action rankings do not reliably lead to a complete route: increasingly cold policies concentrate on wall collisions and traps. Only 3.16% of uniformly weighted nonterminal state/goal combinations choose a wall-rejected greedy action, but those rare states dominate actual cold-policy time, producing >60% wall attempts. About 68.8% of greedy actions reduce BFS distance; the older checkpoint reaches 88.6% on this diagnostic. Neither statistic guarantees that repeated greedy actions reach the goal.

The continuing checkpoint is markedly better at moderate temperatures under identical conditions: T=200 succeeds in 97.6% with 753 mean successful steps; T=100 in 96.1% with 448; T=50 in 94.6% with 293. Even this model deteriorates when made almost greedy: T=.2 succeeds in 26.9%, greedy in 25.1%, epsilon=.02 in 80.8%. Thus graph transfer and imperfect learned rankings also affect the older model. The checkpoint comparison is not a causal training ablation: seeds, task sampling and stopping objectives differ.

### Structural issue worth checking before retraining

The goal-terminated objective excludes the goal-entry state, while the actor still chooses z=B(g). In an exact successor representation, a reward supported entirely inside the terminal goal region has **zero pre-entry return from outside the region**. Consequently its action values cannot themselves distinguish fast goal-reaching actions. The usual continuing-FB interpretation of B(g) as a goal-reaching reward does not transfer automatically to this stopped measure. Finite-rank B features and imperfect densities can create nonzero action differences, but those differences are not guaranteed to encode reliable goal-reaching preferences.

This is a mathematical concern with the stopped-measure/derived-actor combination, not proof that it uniquely caused every observed failure. It merits a separate design analysis before using a temperature change as the only training remedy. No learning changes were made here.

## Recommendation

For this checkpoint, **T=100 is a reasonable evaluation setting to validate next in the native noisy continuous environment**. It improves both success and capped time on this control. Do not jump to T=2 or .2: their action rankings create severe failures.

For a future, explicitly approved retraining study, begin conservatively around **T=100-200**, with T=100 as the first candidate and T=200 retained as a baseline. T=50 can be included as a speed-versus-reliability comparison, but the current goal checkpoint loses success there; T<=20 is not supported by these results. Training changes the Q scale and policy jointly, so the evaluation optimum need not be the retraining optimum. Check the actor/stopping-objective compatibility and native-dynamics performance first; use multiple training seeds and held-out goals for any eventual retraining comparison.

Changing evaluation temperature changes the executed policy. It **does not** make the existing F/B density a successor measure for that new policy. All betweenness scores in this sweep are Monte Carlo first-hitting estimates, not FB predictions relabelled with another temperature. Discounting of these first hits remains .99; model training discount is unchanged.

## Audits, plots and reproduction

Action order LEFT/RIGHT/UP/DOWN, graph directions, all-four-action availability (wall rejection is a self-transition), RBF NumPy/JAX preprocessing, and greedy action agreement with learner.act were checked. Both checkpoint hashes were checked again after evaluation. Seven regression tests passed, including stable cold softmax and common random numbers after other trajectories finish.

- `evaluation_temperature_sweep/temperature_comparison.png`: success, length, wall hits, doorway time, emphasis and entropy.
- `evaluation_temperature_sweep/q_gap_distributions.png`: action-gap/span distributions.
- `evaluation_temperature_sweep/goal_terminated_betweenness.png`, `continuing_betweenness.png`: all policies' MC maps.
- `metrics.json`: audits, Q distributions, entropy and full numerical results.
- `goal_terminated_q.npz`, `continuing_q.npz`: complete Q tables for independent analysis.

Run `python -m fb_navigation.diagnostics.sweep_fb_temperature`; options select checkpoints, pairs, rollout count, horizon, seed, device and output. A short pipeline check uses `--pairs evaluation_bfs_control_smoke/intermediate.npz --rollouts 1 --horizon 100 --output evaluation_temperature_sweep_smoke`. Run tests with `python -m unittest tests.test_sweep_fb_temperature tests.test_evaluate_epsilon_bottlenecks tests.test_compare_bottleneck_policies -v`.
