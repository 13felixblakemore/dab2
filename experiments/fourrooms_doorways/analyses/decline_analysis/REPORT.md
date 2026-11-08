# Why does diagonal success deteriorate after epoch 400?

The observed behavioral mechanism is degraded action selection near walls and
failure to continue after the first room crossing. This is consistent with
instability/drift in learned action advantages, but the measurements do not
establish a general Bellman-error explosion or an optimizer-level root cause.
No training configuration or checkpoint was modified.

## Verification and measurement design

Replayed both checkpoints with native continuous dynamics, online B(goal), online
F, epsilon=.02 and identical evaluation keys. All 600 trial success outcomes at
both horizons (30 and 150) exactly match the uploaded evaluations: **zero
disagreements** at either checkpoint. The 30-step diagonal decline is 48/200 to
24/200: 38 successes are lost, 14 are gained and 10 remain successful.

Compared both models on **9,600 identical state/goal probes**: 16 per trial,
sampled equally from each checkpoint's pre-action states up to step 30. This
prevents changing state distributions from masquerading as changing rankings.
There are 3,925 probes within .05 of an interior wall. Probes are not statistically
independent; correlations below are computed over 200 diagonal trials, not 9,600
independent observations. Results concern one training seed.

## Value scales and residuals

| Diagnostic | Epoch 400 | Epoch 800 |
| --- | ---: | ---: |
| Q RMS, common state/goal probes | 3,964.6 | 5,060.3 |
| Median top-two Q gap | 26.14 | 25.05 |
| Online-target Q RMSE, same online task | 31.24 | 35.25 |
| Online-target centered-action-advantage RMSE | 15.46 | 18.57 |
| Goal-task FB bootstrap RMSE, common probes | 23.19 | 23.57 |
| Same RMSE / RMS bootstrap target | 3.52% | 3.16% |
| Native-noisy held-out Cauchy-task bootstrap RMSE | 25.20 | 29.86 |
| Same RMSE / RMS bootstrap target | 6.29% | 5.36% |

Q scale grows about 28%, while the median action gap does not. Centered-action
target disagreement grows about 20%. These indicate action advantages can change
meaningfully despite a large shared value offset. Q scale is not a calibrated
goal-reaching probability and scale growth alone is not evidence of error.

The held-out test uses 1,920 native noisy random-collection transitions, uniform
resets and the environment's original goal-triggered resets, with 32 independent
pairs of minibatches of size 128 and identical Cauchy task draws for both models.
Goal-conditioned probes use identical nominal zero-noise action transitions and
128 independent uniform free-space backward states. Those two diagnostics use
different distributions and should not be compared as if their absolute scales
were interchangeable.

Both residual measures are the sampled FB **bootstrap matrix term**
`F(s,a,z) B(other) - gamma E_pi[F_target(s_next,a,z)] B_target(other)`.
They exclude the immediate/Dirac occupancy contribution handled separately by
the FB objective. They are not complete pointwise Bellman residuals, estimates of
ground-truth Q error, or sparse-goal-reward TD errors. The normalized residuals
improve even while navigation deteriorates. Thus a blanket claim of exploding
Bellman error would not be supported by these results.

The B second-moment trace changes 9.88 -> 9.58 and participation effective rank
20.82 -> 20.03. This does not show a dramatic B representation collapse between
these checkpoints, although it does not establish that B is sufficient for the
navigation task.

## Action rankings near walls

On identical near-wall probes:

| Metric | Epoch 400 | Epoch 800 |
| --- | ---: | ---: |
| Greedy nominal action intersects an interior wall | 57.4% | 62.4% |
| Greedy fails BFS progress where a progress action exists | 58.6% | 65.3% |

The greedy action changes at 32.3% of near-wall probes; the complete ordering of
four actions changes at 55.5%. Nominal collision checks use the actual thick-wall
geometry at zero noise. BFS progress uses a noiseless restricted .05 lattice and
nearest-node projection, a diagnostic proxy rather than exact optimality for the
stochastic continuous system.

For the **38 lost diagonal trials**, on common probes:

| Metric | Epoch 400 | Epoch 800 |
| --- | ---: | ---: |
| Greedy chooses a nominal wall-hit action | 6.7% | 26.0% |
| Greedy fails available BFS progress | 8.8% | 34.1% |
| Actual wall-hit actions in the trial | 2.85% | 50.00% |
| Actual identical-action consecutive wall hits / all actions | 1.56% | 45.88% |
| Mean longest consecutive identical wall-hit run | 0.58 | 14.05 |

Example, trial 409 at `(0.5175, 0.3175)`, goal `(0.725, 0.225)`:
epoch 400 ranks DOWN (Q=2003.34) over RIGHT (1994.98), moving along the wall toward
the lower door. Epoch 800 ranks RIGHT (1503.15) over DOWN (1486.81), preferring an
action into the vertical wall. This example also illustrates that problematic
ranking changes can happen while the individual state's absolute Q values fall.
Six examples with all four scores are saved in `lost_cases.json`.

Across diagonal trials, Pearson correlation of metric change with success lost
(+1), unchanged (0), or gained (-1):

| Metric change | Correlation with deterioration |
| --- | ---: |
| Greedy wall-hit fraction | .429 |
| BFS nonprogress fraction | .425 |
| Raw goal-task bootstrap RMSE | .110 |
| Normalized goal-task bootstrap RMSE | .046 |
| Q RMS | .080 |
| Online-target Q RMSE | .027 |

Action-ranking degradation tracks failures substantially better than residual
or value-scale growth. These are exploratory paired correlations, not a causal
estimate or evidence that scaling Q down would fix the policy.

## Where doorway navigation fails

A useful doorway connects the current committed room to a neighbor reducing
the number of remaining room changes to the goal room. Both routes are accepted
for diagonal goals. Approaching means within .05 (one nominal step) of the full
doorway rectangle. A completed crossing requires entering another room, not just
the doorway strip. This is a room-topology criterion; it does not prove which
door is globally shortest in continuous path length. Counts at .025 and .1 are
also saved to expose approach-threshold sensitivity.

At 30 steps, each row sums to 200 trials:

| Task/checkpoint | Never approaches useful door | Approaches but no useful crossing | Crosses then fails later | Success |
| --- | ---: | ---: | ---: | ---: |
| Adjacent, 400 | 39 | 56 | 7 | 98 |
| Adjacent, 800 | 47 | 51 | 7 | 95 |
| Diagonal, 400 | 50 | 43 | 59 | 48 |
| Diagonal, 800 | 46 | 57 | 73 | 24 |

The diagonal decline is not primarily an increase in never approaching the first
useful doorway. More trials fail at crossing/control or after the first crossing.
Of the 73 epoch-800 diagonal trials that cross and then fail:

- 45 never approach the next useful doorway.
- 23 approach another useful door but do not complete the route.
- 5 enter the goal room but fail to reach its goal region.
- 57 of the 73 have a run of at least five identical rejected actions **after**
  the first useful crossing, direct evidence of subsequent wall trapping.

Among the 38 lost successes specifically: 5 never approach a useful door, 13
approach but never complete a useful crossing, and 20 cross before failing. Of
those 20, 18 have a five-action post-crossing wall loop. Twelve fail to approach
the next door, seven approach it without finishing, and one fails goal control
after entering the goal room.

At 150 steps in checkpoint 800, adjacent failures split into 33 never approaching,
35 approaching without crossing, and 4 failing after crossing (128 successes).
Diagonal failures split into 37 never approaching, 46 approaching without a useful
crossing, and 64 failing afterward (53 successes). Of the latter 64, 56 have a
post-crossing five-action wall loop. Longer horizons do not resolve these traps.

Do not equate every post-crossing failure with trapping: the explicit run-length
measure supplies that extra evidence. Goal tolerance can also allow success
before entering the nominal goal room.

## Diagnostic intervention: mask known wall-hit greedy actions

As an additional diagnostic only, supplied exact wall geometry at action time and
masked actions whose zero-noise move intersects a wall before the greedy choice.
The epsilon-random branch, real movement noise and stopping rules are unchanged.
This is a map-oracle policy modification, **not** the learned FB policy or a
reference-compatible evaluation score. No model was retrained.

| Diagonal success | Epoch 400 | Epoch 800 |
| --- | ---: | ---: |
| Ordinary FB, 30 steps | 24.0% | 12.0% |
| Map-oracle wall mask, 30 steps | 73.0% | 61.0% |
| Ordinary FB, 150 steps | 37.5% | 26.5% |
| Map-oracle wall mask, 150 steps | 81.5% | 72.5% |

Masking rescues 25 of the 38 lost successes at epoch 800. Nevertheless, it raises
both 30-step checkpoint scores by exactly 49 percentage points: the **12-point
late gap remains**. This shows wall-preference errors are a major general failure
mode but not a sufficient explanation of the entire late-training difference.
Even after masking, more epoch-800 diagonal trials fail to approach/complete the
next doorway. Incorrect rankings among physically feasible actions and route
selection still need attention.

## Next experiment justified by this investigation

Prioritize held-out **action-advantage and near-door ranking diagnostics** during
training, alongside the ordinary FB loss. Keep the existing hyperparameters as
the baseline. A controlled stability experiment could vary target-update cadence
or learning rate one factor at a time, but these results do not yet identify
either as the cause. Use matched seeds and the same software/backend, and a
separate validation goal set for checkpoint selection; retrospectively selecting
400 on this evaluation set would introduce selection bias.

Before another full training run, extend the saved map-oracle diagnostic to
inspect which feasible actions are chosen in the masked policy's failures, and
compare short noisy rollouts of alternate actions at the first disagreement
state. This tests whether the remaining issue is route choice, oscillation or
local control without silently changing the FB training objective.

## Artifacts and reproduction

```powershell
& ./.venv/Scripts/python.exe -m unittest tests.test_navigation_diagnostics -v
& ./.venv/Scripts/python.exe investigate_doorway_decline.py
```

Four tests passed, including synthetic stage labels, repeated wall actions,
paired probes and separation of the oracle mask from ordinary policy behavior.
Outputs include `summary.json`, complete ordinary and oracle traces, per-trial
stages, common probes/action scores, held-out transitions, checkpoint hashes and
software/script provenance. `lost_cases.json` and
`post_cross_and_mask_stages.json` additionally summarize selected cases derived
from those arrays. `lost_trajectories.png` shows blue epoch-400 vs red epoch-800
paths for six lost successes; `paired_deltas.png` shows trial-level changes.
