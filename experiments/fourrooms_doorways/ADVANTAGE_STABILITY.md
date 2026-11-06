# Fixed-doorway action-advantage and Monte Carlo study

## Results

The study completed epochs **200, 400, 600 and 800**. Epoch 100 is unavailable.
Ranks change frequently, but this independently defined local-doorway suite does
**not** reproduce a halving of diagonal success. The strongest conclusion is
state/route-dependent ranking drift rather than uniform loss of doorway control.

| Epoch | Blocked preferred action | Preferred action changed since preceding available checkpoint | Median preferred advantage | MC return, H=30 | MC return, H=150 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100 | unavailable | unavailable | unavailable | unavailable | unavailable |
| 200 | 14.4% | — | 29.44 | .3455 | .4160 |
| 400 | 10.6% | 25.0% | 25.55 | .4154 | .4836 |
| 600 | 12.5% | 21.9% | 24.30 | .4254 | .4893 |
| 800 | 11.9% | 21.9% | 26.21 | .4023 | .4747 |

Between 400 and 800, 37/160 probes (23.1%) change their preferred action: six
switch to a nominal wall-hit action and four switch away. Changes are spread
across all doors: bottom 10, top 14, left 4, right 9. Similar median advantages
therefore conceal changes in *which* action has positive advantage.

The ordinary 32-replica estimates show H=30 discounted return on all probes
changing .4154 -> .4023, a small -.0130. The descriptive bootstrap interval for
that paired change is [-.0569, .0329], spanning zero. H=150 return changes
.4836 -> .4747, also small on average.

For the 37 changed probes, compared old versus new first actions with the **same
epoch-800 continuation policy**, the initial mean return difference is -.1177 at
H=30. A separate validation using **128 rollouts per first action** and an
independent common-random-number seed gives:

| Effect on changed probes | H=30 | H=150 |
| --- | ---: | ---: |
| New minus old first action, epoch-800 continuation | -.1075 | -.0790 |
| New minus old first action, epoch-400 continuation | +.0542 | +.0480 |
| Entire checkpoint policy change on these probes | -.0735 | -.0520 |

For H=30 with epoch-800 continuation, 14 estimated switches are worse, 11 better
and 12 tied. The probe bootstrap interval for the mean switch effect is
[-.2027, -.0212]; this is descriptive for these structured probes, not an
independent-state or training-seed significance claim. Several effects are noisy
even with 128 trials. The dependence on continuation policy is substantive:
there is no universally better first action independent of the route policy.

The aggregate benefit of the new preferences is negative under their own
epoch-800 continuation, supporting imperfect alignment between learned ranks
and measured first-hit goal-reaching return. This does **not** identify a formal
continuing-FB Q error: both reward/stopping convention and continuation policy
differ from the training backup. In particular, the training backup is target
Boltzmann at T=200, while these rollouts use online epsilon=.02.

### Does diagonal navigation deteriorate from these doorway states?

There are 80 adjacent-goal probes and 80 diagonal-goal probes. First action is
forced to each checkpoint's preferred action, followed by its epsilon=.02 policy.

| Epoch | Adjacent success H=30 | Diagonal success H=30 | Adjacent success H=150 | Diagonal success H=150 |
| --- | ---: | ---: | ---: | ---: |
| 200 | 63.2% | 12.3% | 76.3% | 29.4% |
| 400 | 66.4% | 25.5% | 78.5% | 43.3% |
| 600 | 68.8% | 25.2% | 81.7% | 39.5% |
| 800 | 63.5% | 25.8% | 79.3% | 41.4% |

Diagonal H=30 return likewise changes only .2109 -> .2143 between 400 and 800.
Therefore, the earlier 24% -> 12% failure rate on room-interior endpoint trials
should **not** be generalized to a uniform decline around all doorway starts.
Starting near a doorway bypasses some approach/routing decisions, and this suite
uses different goals from the earlier endpoint trials. The previous
trajectory-derived probes deliberately included states visited by failing
policies, which explains why their wall-preference rates were much higher.

This refines the working hypothesis: late failures are concentrated in particular
approach routes and post-crossing states, with consequential rank reversals in
some of them. Generic local doorway states remain broadly functional. Next,
target the first action-ranking disagreement along the lost room-interior trials
and compare alternate-action Monte Carlo returns there, using the same method.
No learning-rate, target-update or other training change is justified solely by
the local-suite averages.

### Results artifacts

- `analyses/advantage_stability/summary.json`: checkpoint and transition measurements.
- `analyses/advantage_stability/advantages.csv`: all 2,560 checkpoint/probe/action rows,
  including raw Q, signed A, blocked/preferred flags and MC returns.
- `analyses/advantage_stability/changed_validation.json` and matching NPZ files: the
  independent 128-replica validation on the 37 changed probes.
- `analyses/advantage_stability/by_goal_kind.json`: adjacent/diagonal breakdown.
- `analyses/advantage_stability/stability.png` and `advantage_traces.png`: summary and
  example advantage trajectories. The latter shows the first eight changed
  probes in deterministic ordering, all at the bottom doorway; it is an example
  selection, not representative sampling of all doors.

All checkpoints and training settings were left unchanged. Four tests passed.

Run from the project root:

```powershell
& ./.venv/Scripts/python.exe track_doorway_advantages.py
```

The default input is the uploaded doubled-doorway seed-0 run. Requested epochs
are 100, 200, 400, 600 and 800. **Epoch 100 was not saved** under its 200-epoch
checkpoint schedule and is explicitly listed as missing. The script includes it
if supplied later, without substituting a narrower-doorway checkpoint or treating
epoch 200 as epoch 100. No training is performed.

## Probe and return definitions

The suite is fixed independently of success/failure outcomes: four doors, both
approach sides, five transverse offsets (-.6, -.3, 0, .3, .6 times doorway width)
and two stand-off distances (.025 and .05). That gives 80 physical states. The
outer transverse offsets deliberately include the doorway shoulders, where some
actions hit walls. Each state receives one fixed goal in the adjacent room across
that doorway and one in the diagonally opposite room, giving **160 state-goal
probes**. Goals are sampled from free room-interior lattice centers outside the
initial goal region. All states, goals and nominal wall masks remain fixed
throughout the study. This geometry-defined suite is different from the previous
failure-selected/trajectory-derived probe distribution.

For each checkpoint, use online `z=B(g)` and online F to calculate
`Q(s,a,z)=F(s,a,z)^T z` and, for every action, exactly
`A(s,a,z)=Q(s,a,z)-max_{b!=a}Q(s,b,z)`. Ties have zero advantage. Saved arrays
contain all four Q scores and all four signed advantages, not only the greedy
gap. No task renormalization or action mask modifies the learned policy.

For each of the 160 probes and all four candidate first actions, estimate returns
from **32 independent noisy rollouts**. The first action is forced; subsequent
actions use that checkpoint's online epsilon=.02 policy. Native movement noise,
wall rejection, boundary clipping and coordinate-wise .1 goal detection are
unchanged. The full rollout horizon is 150; the same hitting times provide the
30-step estimate. The first successful action terminates a rollout.

Return is `.99^(first_hit_step-1)` on success and zero on failure/censoring. Thus
success on the first action gives return 1. This measures discounted first-hit
goal reaching. The trained FB scores represent a continuing occupancy objective;
they are **not calibrated predictions of this stopped sparse-goal return**.
Only ordinal action preferences and behavioral returns are compared. Poor Monte
Carlo agreement is not by itself proof that the continuing FB Bellman equation is
incorrect.

Noise and exploration streams are shared across candidate first actions and
checkpoints using a fixed key per state-goal probe/rollout replica. A first action
that hits a wall remains a legitimate candidate; no map-oracle intervention is
used in this study.

## Separating first-action changes from continuation-policy changes

When a preferred action changes from a_old to a_new, compare:

1. `MC_current(a_new) - MC_current(a_old)`: same current continuation policy,
   different forced first action. This isolates whether the new preference helps
   under that continuation policy.
2. The analogous difference under the previous continuation policy, checking
   whether the conclusion depends on which policy follows the first step.
3. `MC_current(a_new) - MC_previous(a_old)`: the entire checkpoint policy change,
   which combines changes in the first action and later behavior.

Reports cover adjacent available checkpoints plus epoch 400 versus 800. Changes
are counted as positive/negative/tied estimates, and approximate paired Monte
Carlo standard errors distinguish larger differences from noisy ones. Thirty-two
rollouts can miss rare events; a tiny estimated difference is not a definitive
action-order reversal. Empirical best-action regret is upward biased because it
takes a maximum of noisy MC estimates. Fixed-first-action greedy returns also
differ slightly from ordinary epsilon-greedy evaluation that randomizes the first
action as well.

Summary intervals bootstrap the fixed state-goal probes and are descriptive:
states are spatially structured, goals share physical states, and there is only
one training seed. These are not confidence intervals for variation across
training seeds or a representative uniform free-space task distribution.

## Artifacts and validation

Outputs: `analyses/advantage_stability/fixed_probes.npz`, one cached `epoch_NNNN.npz` per
available checkpoint, `summary.json`, `stability.png`, and `advantage_traces.png`.
Each cache retains Q, all signed advantages, all action/replica first-hit times,
first-step actual collisions, nominal blocked-action masks and checkpoint hash.
The suite records software versions, random seed, horizon, replica count and
script hash; changing suite settings requires a new output directory. Caches
avoid repeating completed Monte Carlo work on reruns.

For GPU analysis, append `--device cuda --output <fresh-output-directory>`.
For a more precise repeat, use e.g. `--rollouts 128` and a fresh output directory.
This changes diagnostic precision, not the model or training hyperparameters.

Four tests passed: signed advantage/tie handling, discount convention,
deterministic free-space probes, forced first actions, and the switch comparison's
fixed-continuation definition. Run with:

```powershell
& ./.venv/Scripts/python.exe -m unittest test_doorway_advantages -v
```
