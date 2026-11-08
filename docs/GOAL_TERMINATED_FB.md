# Optional goal-terminated FB successor measures

## Usage and compatibility

```powershell
python -m fb_navigation.training.run_fb_jax --z-sampling replay --goal-terminated --goal-threshold 0.1 --output-dir models/goal_terminated
```

Or construct `FBLearner(..., z_sampling="replay", goal_terminated=True, goal_threshold=0.1)`.

The default is **goal_terminated=False**, preserving the original continuing objective and old checkpoint semantics. Enabling the flag only affects replay-sampled, nonzero goal tasks. Cauchy tasks remain continuing even if the flag is set. Explicit zero-task samples (`zero_z_p`) also remain continuing, since they have no goal. Neither gamma nor network architecture changes; the runner retains gamma=.99.

`goal_terminated` and `goal_threshold` are stored in checkpoint config. Older checkpoints missing these keys load as continuing, with threshold default .1. Loading old weights does not retroactively make them stopped-measure models. A custom learner passed to run_fb owns its configuration; runner constructor options configure a newly created learner only.

The original replay sampler, independent goal draws, detached online B(goal), RBF feature computation, Adam, clipping, target-network averaging, action masking and collection schedule are preserved. Goals passed to the loss are the original raw (x,y) coordinates, not RBF features. The collector's reset goals are unrelated to the per-transition replay goals used for bootstrapping; collector terminal flags remain unused in the FB loss.

## Modified target

For row i and target state j, the new detached bootstrap is:

Z_next[i,j] = (1 - reached[i]) * stop(F_target_next[i] dot B_target_other[j]),

reached[i] = (||raw_next_obs[i] - raw_goal_obs[i]||_2 <= goal_threshold) AND NOT zero[i].

The factor applies to the **entire row**, not a column or a global batch mask. It is enabled only when `goal_terminated` and `z_sampling == "replay"`.

The loss remains:

0.5 mean((Z - gamma Z_next)^2) - mean(z_diag) + reg_coef * reg_loss.

The immediate diagonal term and the backward regularizer, including all detach locations, are unchanged. The unfinished Python `if obs_next.is_close_to(goal)` fragment in the inspected source was replaced with a vectorised JAX mask.

## Why this excludes goal entry

For a start outside G, let tau be the first entry time into the Euclidean goal region. The intended measure is the expectation of sum over t=0,...,tau-1 of gamma^t delta_s_t. On the transition that enters G, the current state still contributes its immediate visitation, but the next-state bootstrap is zero. Therefore the goal-entry state and every subsequent state are excluded. Counting the goal-entry state would require a different boundary/target; it is not what is implemented.

**Start-in-goal limitation:** keeping delta_s unconditional means the recurrence is not the zero measure at tau=0. If a replay transition starts inside G and leaves it, next-state masking alone can even permit subsequent bootstrapping. This change deliberately follows the requested loss and does not claim equivalence at this boundary. Such rows remain in replay training; they can influence approximation quality. The stopped-MC comparison explicitly rejects starts in G. An exact zero boundary for all starts would require changing the immediate term and/or handling inside-goal rows explicitly, beyond this requested mask-only modification.

The Four Rooms environment itself uses coordinate-wise closeness (a square), while this new objective uses the requested **Euclidean disk**. The runner defaults to the environment's numeric threshold value, but does not change the environment's reset rule. Evaluation must use the learner's Euclidean radius, not env.done or Point.is_close_to. The stopped smoke evaluator does so.

## Validation

```powershell
python -m unittest tests.test_fb_training -v
python -m fb_navigation.training.smoke_goal_terminated_fb_jax
```

The regression suite checks inside/outside and boundary cases; independent row masks; the distinction between L2 and square tolerance; zero/Cauchy/disabled modes; full loss agreement with an independent NumPy oracle; detached goals and target gradients; finite online gradients; single updates and compiled scans; a compiled runner epoch; checkpoint round-trips and old checkpoint config compatibility. An analytical deterministic rollout verifies exclusion of the entry state and rejection of starts already at goal.

The standalone smoke script trains a **separate small model** for 3,000 updates on 15,360 Four Rooms transitions, then compares three tasks to 128 independent stopped rollouts of horizon 700 each. It writes only `evaluation_goal_terminated_smoke/`; existing trained checkpoints in models/ are untouched. It uses gamma=.99, radius=.1, the exact replay reference, online predictions, and the frozen target masked softmax (temperature 200) used in the loss. The first action is forced; goals are raw coordinates and z=B_online(goal). No continuing rollouts, goal-entry reward, map clipping, or mass renormalisation are used.

The small model produced finite losses (final -7.163) but is **not an accurate learned measure after this short run**:

| Task | MC stopped mass | MC mass SE | FB mass | Normalised spatial L1 |
|---|---:|---:|---:|---:|
| (.2,.2) -> (.35,.2), initial RIGHT | 12.11 | 1.95 | 2.94 | 1.066 |
| (.2,.7) -> (.8,.2), initial DOWN | 98.71 | .48 | 3.37 | .968 |
| (.8,.8) -> (.2,.7), initial LEFT | 95.22 | 1.12 | 3.28 | .983 |

This validates executable stopped-estimator plumbing, not convergence or measure accuracy. Online/target lag, low capacity and limited training can contribute. The shared representation can also alias goals and mix the stopped and continuing zero-task objectives. The exact start-in-goal boundary remains unresolved by design. Monte Carlo horizon truncation at gamma=.99 is at most .99^700/(1-.99)=.0881 mass units per trajectory and cannot explain the large errors. See `evaluation_goal_terminated_smoke/metrics.json` and `maps.npz` for reproducible outputs. More training and broader stopped-measure evaluation are required before claiming accuracy.
