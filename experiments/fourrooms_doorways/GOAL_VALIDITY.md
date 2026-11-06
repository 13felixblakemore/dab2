# Goal and state validity audit

Checked the saved doubled-doorway run and all subsequent diagnostic probes
against the **exact wall coordinates in its metadata**. The current environment
matches those coordinates exactly. Both float64 inclusive wall checks and the
native float32 wall bounds give the same results.

| Saved set | State rows | Goal rows | Inside walls | Outside square | Disconnected goals |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original evaluation pairs | 600 | 600 | 0 | 0 | 0 |
| Geometry-defined doorway probes | 160 | 160 | 0 | 0 | 0 |
| Matched trajectory-stage probes | 759 | 759 | 0 | 0 | 0 |

Also checked all 79,822 active pre-action positions in the epoch-400 and epoch-800
trace files: zero inside walls or outside the square. These counts include
repeated coordinates and repeated goals; they are not counts of distinct points.
The minimum original goal-center clearance from an interior wall is .02045.

The free .05-center lattice contains 352 nodes in **one connected component**.
Validated all 1,276 directed graph edges using the environment's actual `_move`
kernel with noise set to zero: no rejected edges and no coordinate mismatch above
1e-7. Goal-terminal sets use the exact float32 coordinate-wise .1 predicate.
Every goal has free terminal nodes in the connected graph. Every off-grid stage
state has a collision-free geometric connection to a nearby free graph node.

Thus there is no evidence of goals accidentally sampled in walls or disconnected
free-space regions. This does not guarantee success within a finite horizon under
noise. Off-grid stage BFS lengths use a visible nearest-node attachment and are
approximate, rather than exact optimal noisy-continuous control distances.

## Horizon feasibility caveat

- One of 200 original diagonal pairs has a **31-step noiseless BFS reference**.
  All other original pairs fit 30 steps. The shortest continuous noisy route need
  not equal this restricted-lattice reference, so this is not proof of strict
  impossibility in 30 noisy steps.
- All 160 generic doorway probes and all 759 matched stage probes have reference
  paths within a **fresh 30-step** budget (maxima 28 and 25 respectively).
- 28 stage probes have a reference longer than their **remaining original** budget:
  seven first approaches, eleven just-crossed states and ten second-door attempts.

Excluding those reference-budget-limited cases preserves the performance gap:

| Evaluation on reference-feasible subset | Checkpoint 400 | Checkpoint 800 |
| --- | ---: | ---: |
| Original diagonal starts, 199 pairs, H=30 | 23.90% | 15.15% |
| First approach, 347 probes, original remaining budget | 26.68% | 17.55% |
| Just crossed, 192 probes, original remaining budget | 42.55% | 30.13% |
| Second-door attempt, 192 probes, original remaining budget | 44.68% | 34.41% |

The horizon issue should be reported, but it does not explain the observed
deterioration. Fresh-budget stage comparisons require no exclusions by this
reference feasibility test.

## Goal centers versus goal regions

252/600 original goal **success squares** overlap an interior wall because the
unchanged environment uses coordinate-wise tolerance .1. Their centers are all
outside walls and reachable, and the target square contains reachable free
positions. Overlap does not make a goal invalid. It can allow success before
fully entering the goal's nominal room, as noted in the earlier doorway reports.
Requiring the entire success square to be wall-free would change the task
distribution; no such change was made here.

The training sampler also rejects wall points, but historical training goals are
not stored in replay and were not individually inspected by this saved-data audit.

Reproduce with `python audit_doorway_goals.py`. Two unit tests passed, including
deliberately invalid wall/outside points and a disconnected graph. Raw audit:
`goal_validity_audit.json`; filtered results:
`goal_validity_feasible_performance.json`; reference lengths:
`goal_validity_reference_lengths.npz`.
