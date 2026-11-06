# Four Rooms with released-code FB configuration

The full seed-0 experiment is running separately from Continuous Maze. Launch
metadata: `runs/seed0.process.json`. Live logs: `runs/seed0.stdout.log` and
`runs/seed0.stderr.log`. The older `run_fb_jax.py`, environment files, saved models
and Continuous Maze sources/configuration were not modified for this experiment.

```powershell
# Already launched; do not start another process in the same directory.
& ./.venv/Scripts/python.exe run_fourrooms_reference.py --seed 0
# After an interruption, with unchanged source/config:
& ./.venv/Scripts/python.exe run_fourrooms_reference.py --seed 0 --resume
# Live status:
Get-Content experiments/fourrooms_reference/runs/seed0.stdout.log -Tail 2
Get-Content experiments/fourrooms_reference/runs/seed0.stderr.log -Tail 10
# Validation:
& ./.venv/Scripts/python.exe -m unittest test_fourrooms_reference -v
```

Configuration is `config.json`: 800 epochs × 25 cycles × 40 updates = **800,000
gradient updates**. Four 30-step collection rollouts per cycle produce **2.4
million transitions**. Uniform random collection uses all four available actions,
including wall-rejected actions. RBF features are 21×21 Gaussian centers, sigma
.05, sum normalized. F and B have three hidden layers of 256 ReLUs, d=50, Adam
lr=.0005, T=200, gamma=.99, lambda=1, batch 128, FIFO capacity 1,000,000,
Polyak=.95 once per cycle. Zero-task mixing and gradient clipping are disabled;
the learner uses `b_regularization="touati_continuous"` and `goal_terminated=false`.

This reuses the existing `run_fb_jax.make_epoch` compiled collector/training
schedule and existing `FBLearner`. Continuing refers to the **loss**: no terminal
or goal bootstrap mask. Collection retains the original Four Rooms environment's
goal-triggered reset behavior (`explorable=False`), as the existing runner does.
This differs from Continuous Maze's never-terminal collection and is deliberately
preserved rather than silently changing environment behavior.

Unchanged environment: `ContinuousFourRooms(include_stay_action=False)`, four
actions LEFT/RIGHT/UP/DOWN, step .05, independent noise sigma .005, size 1, seven
thick wall blocks, coordinate-wise goal tolerance .1, existing boundary clipping
and wall rejection. **STAY is not added** and goal tolerance is not changed to the
maze's .05. This is an environment/configuration comparison, not the paper's
continuous-maze experiment on a different wall layout alone.

Every epoch writes a separate model checkpoint and all 1,000 scalar losses to
`losses.npy`. Unfinished entries in that file are NaN sentinels, not training
failures. `training.json` and stdout report completed epochs, losses, update and
transition counts. The latest atomic `resume.pkl` retains replay, collection keys,
and completed diagnostic history; model checkpoints retain learner JAX keys,
optimizer moments and target weights. Source snapshots/hashes and software
versions are archived under each run. Restarts resume at the latest completed
epoch, and preserve historical model checkpoints. Avoid concurrent writers to the
same output directory. Pickle artifacts should only be loaded when trusted.

## Fixed-pair navigation diagnostics

`evaluation_pairs.npz` is generated once and reused at checkpoints **1, 100, 200,
400, 600 and 800**, at horizons 30 and 150. It contains 200 pairs for each of:

| Task | Definition | Main question |
| --- | --- | --- |
| Same room | Both endpoints in the same room interior | Does local goal navigation work? |
| Adjacent rooms | Rooms share one doorway connection | Can the agent find and cross the required doorway? |
| Diagonal rooms | Opposite rooms, requiring at least two room changes | Can the policy combine bottleneck transitions? |

Endpoints are sampled from the existing .05-spaced free-cell-center lattice,
restricted to room interiors, excluding initially reached goals. Sampling is
balanced by task category, **not** uniform over all continuous free-space pairs;
the overall score should not be compared directly with Continuous Maze's uniform
goal score. Different room areas also affect the number of available pairs.
Online B(goal), online F, epsilon=.02, all four actions, native continuous noise
and collisions are used. Goal detection is the exact environment's inclusive
coordinate-wise .1 predicate after an action. Per-trial keys and endpoints are
identical across checkpoints, and the first 30 actions are shared across horizons
until success. Evaluation never advances training RNGs or replay.

Each checkpoint saves `evaluation/epoch_NNNN.json` and raw trial arrays (first
success times, wall-hit counts, four doorway counts) plus updated `navigation.png`.
For each task/horizon the metrics include:

- Success and mean successful path length (failures are excluded from that mean).
- Wall-hit percentage, pooled over all actions executed before success/horizon.
- Completed doorway crossings per trial, fraction with any crossing, and counts
  for bottom/top/left/right doorways. Entering the doorway strip and returning to
  the same room is not counted; exiting into another room is. Repeated completed
  crossings are counted. The .1 goal region can permit success before entering
  the goal's nominal room, so success and completed crossings are distinct events.
- BFS reference length, fraction whose reference path fits the horizon, success
  conditional on those feasible pairs, and successful path length / BFS length.

BFS uses the existing `build_graph(env,20)` and multi-source `bfs_distances` with
the **coordinate-wise goal region**, not just the exact goal node. It is a
noiseless restricted-lattice comparison, not an exact optimum for noisy continuous
movement. Roundoff at inclusive threshold boundaries can shift reference lengths
slightly. Noise may sometimes beat the lattice reference, so ratios below 1 are
possible. The 150-step diagnostic distinguishes short-horizon censoring from
persistent navigation failure without changing training or the 30-step result.

Four automated tests passed: config/dynamics/action preservation, F shape and
objective mode, deterministic stratified pairs/BFS, native-motion doorway and wall
counts, and denominator/BFS-feasibility handling. A two-epoch, 80-update/240-
transition smoke run completed with finite losses at `runs/smoke_seed0`. Its tiny
four-pair-per-category evaluation is only validation, not a performance estimate.

## Interpreting the comparison

If same-room success is high while cross-room success stays low, inspect completed
crossings, wall-hit rates and BFS-feasible subsets. If extending the horizon fixes
cross-room success, insufficient time is part of the problem. If local success
also fails, do not attribute the deficit solely to doorways. Successful path
ratios alone have survivor bias; always inspect success rates alongside them.

The main configuration changes relative to the defaults in the existing Four
Rooms runner are the released B regulariser and zero-task probability 0 instead
of .25. This joint comparison tests configuration as a group; it does **not**
isolate which of those changes matters. Any later single-factor ablation should
use a separate run/config, after this full experiment is evaluated. No additional
training branches have been launched.
