# Width 2: replay-continuing FB with goal tolerance 0.025

The old coordinate-wise success tolerance is 0.1; wall thickness is 1/11,
approximately 0.09091. For goals sufficiently close to a wall, the old success
square can extend into free space on its far side. A regression test reproduces
this with two free points on opposite sides of the central vertical wall.
Tolerance 0.025 rejects that success. It is half the movement step (0.05) and
smaller than the full wall thickness.

The audit of the existing 6,400 diagonal traces found 42 of 2,357 successes at
30 steps, and 62 of 3,545 successes at 150 steps, with a wall intersecting the
straight segment between the stopping position and goal. These include corner
occlusion; they are not all opposite-full-wall cases. Audit counts are saved in
`analyses/threshold025_validation/old_success_audit.json`. All goals remain
reachable; this issue concerns the acceptance region.

The new experiment changes only the active environment goal tolerance, from
0.1 to **0.025**. The learner configuration, RBF configuration, geometry,
actions, noise, collection budget and update budget match the previous
replay-continuing run. Defaults for existing environments and runs remain 0.1.
Goal detection still uses coordinate-wise distance; this experiment adds no
line-of-sight requirement, so a small acceptance region can still span a corner.

The continuing FB loss does not use the goal threshold directly. The collector
does reset position and goal after reaching its randomly sampled collection
goal; a smaller threshold changes those resets and hence the replay data.
Evaluation also uses the smaller threshold. This tests a stricter acceptance
rule and its collection consequences, rather than introducing a wall penalty
or changing the objective to a goal-terminated loss.

## Train on the cluster

From the repository root, using a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES=0 python run_fourrooms_doorways.py \
  --config experiments/fourrooms_doorways/replay_continuing_threshold025.json \
  --device cuda --doorway-scale 2 \
  --checkpoint-every 200 --recovery-every 200 --log-every 20 \
  --seed 0 \
  --output experiments/fourrooms_doorways/runs/width2_replay_continuing_threshold025_seed0
```

This runs 800 epochs, 25 cycles per epoch, 40 updates per cycle: **800,000
updates**. Four 30-step collection episodes per cycle give **2.4 million
transitions**. Keep models at epochs 200, 400, 600 and 800. The same 600
start-goal pairs are used at each evaluation, with BFS lengths recomputed for
the smaller acceptance region; online F/B, epsilon 0.02, and 30/150-step
horizons are unchanged. Checkpoint evaluations record success, wall-action
percentage, successful path length, doorway crossings, and BFS-relative results.

The learner still uses replay-goal z sampling, d=50, learning rate 0.0005,
Polyak 0.95 per cycle, the released B regularizer, zero_z_p=0, no gradient
clipping, and the continuing FB objective.

## Compare both final models under the same stricter rule

Do not compare the new run's strict score directly with the old run's 0.1 score.
After training, evaluate both checkpoints at **0.025** with identical start-goal
pairs and action/noise keys:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_width2_goals.py \
  --device cuda --goal-threshold 0.025 \
  --models width2_replay_continuing_seed0 width2_replay_continuing_threshold025_seed0 \
  --pairs-per-category 200 --rollouts 32 \
  --output experiments/fourrooms_doorways/analyses/replay_threshold025_comparison
```

This reports same-room, adjacent-room and diagonal success, mean successful
steps, wall-action percentage, repeated-wall-action percentage and BFS
feasibility at 30/150 steps. Epsilon 0.02, greedy and random diagnostics remain
separate. Results and common threshold are saved in `results.json` and
`config.json`; the comparison plot is `success.png`.

To obtain a strict baseline before the new model finishes, use the same command
with only `--models width2_replay_continuing_seed0` and a separate output folder,
such as `analyses/replay_threshold025_old_baseline`.

The new model's failed diagonal trajectories can then be recorded with the same
protocol and correctly sized success regions:

```bash
python experiments/maintenance/record_replay_failures_20261106.py \
  --run experiments/fourrooms_doorways/runs/width2_replay_continuing_threshold025_seed0 \
  --evaluation experiments/fourrooms_doorways/analyses/replay_threshold025_comparison \
  --output experiments/fourrooms_doorways/analyses/replay_threshold025_failures \
  --device cpu
```

## Validation completed

- Ten environment/navigation tests passed, including a cross-wall success
  regression, unchanged dynamics, unchanged start-goal endpoints, stricter BFS
  targets, invalid thresholds, and retained checkpoint schedule.
- The full new learner/RBF configuration matches the previous run exactly.
  Budget arithmetic was checked explicitly.
- A reduced two-epoch training and checkpoint-evaluation run completed with
  finite losses in `smoke_runs/threshold025_seed0`; its input configuration is
  `analyses/threshold025_validation/smoke_input.json`.
- The updated standalone evaluator completed epsilon-greedy, greedy and random
  checks on the old checkpoint at threshold 0.025 in
  `analyses/threshold025_validation/evaluator_smoke`. This tiny check is only a
  pipeline test, not a performance estimate.

The full 800-epoch experiment has not been launched locally.
