# Classic FB dimension × learning-rate × Polyak sweep

Unique entry point: `experiments/fb_width2_lr_polyak_sweep_20261106.py`.
Uses the existing JAX learner/compiled trainer, not a new FB implementation.
**Classic Cauchy task sampling, continuing objective, no goal termination.**
Replay/stopped configurations are rejected to avoid mixing experimental questions.

From the project root on the GPU cluster:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/fb_width2_lr_polyak_sweep_20261106.py
```

The default grid has **18 full runs**, sequentially on the selected GPU:

| Parameter | Values |
| --- | --- |
| Dimension d | 50, 100 |
| Constant Adam learning rate | 0.0005, 0.0001, 0.00005 |
| Polyak old-target weight | 0.90, 0.95, 0.99 |
| Training seed per combination | 0 |
| Budget per model | 800 epochs, 800,000 updates, 2.4 million transitions |

Total default budget: **14.4 million gradient updates and 43.2 million collected
transitions**. This is a full training sweep, not 18 short tests. No full sweep
has been launched locally. Plan cluster wall time accordingly.

Fixed settings: doubled doorways, four actions, native noise/dynamics and goal
tolerance; three 256-unit ReLU layers, normalized 441-RBF inputs, Cauchy tasks,
gamma=.99, T=200, batch 128, released authors' B regulariser coefficient 1,
zero-task mixing 0, no clipping, uniform random action collection. All models
start from scratch; there is no late-epoch learning-rate schedule or warm start.
The original `(d=50, lr=.0005, polyak=.95)` control is included on the same stack.

Polyak means `target <- p*target + (1-p)*online`, **once per cycle after 40
updates**. Thus p=.99 tracks more slowly than .95; .90 tracks faster. Each
learning rate is constant for the entire run. The grid tests alternatives; it
does not assume that reducing learning rate will improve performance.

## Smaller or expanded sweeps

For dimension 50 only (nine full runs):

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/fb_width2_lr_polyak_sweep_20261106.py \
  --dimensions 50 \
  --output experiments/sweeps/fb_width2_d50_lr_polyak_20261106
```

Customize any axis, using a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/fb_width2_lr_polyak_sweep_20261106.py \
  --dimensions 50 100 --learning-rates 0.0001 0.00005 \
  --polyaks 0.95 0.99 --seeds 0 \
  --output experiments/sweeps/fb_width2_stability_replications
```

Use `--prepare-only` to write configs/source snapshots and print cell indices
without starting training. The launcher itself does not initialize JAX/GPU;
each child process releases its device allocations when it exits.

## Outputs, comparison and recovery

Default output: `experiments/sweeps/fb_width2_lr_polyak_20261106/`:

```text
sweep.json                    Immutable grid, configs and source hashes
configs/                      One resolved configuration per run
source/                       Sweep launcher and original training code
runs/d50_lr0.0001_polyak0.95_seed0/
  epoch_0200.pkl ... epoch_0800.pkl
  resume.pkl                  One rolling full recovery state
  evaluation_pairs.npz        Same fixed pairs across all cells and seeds
  evaluation/                 Epoch 1/200/400/600/800 room-group metrics
  stdout.log, stderr.log      Per-run logs
goal_reaching.csv              All checkpoint/task/horizon measurements
final_goal_reaching.csv        Final-epoch measurements only
diagonal_learning_curves.png   Compare long-range navigation
adjacent_learning_curves.png   Compare one-door navigation
status.json                   Completed/incomplete run counts
```

Evaluation uses the **same 600 start-goal pairs**, fixed noise keys, online B/F,
epsilon=.02, and horizons 30/150 across all cells. Metrics include same/adjacent/
diagonal success, path lengths, collisions, doorway crossings and BFS references.
The fixed evaluation set is a sweep validation set: use new goals and multiple
training seeds for a final selected-model claim. The per-checkpoint evaluation is
one noisy rollout per pair, not the later 32-replica independent test suite.

Checkpoint and recovery defaults are every 200 epochs; logging defaults to every
20. Interruptions can lose up to 199 epochs. Resume the recorded grid without
repeating its axis flags:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/fb_width2_lr_polyak_sweep_20261106.py --resume
```

Completed cells are checked and skipped. An unfinished cell resumes from its
matching recovery, with source/config checks. If a job stopped before its first
recovery, `--resume --restart-without-recovery` archives that partial cell and
restarts only it; other finished cells are preserved. This does not silently
overwrite experiment data. Preserve source snapshots when moving between stacks.

Refresh summary CSVs/plots without loading models or training:

```bash
python experiments/fb_width2_lr_polyak_sweep_20261106.py --summarize-only
```

For custom output roots, supply the same `--output` to resume/summarize. Curves
average available seeds; inspect `status.json` and finish the intended cells
before comparing configurations with different seed counts.

## Optional GPU job arrays

Prepare the full manifest on the cluster first:

```bash
python experiments/fb_width2_lr_polyak_sweep_20261106.py --prepare-only
```

Then each separately GPU-allocated array job can run one cell:

```bash
python experiments/fb_width2_lr_polyak_sweep_20261106.py --run-index "$SLURM_ARRAY_TASK_ID"
```

The default indices are 0–17. Each job inherits the cluster-assigned
`CUDA_VISIBLE_DEVICES`; do not force every parallel job onto physical GPU 0.
Do not launch the same cell twice. After all jobs finish, run `--summarize-only`.
Array jobs avoid writing shared summaries concurrently.

## Validation

Three automated tests cover the grid, dimension/config isolation, rejection of
non-classic FB, input validation and child resume/device/seed arguments. Local
CPU smoke runs exercise the actual compiled training runner and summary export;
the smoke flag explicitly reduces each cell to two epochs/80 updates and four
pairs per room category. It never substitutes that budget for a full run.

```bash
python -m unittest test_fb_sweep_20261106 -v
python experiments/fb_width2_lr_polyak_sweep_20261106.py \
  --smoke --dimensions 50 100 --learning-rates 0.0001 --polyaks 0.95 --seeds 0 --device cpu
```
