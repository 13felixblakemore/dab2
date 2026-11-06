# Adjustable Four Rooms doorways

## Replay-goal sampling with stopped bootstrapping

New separate variant: replay-sampled raw goal observations, task
`z=stop_gradient(B(g_replay))`, and zero continuation on next-state goal entry.
Upload the updated `forward_backward_jax.py` and
`replay_goal_terminated.json` before running this command on the cluster:

```bash
python run_fourrooms_doorways.py \
  --config experiments/fourrooms_doorways/replay_goal_terminated.json \
  --device cuda --doorway-scale 2 \
  --checkpoint-every 200 --recovery-every 200 --log-every 20 \
  --seed 0 --output experiments/fourrooms_doorways/runs/width2_replay_stopped_seed0
```

This is 800 epochs / 800,000 updates and retains the authors' released B
regulariser, zero-task mixing 0 and no clipping. Random action collection,
environment dynamics and evaluation pairs are unchanged. No full variant run is
launched by preparing the configuration.

The configuration selects `z_sampling="replay"`, `goal_terminated=true`,
`goal_threshold=.1`, `goal_norm="linf"`. For each independently sampled replay
goal, the entire corresponding FB bootstrap row is set to zero if
`all(abs(s_next-g)<=.1)`, exactly matching the Four Rooms success convention.
Other rows bootstrap normally. The loss's immediate term is unchanged; this is
a pre-entry stopped-bootstrap variant, not an imposed zero value for states
already inside the goal. It deliberately differs from the authors' continuing
Cauchy-task experiment.

The configurable norm preserves the historical Euclidean mask by default
(`goal_norm="l2"`); only this variant opts into the environment-compatible square
predicate. Twelve regression/parity tests passed, including the new square-mask
case, and a two-epoch compiled smoke run exercises replay sampling plus the mask.
Recovery every 200 epochs can lose up to 199 epochs after interruption.

## Cluster I/O and timing

The runner prints only once per epoch in its original form, but its original
rolling recovery saves copied full replay/model state to host and disk every
epoch. On shared cluster storage that can cause pauses independent of printing.
Evaluation and JIT compilation also cause expected spikes at scheduled epochs;
the cluster has not been profiled, so these are potential causes, not a measured
diagnosis.

For fewer writes and progress messages in a new run:

```bash
python run_fourrooms_doorways.py --device cuda --doorway-scale 2 --checkpoint-every 200 --recovery-every 200 --log-every 20 --seed 0 --output experiments/fourrooms_doorways/runs/width2_lowio_seed0
```

`--log-every 0` suppresses progress prints entirely. Its default is 10.
`--recovery-every` defaults to 1 to preserve the original crash-recovery behavior;
setting it to 200 means interruptions may lose up to 199 epochs. Recovery is also
always saved at retained model checkpoints and the final epoch. All epoch losses
and metrics remain in memory and are flushed at logging/recovery epochs. Training
hyperparameters, target updates, fixed pairs and evaluation cadence are unchanged.

Progress records include `train_seconds` (synchronized training, including JIT on
first invocation), `checkpoint_seconds`, `evaluation_seconds` and
`recovery_seconds`. `seconds` covers those stages, excluding the small final
metrics/console writes. Compare these fields to identify the source of pauses.
Changing recovery/log cadence does not reduce arithmetic or change learning.
The updated flags require the updated script; existing running Python jobs do
not pick up edits. Existing runs' strict source checks require their archived
script versions for resume; do not overwrite an active output directory.

Run from the project root (800 epochs, released B regulariser, continuing FB,
zero-task probability 0, no gradient clipping, all other reference settings):

```powershell
& ./.venv/Scripts/python.exe run_fourrooms_doorways.py --doorway-scale 2 --checkpoint-every 200 --seed 0 --output experiments/fourrooms_doorways/runs/width2_seed0
```

`--doorway-scale` multiplies **all four doorway widths** around their existing
centers. Original width is 1/11 of environment size, approximately .0909 in the
unit square. Scale 2 gives width 2/11, approximately .1818. Valid values are
strictly between 0 and 3; this keeps doorways inside the boundary and leaves wall
segments with positive lengths. Scale 1 preserves the original seven wall blocks
exactly. Wall thickness, wall intersection, room geometry outside the extended
openings, step .05, noise .005, four actions and goal tolerance .1 are preserved.

Examples: scale 1.5 gives width .1364, scale 2 gives .1818, scale 2.5 gives .2273.
Use distinct output directories for each width/seed. Resolved configuration,
actual wall block coordinates, source snapshots and geometry preview
`environment.png` are saved in each output directory. No full larger-doorway run
was launched automatically; the command above starts one.

Retained model files are `epoch_0200.pkl`, `epoch_0400.pkl`, `epoch_0600.pkl`, and
`epoch_0800.pkl`. The final epoch is always retained, including explicitly shorter
configs. An atomic `resume.pkl` is overwritten every epoch and contains matching
model/target/optimizer state, replay, losses and RNG streams; it is a recovery
snapshot, not an accumulation of per-epoch models. All scalar losses remain in
`losses.npy`, with NaN placeholders for unfinished epochs. This avoids the earlier
every-epoch checkpoint disk growth. Old run outputs are never deleted.

Resume the same run using exactly the same options plus `--resume`:

```powershell
& ./.venv/Scripts/python.exe run_fourrooms_doorways.py --doorway-scale 2 --checkpoint-every 200 --seed 0 --output experiments/fourrooms_doorways/runs/width2_seed0 --resume
```

Evaluation is at epoch 1 plus the retained checkpoint epochs. It uses the same
600 baseline room-interior endpoint pairs (200 per task category) across widths,
seeds and checkpoints, with fixed noise/action keys, online B/F, epsilon=.02,
horizons 30/150 and native dynamics. BFS lengths are recomputed for the chosen
geometry. This holds evaluation tasks fixed even though wider doorways alter
training free-space sampling. The reference training and learner configs are
unchanged; the resolved config records the new width and checkpoint interval.
The automatic evaluation uses the correct adjustable environment. The older
standalone `fourrooms_reference_diagnostics.py` CLI assumes original doorways
and should not be used to evaluate a wider-doorway model.

This implementation is in `adjustable_four_rooms.py` and
`run_fourrooms_doorways.py`. Existing environment/runner files and the active
baseline Four Rooms experiment are unchanged, including that active process's
original checkpoint schedule. For a new original-width run with reduced checkpoint
frequency, use this new runner with `--doorway-scale 1` and a fresh output path.

Validation:

```powershell
& ./.venv/Scripts/python.exe -m unittest test_adjustable_four_rooms test_fourrooms_reference -v
```

Eight tests cover exact scale-1 compatibility, expanded openings allowing formerly
blocked movement, invalid widths, fixed endpoints/BFS recomputation and the
200/400/600/800 save schedule, alongside the existing navigation checks. A short
two-epoch width-2 run is saved separately at
`experiments/fourrooms_reference/runs/doorways_smoke_seed0`.
