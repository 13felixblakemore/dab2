# Adjustable Four Rooms doorways

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
