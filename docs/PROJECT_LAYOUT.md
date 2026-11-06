# Project organization

The cleanup groups active implementation code by purpose and preserves saved
experiment data. `project_organization_20261106.json` records every file move,
source hashes before/after relocation, and hashes of protected experiment files.

| Area | Location |
|---|---|
| JAX learner | `fb_navigation/agents/forward_backward_jax.py` |
| PyTorch learner | `fb_navigation/agents/forward_backward.py` |
| Original Four Rooms environments | `fb_navigation/environments/continuous_four_rooms*.py` |
| Adjustable doorways | `fb_navigation/environments/adjustable_four_rooms.py` |
| Continuous Maze | `fb_navigation/environments/continuous_maze.py` |
| Training and sweep implementations | `fb_navigation/training/` |
| Goal/action-value evaluators | `fb_navigation/evaluation/` |
| Reward, wall, doorway, and bottleneck diagnostics | `fb_navigation/diagnostics/` |
| Test modules | `tests/` |
| Notebook | `notebooks/forward_backwards_4rooms.ipynb` |
| Historical unfinished DAB draft | `archive/drafts/dab.py` |

## Compatibility

The root learner/environment modules forward imports to the canonical package,
including private helpers and class definitions used by older scripts and
checkpoints. Root training entry points and `evaluate_width2_goals.py` forward
execution and imports as well. Both forms work:

```bash
python run_fourrooms_doorways.py --help
python -m fb_navigation.training.run_fourrooms_doorways --help
```

The dated analysis commands under `experiments/maintenance/` and the existing
`experiments/fb_width2_lr_polyak_sweep_20261106.py` path are compatibility
launchers. Their implementations are in the package. Other diagnostics now run
as modules, for example:

```bash
python -m fb_navigation.diagnostics.diagnose_fb_walls --help
python -m fb_navigation.diagnostics.record_replay_failures_20261106 --help
python -m fb_navigation.training.fb_width2_lr_polyak_sweep_20261106 --help
```

Use qualified package imports in new code. `fb_navigation.paths.PROJECT_ROOT`
locates configuration/data files independently of module location;
`source_path(name)` resolves historical source basenames to the active source
used for new run snapshots. Existing archived source files are not rewritten.

## Results and source-hash guards

Checkpoints, saved configurations, training/evaluation arrays, analysis reports
and archived run sources retain their paths and contents. Active method notes
and experiment instructions have updated runnable commands. Models remain in
`models/` or their existing experiment run folders. The migration checksum record
verifies protected files against their pre-cleanup state.

New runs snapshot the canonical implementation rather than the compatibility
launcher. Older runs retain strict source-hash resume checks. Restoring an old
run requires its archived source in the original layout; the organization
change does not waive those checks. Evaluations with source-hash manifests
should use fresh output folders when executed with the reorganized source.

## Validation

The pre-cleanup baseline passed 82 tests. Use the complete organized suite:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Validation logs are kept under `docs/validation/`. The unfinished `dab.py` was
already syntactically incomplete; it is archived unchanged and excluded from
active modules and test collection.

The complete suite passed all 82 tests after the move. Seven CLI entry points,
the notebook setup cell, identical predictions from the saved epoch-800 model,
and a two-epoch training/save/load smoke run also passed. New source snapshots
were checked against their canonical package files, including `paths.py` and
the saved source-location map. Details are in
`validation/organization_checks.json`.
