# Four Rooms doorway experiments

Start here. Uploaded `fourrooms_doorways/fourrooms_doorways/...` directories have
been merged into the canonical layout below. All distinct checkpoint content was
preserved. Byte-identical duplicate files were removed after SHA-256 comparison;
differing copies were archived instead of overwritten. Exact operations and
hashes are in `organization_manifest.json`.

```text
fourrooms_doorways/
  runs/
    width2_seed0/                 Original Cauchy, continuing FB; epochs 200/400/600/800
    width2_replay_stopped_seed0/  Replay tasks + goal-entry bootstrap mask; same epochs
  analyses/
    goal_reaching/               NEW: fresh matched-goal tests, results and success.png
    advantage_stability/         Fixed doorway advantages and action-conditioned MC
    decline_analysis/            Late decline, wall loops and oracle-mask diagnostics
    stage_analysis/              First approach / crossing / second-door comparisons
    goal_validity/               Goal/state wall and connectivity audits
  smoke_runs/                    Small validation runs, separate from full experiments
  archive/import_conflicts/      Differing versions found in repeated uploads
  replay_goal_terminated.json    Runnable 800-epoch variant config
  README.md                     Training commands and experiment settings
```

Reports in this folder: `RESULTS.md` (original-width comparison),
`ADVANTAGE_STABILITY.md`, `GOAL_VALIDITY.md`. Detailed stage/decline reports remain
beside their corresponding analysis outputs. The newest goal-reaching report is
`analyses/goal_reaching/REPORT.md`.

Checkpoint, recovery, config, evaluation and archived-source filenames inside
each full run are unchanged. Old manifests/source snapshots retain historical
paths as provenance; use the organization manifest to resolve moved artifacts.
Root analysis scripts now default to the canonical paths. The path-only changes
to stage/advantage analysis manifests are recorded in `analysis_path_migrations.json`;
their untouched previous manifests are `config.before_organization.json` beside
each analysis. Mathematical settings and checkpoint hashes are unchanged. Other
suite changes still require a fresh output directory. Training resume also
requires its matching archived source, as before.

Reproduce the new goal-reaching test:

```bash
python evaluate_width2_goals.py
# Optional GPU execution in a fresh analysis output:
python evaluate_width2_goals.py --device cuda --output experiments/fourrooms_doorways/analyses/goal_reaching_gpu
```

This evaluates both final checkpoints, not new training. Source modules and
notebook imports remain at the repository root to preserve compatibility.
Python caches are ignored and removed from Git tracking; checkpoint ignores
remain in place. Smoke data and old reports are not used as final performance.
