# Experiment directory

- `fb_width2_lr_polyak_sweep_20261106.py`: GPU sweep of classic Cauchy FB
  dimension, learning rate and Polyak weight; usage in its matching `.md` file.

- `fourrooms_doorways/INDEX.md`: canonical width-2 runs and all doorway analyses.
- `fourrooms_doorways/analyses/goal_reaching/REPORT.md`: newest independent
  goal-reaching comparison of Cauchy continuing and replay/stopped runs.
- `fourrooms_reference/README.md`: original-width reference-configuration run.
- `continuous_maze/README.md`: authors' Continuous Maze reproduction.

Full runs are separated from smoke runs. Models remain ignored by Git, while
configs, metrics and analysis reports can be versioned. Archived `source/` folders
inside runs preserve the code that actually generated those checkpoints.
`fourrooms_doorways/organization_manifest.json` records imported-file moves,
SHA-256-verified duplicate removal and archived differing copies. Entry-point
scripts and notebook imports remain at the repository root for compatibility.
