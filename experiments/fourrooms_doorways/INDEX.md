# Four Rooms doorway artifact index

For experiment settings, numerical results and interpretation, start with the
[experiments and results overview](../README.md).

## Full runs

All four runs completed 800 epochs, with retained model checkpoints at epochs
200, 400, 600 and 800. The original-width reference is stored separately in
`../fourrooms_reference/runs/seed0/`.

| Run folder under `runs/` | Condition |
| --- | --- |
| `width2_seed0` | Cauchy tasks, continuing objective |
| `width2_replay_continuing_seed0` | Replay-goal tasks, continuing objective |
| `width2_replay_stopped_seed0` | Replay-goal tasks, stopped bootstrap |
| `width2_replay_continuing_threshold025_seed0` | Replay-goal tasks, continuing objective, goal tolerance 0.025 |

## Saved analyses

| Folder under `analyses/` | Contents |
| --- | --- |
| [goal_reaching](analyses/goal_reaching/REPORT.md) | Initial matched Cauchy/stopped/random comparison |
| [replay_continuing_comparison](analyses/replay_continuing_comparison/REPORT.md) | Three-model comparison and replay-continuing learning curves |
| [replay_continuing_failures](analyses/replay_continuing_failures/REPORT.md) | Diagonal failure recordings, animations and collision maps |
| [replay_threshold025_comparison](analyses/replay_threshold025_comparison/REPORT.md) | Both replay-continuing models evaluated at tolerance 0.025 |
| [replay_threshold025_failures](analyses/replay_threshold025_failures/) | Strict-model diagonal failure recordings |
| [replay_threshold025_old_failures](analyses/replay_threshold025_old_failures/) | Original model's failures under the same strict tolerance |
| [implicit_reward_wall_traps](analyses/implicit_reward_wall_traps/REPORT.md) | Reward maps, Q/MC comparisons, covariance audit and detour controls |
| [advantage_stability](ADVANTAGE_STABILITY.md) | Fixed-doorway action ranks and forced-action Monte Carlo |
| [decline_analysis](analyses/decline_analysis/REPORT.md) | Cauchy late decline, wall loops and action-ranking diagnostics |
| [stage_analysis](analyses/stage_analysis/REPORT.md) | Matched approach, crossing and second-door probes |
| [goal_validity](GOAL_VALIDITY.md) | Goal/state geometry, connectivity and budget audits |
| threshold025_validation | Acceptance-region audit, validation config and small evaluator checks |

[README.md](README.md) and [THRESHOLD025.md](THRESHOLD025.md) contain the original
experiment setup notes. Historical reports may describe removed launchers or
smoke-run folders. The [current evaluation guide](../../docs/EVALUATION_TOOLS.md)
lists supported package commands. Smoke-run outputs and redundant import-conflict
archives were removed; real run outputs and their source snapshots remain.

`organization_manifest.json` and `analysis_path_migrations.json` describe earlier
file moves. Their historical paths are provenance, not a current file inventory.
