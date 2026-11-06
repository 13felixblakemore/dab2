# Results and run status

The full seed-0 run was launched in a hidden background process on 2026-11-06.
It is **not yet an achieved replication result**. See `runs/seed0.process.json`
for its launch PID, `runs/seed0.stdout.log` / `seed0.stderr.log` for live output,
and `runs/seed0/training.csv` for completed epochs. Its configured target is
800 epochs, 800,000 optimizer steps and 2.4 million collected transitions.
`runs/seed0/final.json` and plots are written after the complete run.
No full-run success rate or claim of matching Figure 9 is available at handoff.

This installation exposes `CpuDevice(id=0)` only. The 10-epoch benchmark took
6.4–7.2 seconds per steady-state epoch plus compilation, checkpointing and final
evaluation overhead. Budget roughly 90 minutes for 800 epochs; actual runtime
depends on machine load. Every epoch retains its model (roughly several GB for
800), and the latest replay snapshot supports recovery after interruption.

Validation: **17 tests passed** (7 new parity/FB tests and 10 existing
goal-termination tests), with no algorithm hyperparameter tuning. The parity
tests compare 1,000 noisy environment transitions and explicit deterministic
collision/boundary/endpoint cases to the downloaded source; 1,000 random geometry
cases; STAY noise statistics; RBF values; F/B shapes; Cauchy norm statistics;
the actual released loss and every parameter gradient; target timing; and
checkpoint reload behavior. The latest run completed in 16.559 seconds.

Completed seed-0 **10-epoch diagnostic** (`runs/short_seed0`):

| Policy | Successes / 1,000 | Success rate | Mean successful steps | Rejected wall moves |
| --- | ---: | ---: | ---: | ---: |
| Paper evaluation, epsilon=.02 | 288 | 28.8% | 6.229 | 3,650 |
| Greedy diagnostic | 289 | 28.9% | 6.291 | 3,872 |
| Random diagnostic | 105 | 10.5% | 13.610 | 976 |

This is 10,000 gradient updates and 30,000 transitions, **1/80 of the training
budget**. Its last 20-goal evaluation was 35%; small epoch evaluations fluctuate
considerably. All recorded losses and diagnostic components were finite. Target
policy entropy went from 1.609 to 1.534 nats (maximum log(5)=1.609); diagnostic
B norms grew from .97 to 1.30 and F norms from 8.3 to 106.0. Increasing F norms
and negative total losses alone do not establish divergence for this objective.
Trajectories show boundary-seeking and repeated wall attempts at this early stage.
Greedy and epsilon=.02 perform similarly, so evaluation exploration alone does
not explain the weak short-run result. Training is still severely truncated.

The paper's Figure 9 FB-50 bar is approximately 83% by visual inspection, after
800 epochs and across three training seeds. Comparing that bar directly to this
10-epoch diagnostic would confound training duration with implementation quality.
The full run must finish before diagnosing any final performance gap. Relevant
remaining differences and a source-driven diagnostic checklist are in README.md.

The earlier one-epoch `runs/smoke_seed0` is retained as an initial validation
artifact. It predates per-trial independent evaluation seeding and is not used
in the comparison table. `short_seed0/source/` archives the sources actually used
for that short run (verified against its metadata hashes); its plot labels predate
the shorter final labels. These historical outputs have not been overwritten.

To inspect the full run from PowerShell:

```powershell
Get-Content experiments/continuous_maze/runs/seed0.stdout.log -Tail 2
Get-Content experiments/continuous_maze/runs/seed0.stderr.log -Tail 10
Get-Process -Id (Get-Content experiments/continuous_maze/runs/seed0.process.json | ConvertFrom-Json).pid
```

If the process has stopped before epoch 800, use the README's `--resume` command
with the unchanged full config and archived source. Do not start a second process
in the same output directory while the first one is running.
