# Tests

Run from the repository root:

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
```

The suite is grouped by behavior. Each file keeps separate test classes for
the components it covers.

| Module | Coverage |
| --- | --- |
| `test_fourrooms` | Doorway geometry, goal tolerances, fixed pairs, validity audits, reference evaluation and report denominators |
| `test_goal_evaluation` | Native goal rollouts, collision masks, legacy PyTorch checkpoint loading and device portability |
| `test_fb_training` | Bellman loss oracle, goal termination, gradients, compiled training, target updates, checkpoint compatibility, import-safe training entry points and sweep planning |
| `test_bottlenecks` | BFS controls, FB/MC estimators, exclusions, policy probabilities, temperature sweeps and shared random numbers |
| `test_navigation_diagnostics` | Wall masks, collision streaks, doorway advantages, failure stages and remaining rollout budgets |
| `test_reward_diagnostics` | Noisy Bellman backups, forced-action returns, continuing rewards, collector equivalence and failure classifications |
| `test_continuous_maze` | Maze motion, walls, noise, goal detection, evaluation and bootstrap diagnostics |

Run one topic or class, for example:

```powershell
python -m unittest tests.test_fourrooms -v
python -m unittest tests.test_fb_training.GoalTerminationTests -v
```

The consolidation removed a standalone room-distance lookup test; the navigation
tests exercise room progress through failure-stage classification. External-code
maze parity tests depended on the deleted `experiments/continuous_maze/reference`
tree and experiment configuration. They were replaced with self-contained motion
and geometry checks. Exact parity with the released third-party implementation
is no longer checked. Continuing target updates and checkpoint round-trips remain
covered with a small in-memory learner.

Sweep tests import the active package directly. Saved run directories, models,
and archived source snapshots are not required; the Four Rooms tests and sweep
planner use the checked-in experiment configurations.

The Four Rooms and bottleneck suites also check dispatch and argument handling
for the consolidated evaluation commands listed in
[the evaluation guide](../docs/EVALUATION_TOOLS.md).
