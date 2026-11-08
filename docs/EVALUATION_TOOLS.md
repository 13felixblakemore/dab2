# Evaluation and diagnostics

The active package has four evaluation modules and twelve diagnostic modules.
Run commands from the repository root using the project's Python environment.

| Tool (under `fb_navigation`) | Purpose |
| --- | --- |
| `evaluation.fourrooms reference` | Evaluate a checkpoint on fixed pairs; shared training-time evaluation helpers |
| `evaluation.fourrooms width2` | Compare width-2 models with matched endpoints and goal tolerances |
| `evaluation.evaluate_original_goals_jax` | Native continuous goal benchmark for JAX checkpoints |
| `evaluation.evaluate_torch_goals` | Corresponding benchmark using native PyTorch inference |
| `evaluation.evaluate_fb_action_values` | Bellman backups, action values and Monte Carlo returns |
| `diagnostics.bottleneck_policies compare` | Matched BFS/FB policy and discount comparison |
| `diagnostics.bottleneck_policies epsilon` | Epsilon-greedy bottleneck comparison and doorway arrivals |
| `diagnostics.control_bfs_bottlenecks` | BFS graph and control rollouts |
| `diagnostics.estimate_fb_bottlenecks_jax` | Vectorized FB and Monte Carlo bottleneck estimators |
| `diagnostics.sweep_fb_temperature` | Policy temperature comparisons with common random numbers |
| `diagnostics.audit_doorway_goals` | Goal geometry, wall overlap and reachability audits |
| `diagnostics.diagnose_fb_walls` | Wall masks, collision streaks and trajectories |
| `diagnostics.investigate_doorway_decline` | Navigation failure stages and paired probes |
| `diagnostics.compare_navigation_stages` | Probe comparisons with the remaining rollout budget |
| `diagnostics.track_doorway_advantages` | Action advantages and forced-action policy comparisons |
| `diagnostics.analyse_fb_rewards_failures` | Reward/value calibration and failure classification |
| `diagnostics.diagnose_continuous_maze` | Held-out maze transitions and bootstrap residuals |
| `diagnostics.learned_rewards` | Reward bounds and continuing learned-reward rollouts; library only |

```powershell
python -m fb_navigation.evaluation.fourrooms reference --help
python -m fb_navigation.evaluation.fourrooms width2 --help
python -m fb_navigation.diagnostics.bottleneck_policies compare --help
python -m fb_navigation.diagnostics.bottleneck_policies epsilon --help
```

The root command launchers have been removed. Use the package commands above.

The five dated `*_20261106.py` diagnostic report generators were removed. Saved
reports, arrays, figures, checkpoints and archived source copies remain in their
experiment directories. Commands recorded in historical reports describe the old
layout and are no longer active package entry points. The reusable learned-reward
bounds and rollout kernel remain in `diagnostics.learned_rewards`.

New source snapshots resolve the former evaluator basenames through
`fb_navigation.paths.source_path`. Older run hashes still describe their original
archived sources; resuming those runs requires the matching sources.
