# Evaluation protocol audit

Audited the running seed-0 source snapshot (`runs/seed0/source/`) and resolved
`runs/seed0/config.json` against the archived released code at commit
`cea625688c7e9680bb98d82339d702758d812b3e` and paper Appendix D / Figure 9.
No training sources or running-process configuration were modified for this audit.

Completed fixed-suite diagnostics during this audit:

| Epoch | Success / 1,000 | 95% trial CI | Successful steps | Wall-hit actions | Bootstrap RMSE | Relative RMSE | Mean Q gap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 100 | 86.3% | 84.0–88.3% | 8.287 | 22.96% | 15.05 | .0991 | 20.66 |
| 200 | 87.1% | 84.9–89.0% | 8.489 | 16.94% | 17.87 | .0777 | 22.35 |

These checkpoint results are encouraging but do not replace the 800-epoch,
three-seed comparison. Overlapping marginal trial intervals and only two evaluated
checkpoints do not establish a learning plateau or a significant improvement.
The raw/relative residuals move in opposite directions, illustrating the scale
confound. Checkpoints 400/600/800 were unavailable when this suite first ran.

| Item | Authors' released behavior | Running JAX reproduction | Finding |
| --- | --- | --- | --- |
| Initial state | `env.reset()` -> `sample_point()`, uniform [0,1]^2 with exact-on-wall rejection | Same sampling/rejection using local RandomState | Distribution matches |
| Goal | Independent second `sample_point()` from reset | Independent second draw | Distribution matches; no distance filtering or fixed start |
| Success | `Point.is_close_to`, both absolute coordinate errors <= .05 | `goal_reached`, same inclusive comparisons | Matches; square goal region, not Euclidean radius |
| Horizon | `continuous_main.get_env_params`: 30 timesteps | Config `steps=30` | Matches |
| Early stopping | Test reward after each step, break on positive reward | Same; no goal check at time zero | Matches, including initial-in-goal edge case |
| Task | Online `backward_network(RBF(g))` | `learner.embed_B(g)` uses `state['params']['b']` | Matches; no normalization of B(g) before scoring |
| Task preprocessing | F internally bounds z by sqrt(1+norm(z)^2/d); score uses original z | Same ForwardNet behavior | Matches |
| Action | `random.random()<.02` -> uniform `randrange(5)`, otherwise argmax F(s,z)^T z | Same epsilon mixture and first-index argmax | Matches; random branch includes greedy action |
| F network | `act_e_greedy` calls `act` with default `target_network=False` | `act` uses `state['params']['f']` by default | **Online F**, matches |
| B network | `self.backward_network`, not backward target | Online B | **Online B**, matches |
| Epoch/final trials | Figure 9: 20 per epoch, 1,000 after 800 epochs | Same in resolved config | Matches counts |

Evidence in archived source: `continuous_world_modules/fb_agent.py` `_eval_agent`
lines 352–381, `act` / `act_e_greedy` lines 223–241; `env.py` reset/sample_point,
step reward; `geometry.py` Point.is_close_to; `continuous_main.py` setup.
Reference URL:
https://github.com/ahmed-touati/controllable_agent/blob/cea625688c7e9680bb98d82339d702758d812b3e/continuous_world_modules/fb_agent.py

Statistical equivalence is not seed-stream identity. The original shares global
NumPy/Python RNGs across training, ordinary evaluation and extra GPI evaluation.
Our evaluation is isolated from training, with independent per-trial seeds.
Computing B(g) once per trial rather than once per step is equivalent because
weights and goal stay fixed during evaluation. No GPI is used in the reported
ordinary goal-reaching score. Figure 9 aggregates training seeds; seed 0 alone
does not supply that uncertainty.

New automated checks in `test_continuous_maze_diagnostics.py` verify online B/F
selection, unmodified task passed to F, post-action early stopping, exactly 30
actions for failures, known bootstrap residual values, and deterministic held-out
probes. All four passed. Existing environment/source-gradient parity checks remain
in `test_continuous_maze.py`.

# Checkpoint diagnostic suite

```powershell
& ./.venv/Scripts/python.exe diagnose_continuous_maze.py
```

Defaults: epochs 100, 200, 400, 600, 800; 1,000 fixed goal trials per checkpoint;
epsilon=.02, online F and B, 30 steps; independent fixed evaluation seed 918273.
Outputs go to `experiments/continuous_maze/diagnostics/seed0/`, separate from
training. Missing future checkpoints are listed in `status.json`. Rerun the same
command after they appear: completed diagnostic results are cached and verified
against checkpoint hashes. Use a different `--output` when changing suite options.
This is a separate fixed evaluation sample from the runner's final 1,000 goals;
small differences between those estimates are expected.

Reported metrics:

- Success rate and 95% Wilson interval (evaluation-trial uncertainty, not seed variance).
- Mean successful steps, plus mean capped steps including 30-step failures.
- Wall rejection percentage: rejected moves / all executed actions, pooled across
  trials; failed trials contribute up to 30 actions.
- Mean signed, mean absolute and root-mean-square sampled FB bootstrap residual,
  plus relative RMSE and backup entropy.
- Mean/median top-two online Q gap on visited pre-action states, and mean gap
  divided by the within-state Q range. These states depend on the checkpoint's
  policy, so gap changes reflect both policy and state-visitation changes.
- Deterministic first-argmax greedy entropy is exactly zero; top-two gaps are the
  useful ambiguity diagnostic. Epsilon-greedy behavior entropy remains separate.

Residual definition: on 4,096 held-out random-policy transitions with uniform
resets every 30 steps, sample 32 pairs of independent minibatches of size 128 and
fixed Cauchy tasks. Compute

```text
e_ij = F_online(s_i,a_i,z_i)^T B_online(s'_j)
       - gamma * sum_a softmax(F_target(s_next_i,a,z_i)^T z_i / 200)_a
                       * F_target(s_next_i,a,z_i)^T B_target(s'_j).
```

This matches the training loss's sampled bootstrap matrix before its square.
It excludes the immediate occupancy/Dirac term represented by the separate
diagonal term in the FB objective. Thus it is **not** a complete pointwise Bellman
residual and need not vanish at a good solution. At clipped boundaries or rejected
moves, held-out states may repeat as well. Raw residual size depends on embedding
scale; relative RMSE divides by RMS bootstrapped target. No claim of poor learning
should rely on raw residual or total loss alone. Held-out observations, checkpoint
hashes, suite settings, trajectories and metrics are saved for inspection.

# Next experiment decision

Choose the branch using the **completed 800-epoch** fixed-suite and runner final
scores, their trial uncertainty, trajectories, and checkpoint trend. Do not select
hyperparameters from the noisy 20-goal epoch score. The ~83% FB-50 reference is a
visual estimate of Figure 9, not a precise published scalar; 70–80% and results
above 83% warrant checking uncertainty rather than forcing a category.

| Outcome | Next experiment | Controls and evidence |
| --- | --- | --- |
| Comparable to ~83% | First reproduce seeds 1 and 2, then increase wall complexity gradually | Separate new environment/configs. Start with original two walls; increase wall lengths before adding alternating segments. Hold dynamics, goal distribution, training budget and evaluation suite fixed. Track route feasibility within 30 steps so longer necessary paths do not masquerade as FB failure. Save a baseline checkpoint for every seed. |
| Moderate, 40–70% | Compare 100/200/400/600/800 fixed-suite curves, then audit residual implementation differences | Look for plateau vs late decline, collision percentage, Q gaps, F/B scale and backup entropy. Verify 800,000 Adam steps, 20,000 cycle target updates and actual saved config. Prioritize optimizer/rounding drift, replay sampling/coverage, source regulariser behavior and evaluation seed variance. Change one factor per separately labeled ablation; keep this reference run intact. |
| Poor, <40% | Paired PyTorch-vs-JAX training on identical transitions, initial weights, minibatch indices and Cauchy task tensors | Use archived actual authors' update method, three 256-unit layers and same .0005 Adam. Force all stochastic inputs and once-per-cycle targets to match; compare losses/gradients/Adam moments/parameters after updates 1, 10, 40, 400 and 1,000 before long training. The existing single-update gradient parity test is the starting check, not proof of multistep trajectory parity. Evaluate both online policies on the same trials. |

Do not start additional training branches until the full run is evaluated.
Any environment extension must remain separate from both the original Continuous
Maze and Four Rooms. No hyperparameters were changed during preparation.
