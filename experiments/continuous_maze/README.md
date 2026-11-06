# Continuous Maze reproduction

Source: [Touati & Ollivier (2021), Appendix D.1–D.4](https://arxiv.org/abs/2103.07945).
Reference sources are archived under `reference/`, from
[cea625688c7e9680bb98d82339d702758d812b3e](https://github.com/ahmed-touati/controllable_agent/tree/cea625688c7e9680bb98d82339d702758d812b3e).
`reference/sha256.json` records source hashes; `paper.pdf` and `figure9_page.png` preserve the comparison source.
The reference files are for validation and attribution, not a replacement learner.

From the repository root in PowerShell, using this project's configured interpreter:

```powershell
& ./.venv/Scripts/python.exe -m unittest test_continuous_maze test_goal_terminated_fb_jax -v
& ./.venv/Scripts/python.exe run_continuous_maze.py --epochs 1 --output experiments/continuous_maze/runs/my_smoke
& ./.venv/Scripts/python.exe run_continuous_maze.py --seed 0
& ./.venv/Scripts/python.exe run_continuous_maze.py --seed 0 --resume
# Subsequent independent replications:
& ./.venv/Scripts/python.exe run_continuous_maze.py --seed 1
& ./.venv/Scripts/python.exe run_continuous_maze.py --seed 2
```

Dependencies already available here: Python 3.12.10, NumPy, JAX 0.12.0.1,
Matplotlib, and PyTorch 2.15.0+cpu (PyTorch is only needed for parity tests).
`--device cuda` selects a JAX GPU on a machine with a compatible installation;
this Windows installation exposes CPU only. Host replay avoids GPU memory assumptions.
No package installs or existing Four Rooms changes are required.

`fb50.json` is the full configuration. `num_layers=2` in the existing learner means
**three** hidden layers (`num_layers + 1`), not two. Raw replay states have width 2;
the shared RBF featuriser maps these to 441 inputs. F outputs `(batch,50,5)` and
B outputs `(batch,50)`. The schedule is:

| Quantity | Per cycle | Per epoch | Full run |
| --- | ---: | ---: | ---: |
| Cycles | 1 | 25 | 20,000 |
| Gradient updates | 40 | 1,000 | 800,000 |
| Collected transitions | 120 | 3,000 | 2,400,000 |
| Polyak updates | 1 | 25 | 20,000 |

Evaluation is after each epoch on 20 independent uniform start/goal pairs.
Final evaluation uses 1,000 pairs for epsilon=.02, and separate greedy/random
diagnostics. All three final policies share the same start/goal pairs, using
per-trial seeds so stopping early does not change subsequent pairs. Reward is
checked after moving (including initially successful states), matching the source.
Goals use **both coordinate differences <= .05**, not a Euclidean ball.

Each output directory contains the resolved configuration, source hashes/runtime
versions, `training.csv`, one model checkpoint per epoch, `resume.pkl`, final
metrics, full evaluation trajectories, `success.png`, and `trajectories.png`.
New runs also archive their executed sources in `source/`. Existing run directories
are never silently reused. `--epochs` explicitly makes a truncated experiment;
its results are not an 800-epoch comparison. Resume requires the original config
and sources; it restores the latest completed epoch, optimizer/targets/JAX key,
replay contents and replay/environment/action RNGs. A partially completed epoch is
discarded after interruption. Models are retained at every epoch; only the latest
full replay snapshot is retained. Pickle checkpoints must be trusted.

## Source compatibility audit

| Detail | Reproduction |
| --- | --- |
| Continuing objective | No terminal or goal masks; training never stops on goal success |
| RBF | Same grid order, sigma=.05, `exp(-distance/(2*sigma**2))`, sum normalization |
| Task sampling | Gaussian direction, scalar centered Cauchy scale .5, multiply by sqrt(d); same F task preprocessing |
| Zero tasks | Probability 0; released code does not mix them |
| Gradient clipping | Disabled; released clipping line is commented out |
| Adam | Joint F/B optimizer, lr=.0005, beta=.9/.999, eps=1e-8 outside sqrt |
| Targets | Initially copied; fixed for all 40 updates; .95 old + .05 new once per cycle |
| Backup policy | Target F, Boltzmann softmax(Q/200), all five actions allowed |
| Replay | FIFO ring, independent with-replacement transition/other minibatches |
| B regulariser | Exact released expression, opt-in `b_regularization="touati_continuous"` |

The released regulariser is **not** the expression previously in this JAX learner:

```text
C = B(obs) @ B(other).T
D = B(obs) @ stop(B(other)).T
n = sum(B(obs) * stop(B(obs)), axis=1)
released: mean(n * stop(C)) - mean(D)  # n broadcasts over columns
existing: mean(D * stop(C)) - mean(n)
```

We deliberately preserve the released formula, including its broadcasting and
detach locations. The existing default remains unchanged. The paper's abstract
orthonormalisation formula and this released implementation differ; this is a
**released-code reproduction**, not a claim that those formulas are identical.
Tests execute the actual reference `_update_network` with copied JAX weights,
identical minibatches and tasks, then compare its scalar loss and every F/B
parameter gradient (including detach behavior) against JAX.

Remaining differences: JAX vs PyTorch RNG/linear algebra/Adam rounding; host replay
stores float32 coordinates where the source stores NumPy coordinates then casts
each minibatch to float32 (same learner inputs). Local environment RandomState
replaces global NumPy state. Evaluation uses independent RNG streams and omits
the released auxiliary GPI evaluation, so evaluation cannot alter future training
data. Replay indices use NumPy Generator rather than Python random. Consequently,
the same seed is reproducible within this runner but does not yield the authors'
exact transition or task sequences. CPU parallel reductions can also differ at
floating-point roundoff level. No hyperparameters are adjusted in response to
success rates.

The geometry intentionally retains `>=` orientation tests and the reference's
ordered-y `on_segment` check for vertical walls, even for the reversed second
wall. Noise is float32 **before** addition. Scalar types after clipping are
retained because NumPy 2 promotion would otherwise cause tiny differences on
later transitions. Seeded trajectory tests compare exact positions to the source,
not just approximate wall layouts.

## Metrics and interpretation

`loss_mean` averages every training loss in an epoch. `td`, `immediate`, `fb`,
`regularization`, `total`, `entropy`, `f_norm`, `b_norm` come from one deterministic,
independently sampled post-epoch diagnostic minibatch; these need not equal
`loss_mean`. `entropy` is the target Boltzmann backup entropy (nats);
`behavior_entropy` is the epsilon-greedy execution distribution entropy. Wall
collisions count rejected moves, including repeats; they are not unique wall hits.
Path length averages only successful trials and is null if there are none.

Figure 9 visually places FB-50 final success near **0.83** (approximate visual
reading, not digitized data); the paper averages three training seeds. A single
short seed-0 run cannot establish comparable performance. Examine final metrics
only after 800 epochs, then repeat seeds 1 and 2 before claiming replication.
`RESULTS.md` records the work actually performed in this session.

If the full run is substantially below Figure 9: first verify optimizer update
count 800,000 and target update count 20,000; inspect entropy collapse, F/B norms,
TD vs immediate/regularisation terms, collision-heavy trajectories and greedy vs
epsilon vs random outcomes. Re-run parity tests before trying alternate
regularisation or sampling. Any ablation must use a separate config/output and
be reported as a deviation; do not tune this reference configuration silently.
