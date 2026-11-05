# Evaluating FB successor measures

`evaluate_fb_jax.py` compares signed, unnormalised online F/B predictions with independent Monte Carlo visitation measures. It does not use `export_sr` (which clamps), greedy actions, or goal-reaching scores.

## Definitions verified against the loss

The diagonal loss is `-mean(F(s,a,z) dot B(s))`. Consequently the Bellman equation has immediate measure **delta_s**, not delta_next_s. Rollouts count their initial state with weight 1, force a0 on the first transition, and then sample the frozen target-network policy:

`softmax(mask(F_target(s,z) dot z / temperature))`.

The temperature is read from the checkpoint (200 in this run). The current environment allows every action everywhere; wall collisions reject movement rather than removing actions. The same availability mask is used here. Zero tasks give uniform actions. F internally rescales z exactly as in training; action scores use the original z.

Terminal flags are absent from the loss and no terminal multiplier is applied to bootstrapping. Thus the learned equation describes continuing physical dynamics, without goal resets or the collector's resets every 30 steps. These resets shape replay rho but do not appear as transitions in the Bellman target. Evaluation never stops/resets on reaching a task goal. Goal-conditioned tasks are z=B_online(goal), and do not change physical dynamics.

Online F and B are compared with the frozen target policy because that is the policy used by the loss. The loss's bootstrap also uses target F and B, so target lag is a possible error source; this is not an exact converged fixed-point guarantee. Finite-dimensional approximation also cannot represent every delta distribution exactly in continuous space. Spatial binning tests integrated measures, rather than point densities.

## Run

CPU smoke test (loads the real checkpoint, verifies horizon-one indexing, exact region rewards, and continuing discounted mass):

```powershell
.\.venv\Scripts\python.exe evaluate_fb_jax.py --smoke-only
```

Full evaluation, when replay observations were saved during training:

```powershell
.\.venv\Scripts\python.exe evaluate_fb_jax.py --device cuda
```

Defaults: 500 independent rollouts per configuration, horizon 700, 20 by 20 bins, fixed seed 2026, 100,000 reference observations. There are 13 configurations: eight goal-conditioned cases covering four rooms, varied starts and initial actions; four Cauchy-ball task vectors sampled as in training; and a zero task. Full evaluation runs the smoke test first. Use `--device cpu` when no GPU backend is installed. JIT/vmap parallelise rollouts; replay B embeddings are computed once in batches and reused. Only accumulator maps and reward returns are retained, rather than entire trajectories.

A short pipeline check, distinct from a valid full evaluation:

```powershell
.\.venv\Scripts\python.exe evaluate_fb_jax.py --rollouts 16 --horizon 30 --max-configs 2 --output evaluation_fb_fast
```

## Replay reference: existing checkpoint limitation

The original checkpoint contains network/optimizer/RNG state, **not replay observations or environment RNG/collection metadata**. It is impossible to retrieve the exact original replay from this checkpoint alone. Never replace rho with uniform samples.

The training runner now writes `fb_fourrooms_jax_reference.npz`, sampling 100,000 observations with replacement from its final replay, using a separate NumPy RNG that cannot change training. This is an empirical final-buffer reference, not the average of every replay distribution encountered during learning. The FB learning algorithm is unchanged. Checkpoints remain in the existing format.

For the existing 800,000-update checkpoint, a conditional reconstruction is provided:

```powershell
.\.venv\Scripts\python.exe evaluate_fb_jax.py --reconstruct-reference-cycles 20000
```

This reruns the *actual training collector* (four 30-step random-action rollouts per cycle, including goal resets), without any learner updates. It samples uniformly from the final 1,000,000-state replay window. The checkpoint seed is used; movement noise, geometry and action set match the current runner defaults. It assumes this checkpoint came from the unchanged runner with 40 updates/cycle. It does not reconstruct a uniform distribution. Original CPU/GPU backend differences and unrecorded environment/custom settings can change reset trajectories. The resulting file is explicitly labelled `reconstructed_final_replay`; results using it are conditional, not definitive validation under the unavailable original rho. For certainty, rerun training with reference saving or export replay from a still-live trained learner.

`--reference PATH`, `--checkpoint PATH` and `--output DIR` select artifacts. Environment size/noise/actions and update counts are checked when reference metadata is present. The original checkpoint is never overwritten.

## Outputs and interpretation

- `metrics.csv`, `metrics.json`: L1/MC mass (neither map is normalised), signed and relative mass errors, negative-bin fraction, integrated negative mass, negative point-score statistics, correlation, per-room signed mass errors, and MC split-half variability. JSON records task vectors, starts/actions, hashes, seed, reference provenance and assumptions.
- `reward_predictions.csv`: exact continuous room, corridor and doorway indicator returns; MC and reference-sampling standard errors. These are computed from continuous states, not grid-cell centers. Room indicators exclude the door corridors; the extra corridor region completes the partition.
- `maps.npz`: raw signed FB maps, MC maps and pointwise MC standard errors, identical bin definitions (y rows, x columns; domain upper boundary included in last cell).
- Per-configuration PNGs: empirical, predicted, absolute error; empirical and predicted share a diverging scale with negative values visibly distinct. Walls use the environment rectangles, leaving doors open.
- `summary.png`, `interpretation.txt`: configuration errors and a critical summary.

Theoretical infinite mass is 100 at gamma=.99. The horizon-700 missing mass is gamma^700/(1-gamma), about .088; this bound is reported, not used to rescale either map. MC return uncertainty comes from independent rollout returns (not treating correlated timesteps as independent samples). Reference-sample uncertainty is conditional on the empirical replay distribution and does not cover reconstruction error. Split-half L1 is a noise diagnostic, not a confidence interval for FB accuracy.

Large discrepancies beyond the horizon tail and MC variability rule out accuracy of the evaluated representation/reference combination. They do not by themselves distinguish incorrect rho from factorisation error. A near-correct total mass alone is also insufficient: spatial errors, negative mass, room errors, and arbitrary-reward returns must agree. Low loss and goal-reaching success provide no substitute for this comparison.

The evaluation also records target-F/B prediction L1 and mass against the same Monte Carlo samples as a diagnostic of target lag. See EVALUATION_RESULTS.md for the completed full evaluation and its limitations.

Analytical regression tests: .\.venv\Scripts\python.exe -m unittest test_evaluate_fb_jax -v

