# Vectorised goal-terminated FB bottleneck evaluation

## Run

```powershell
python -m fb_navigation.diagnostics.estimate_fb_bottlenecks_jax
```

The default checkpoint is the existing `models/fb_fourrooms_jax_goal.pkl` (the goal-trained model found in models). The script refuses continuing or non-replay checkpoints. It never trains, mutates model parameters, or writes a checkpoint.

Defaults: 1,000 start-goal pairs; fixed 20 by 20 candidate grid; seed 2026; horizon 1000; CPU backend. Configure:

```powershell
python -m fb_navigation.diagnostics.estimate_fb_bottlenecks_jax --checkpoint models/fb_fourrooms_jax_goal.pkl --samples 1000 --grid 20 --seed 2026 --horizon 1000 --device cuda --fb-batch 8 --mc-batch 128 --output evaluation_fb_bottlenecks
```

Use `--device cpu` if a GPU JAX backend is unavailable. FB pair batches bound the temporary candidate-by-goal forward activations. B(candidate) and B(goal) are computed once. Candidate features are reused. JIT/vmap handles pair/candidate forward evaluation and rollout dynamics; scan handles rollout time. A separate random key per pair makes rollout random streams independent of batch partition.

Fast smoke test:

```powershell
python -m fb_navigation.diagnostics.estimate_fb_bottlenecks_jax --samples 16 --horizon 30 --output evaluation_fb_bottlenecks_smoke
python -m unittest tests.test_estimate_fb_bottlenecks_jax tests.test_goal_terminated_fb_jax -v
```

The short smoke run checks the pipeline, not statistical accuracy. Analytical tests independently check scalar versus vectorised FB evaluation, exact forced-motion first entry and goal exclusion, identical sampling under batch splits, region/wall geometry, and invalid-denominator handling.

## Estimands and policy

Each candidate is a square grid cell, with c at its center. Cells having any positive-area overlap with a wall rectangle are excluded, rather than just checking centers. This leaves 301 whole free cells at resolution 20. Gray edge bands in plots can therefore be excluded partial-wall cells, not model-invalid estimates; walls themselves are black.

Starts and goals are sampled continuously, uniformly in free space. Entire pairs are rejected if the start is in the Euclidean goal disk, giving the joint-uniform distribution conditioned on valid starts. This avoids the learner's unresolved start-inside-goal boundary. Each goal gives z=B_online(goal).

Both estimators use the frozen target-network masked softmax at the checkpoint's temperature (200 here), gamma (.99), and Euclidean goal threshold (.1). The Four Rooms availability mask includes all actions, including wall-rejected moves. No greedy-policy substitution is made. The environment's square goal-reset predicate is not used: goal termination is the disk used by the training loss.

F is action conditioned. The state density used here is therefore:

m_g(x,c) approximately [sum_a pi_g(a|x) F_online(x,a,B_online(g))] dot B_online(c).

The **same target policy** averages the numerator at u and the diagonal at c, and also samples the initial action of MC (no forced initial action). The checkpoint is already goal terminated: no continuing-to-terminal resolvent correction is applied. Online/target lag remains a possible approximation error.

For each candidate/pair, exclude the whole candidate cell if it intersects the goal disk. Intersection is calculated by distance from the goal to the nearest point in the cell, not just by testing the center. Both FB and MC use this exact eligibility matrix. Per-cell means divide by the number of eligible pairs N_c, rather than treating excluded pairs as zero. Starts inside a candidate cell count as first visits at t=0, as required by the current-state convention; exact u=c has probability zero under continuous sampling.

## MC validation

One independently randomised trajectory is run for each of the SAME 1,000 pairs. The first visit to each cell is recorded before goal entry and weighted gamma^tau. The goal-entry state is excluded and trajectories freeze after entering the goal disk; no resets occur. No complete position trajectories are retained: the first-hit matrix is sufficient. MC standard errors in cells.csv reflect joint pair/policy randomness and correlated temporal samples are not treated as independent observations.

Horizon truncation only omits first hits at times >= H, so its per-pair absolute discounted error is at most gamma^H. At H=1000 this is about .0000432, not the total-visitation bound gamma^H/(1-gamma). The estimate is at the checkpoint's fixed discount, not the gamma->1 limit.

## Invalid and unreliable estimates

A denominator is valid only if finite and greater than `--denominator-tolerance` (default 1e-6); the numerator and ratio must also be finite. The actual denominator is **never clamped, shifted or replaced by epsilon**. Negative numerators and finite ratios outside [0,1] are retained, flagged, and not clipped.

The strict map is missing wherever any eligible pair has an invalid ratio. A separately labelled conditional map averages only valid pairs; its matched-MC map averages exactly those same pairs. This makes selection effects visible. Metrics report strict and conditional comparisons, per-cell coverage, nonpositive/nonfinite/small diagonals, and out-of-range ratios. Correlation/MAE always compare compatible subsets. Gray indicates missing/excluded cells, distinct from zero scores.

## Interpretation limits

The FB density ratio is a **proxy for finite-region first-hitting probabilities**. A first-entry distribution into a region is not a point at the region center: the exact discrete scalar Markov identity does not automatically transfer to aggregated cells. The immediate-state Dirac term and rejected-movement atoms also limit a smooth diagonal-density interpretation. Same-state base-measure factors cancel in a density ratio, so no uniform-for-replay-density substitution is made and no saved replay is required.

The learned softmax policy is not guaranteed optimal; the original dynamics-aware betweenness definition's optimal-policy interpretation therefore cannot be asserted here. A high spatial correlation alone does not establish calibrated hitting probabilities or clean doorway discovery. Check absolute bias, negative ratios, doorway rankings, grid sensitivity, and MC uncertainty. A finite discount can favor local room visitation over long doorway-crossing paths.

## Saved artifacts

- `comparison.png`: strict FB, full MC, validity, conditional FB, matched MC and absolute error. Score panels share a symmetric logarithmic display scale to show signed values and outliers; this changes display only, not data or metrics.
- `mc_heatmap.png`: a separate linear-scale MC map for easier spatial interpretation.
- `metrics.json`, `RESULTS.md`, `cells.csv`: comparisons, doorway-region scores, counts, coverage, MC standard errors and provenance.
- `intermediate.npz`: starts, goals, tasks, candidate bounds, numerator/diagonal matrices, eligibility/validity masks, raw unclipped ratios, first-hit/goal-hit times and maps. This supports auditing or replotting without rerunning trajectories.

The original learner, environment, training script and existing smoke evaluator are unchanged by this implementation.
