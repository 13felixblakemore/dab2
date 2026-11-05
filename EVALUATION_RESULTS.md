# Results: trained Four Rooms FB checkpoint

The evaluation ran on the saved 800,000-update checkpoint on CPU, with 500 independent rollouts of horizon 700 for each of 13 configurations, a 20 by 20 spatial grid, and seed 2026. Primary results use online F/B and the exact frozen target-network masked softmax policy at temperature 200.

**This model is not validated as an accurate successor measure across the tested tasks.** The maps and reward-return comparisons show meaningful errors and a severe failure for one random task. Conclusions are conditional on the reconstructed replay reference described below.

| Quantity | Result |
|---|---:|
| Mean normalized spatial L1, all 13 cases | 0.5081 |
| Median normalized spatial L1 | 0.2693 |
| Mean goal-conditioned normalized L1, 8 cases | 0.2850 |
| Range excluding the extreme random task | 0.1404 to 0.3937 |
| Worst normalized L1, random_2 | 3.5015 |
| Mean relative total-mass error vs 100 | 4.88% |
| Monte Carlo total mass | 99.9120 |
| Finite-horizon missing mass | 0.0880 |

Normalized L1 is the sum of absolute bin errors divided by MC total mass. Neither prediction nor ground truth is separately normalised. Total mass looks substantially better than the spatial measure; it would be misleading to validate this model using mass alone.

The MC split-half spatial discrepancy ranges from 0.0554 to 0.1596. This diagnostic is noisy and uses half-sized samples, so it is not a confidence interval for the reported errors. In particular the zero-task error (0.1404) is comparable to its split-half discrepancy (0.1503), so we cannot confidently attribute all of that case's spatial discrepancy to representation error. Most goal-conditioned discrepancies are larger, and the extreme random-task error dwarfs MC variability (split-half 0.0554).

## Severe random-task failure

For random_2:

- Task norm: 25.37, compared with 1.22, 3.30 and 4.70 for the other sampled random tasks. It is a legitimate draw from the training Cauchy-ball distribution, though its larger magnitude makes task-space generalisation a plausible issue; this is a hypothesis, not a proven cause.
- Predicted total mass: 117.25.
- Negative bin fraction: 63.25% (including the entire spatial grid in the denominator).
- Integrated negative mass: 157.70. Positive and negative errors cancel heavily in the total mass.
- Target-F/B diagnostic L1: 3.5173, compared with online L1 of 3.5015. Switching prediction networks does not remove the failure.

| Indicator reward | MC return | MC standard error | FB predicted return |
|---|---:|---:|---:|
| Bottom-left room | 79.40 | 0.32 | 185.33 |
| Top-left room | 0.00 | 0.00 | -26.08 |
| Bottom-right room | 14.36 | 0.30 | -41.99 |
| Top-right room | 0.00 | 0.00 | -4.68 |

An indicator-reward discounted return must lie between 0 and 100 in the infinite-horizon continuing setting. Negative predictions and 185.33 violate these bounds. The FB reference-sampling standard error for the bottom-left prediction is 3.14, much smaller than its 105.93 return error. The finite-horizon tail is at most 0.088 for any of these indicator rewards, so truncation cannot explain these discrepancies. No clipping was used.

## What can and cannot be concluded

- **Policy mismatch:** greedy-policy mismatch has been avoided: evaluation samples from the training target softmax, including masking and temperature. Analytical tests check that a deliberately masked high-logit action is never sampled, and that available-action frequencies match the temperature-scaled probabilities. Zero tasks induce uniform actions.
- **Indexing and termination:** visitation starts at s0 with discount 1; a0 controls the following transition. The objective bootstraps through every transition, so evaluation continues without goal resets. Horizon-one and forced-action tests independently check this convention.
- **Finite horizon:** the omitted tail is 0.088 total mass, far below the substantial errors observed.
- **Target lag:** target prediction errors are close to online errors throughout the suite (target L1 for random_2 is 3.5173), arguing against target lag as the sole explanation.
- **Reference uncertainty:** the old checkpoint did not save replay. The evaluation reconstructed the actual epsilon=1 collector for 20,000 cycles, then sampled 100,000 states uniformly from the final 1,000,000-state replay window. This assumes the original default environment, seed and collection schedule. Original backend differences or unrecorded settings may change reset trajectories. This is explicitly not a recovered original buffer. A final-buffer reference also differs from the time-varying references encountered during training. These uncertainties prevent a definitive decomposition into rho mismatch versus representation error.
- **Representation accuracy:** under the tested replay reconstruction, the factorisation is not a reliable successor-measure predictor across tasks. The extreme random task gives especially strong contrary evidence. Some individual cases, particularly the zero task, require more MC samples to separate smaller representation errors from sampling noise. Low training loss or successful navigation cannot override these measure-level results.

Future training now writes a checkpoint-bound sample of real final replay observations; evaluating that artifact eliminates the reconstruction assumption. It does not eliminate finite-sample uncertainty or the need to examine spatial and arbitrary-reward errors.

## Artifacts

See `evaluation_fb/metrics.json` and `metrics.csv` for every configuration; `reward_predictions.csv` for nine reward functions per configuration; `maps.npz` for raw maps and MC standard errors; and `summary.png` plus the per-configuration heatmaps. `metrics.json` records checkpoint/reference hashes, the actual task vectors and policy/convention metadata. `EVALUATION.md` gives commands and explains the estimator definitions.
