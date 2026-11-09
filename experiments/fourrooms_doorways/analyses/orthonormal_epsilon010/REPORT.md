# Orthonormal final model: evaluation epsilon 0.1

The epoch-800 checkpoint was evaluated on its 600 saved monitor pairs (200 per category), with one noisy rollout per pair, online F/B, doubled doorways and coordinate-wise goal tolerance 0.1. Epsilon 0.02 was rerun on the same local CPU backend as epsilon 0.1 to control for backend/version differences. Both policies use identical per-trial random seeds. No training or checkpoint changes were made.

| Category | Horizon | Success eps=.02 | Success eps=.10 | Change (pp) | Paired 95% interval (pp) | Wall actions eps=.02 | Wall actions eps=.10 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| same_room | 30 | 99.5% | 100.0% | +0.5 | [+0.0, +1.5] | 0.82% | 0.59% |
| adjacent_rooms | 30 | 64.0% | 70.5% | +6.5 | [+2.5, +11.0] | 46.01% | 36.43% |
| diagonal_rooms | 30 | 34.0% | 40.0% | +6.0 | [+1.0, +11.0] | 48.18% | 37.53% |
| same_room | 150 | 99.5% | 100.0% | +0.5 | [+0.0, +1.5] | 1.80% | 0.59% |
| adjacent_rooms | 150 | 79.5% | 93.5% | +14.0 | [+8.5, +19.5] | 72.86% | 53.54% |
| diagonal_rooms | 150 | 53.5% | 82.0% | +28.5 | [+22.0, +35.0] | 75.31% | 55.48% |

Intervals use 20,000 paired endpoint bootstrap samples (seed 20261109). They do not capture training-seed uncertainty. Each pair has only one noise realization, so repeated rollouts would better establish robustness. Wall-action percentages pool executed actions and exclude actions after success.

The policies differ only in their random-action probability: 2% versus 10%; a random choice may select the greedy action too. Longer-horizon gains are consistent with exploration helping escape traps, but this evaluation does not directly classify stationary trap escapes.

## Saved baseline reproduction

Exact array agreement with the original cluster epsilon-0.02 evaluation:

```json
{
  "hits_30": {
    "exact": true,
    "differing_elements": 0
  },
  "collisions_30": {
    "exact": true,
    "differing_elements": 0
  },
  "doors_30": {
    "exact": true,
    "differing_elements": 0
  },
  "hits_150": {
    "exact": true,
    "differing_elements": 0
  },
  "collisions_150": {
    "exact": true,
    "differing_elements": 0
  },
  "doors_150": {
    "exact": true,
    "differing_elements": 0
  }
}
```

Full settings and input hashes: [config.json](config.json). Metrics, paired gains/losses and baseline checks: [results.json](results.json).

Reproduce from the repository root:

```bash
python experiments/fourrooms_doorways/analyses/orthonormal_epsilon010/evaluate.py
```
