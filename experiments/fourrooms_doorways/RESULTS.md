# Doubled-doorway seed-0 results

Inspected uploaded results at `fourrooms_doorways/runs/width2_seed0/` against
`../fourrooms_reference/runs/seed0/`. Both completed 800 epochs; the width-2 loss
array has shape (800,25,40), all finite, and final counters report 800,000 updates
and 2,400,000 transitions. Width is 2/11 = .1818, twice the original .0909.
Resolved configuration retains the released regulariser and continuing loss,
zero-task mixing 0, no clipping, d=50, T=200 and gamma=.99.

## Final success on the same 200 endpoint pairs per category

| Task | Original, H=30 | Doubled, H=30 | Original, H=150 | Doubled, H=150 |
| --- | ---: | ---: | ---: | ---: |
| Same room | 90.5% | 94.0% | 95.5% | 96.5% |
| Adjacent rooms | 36.5% | 47.5% | 54.5% | 64.0% |
| Diagonal rooms | 9.0% | 12.0% | 20.0% | 26.5% |

Endpoint coordinates, categories, room labels and trial seed arrays match exactly.
The common learner and evaluation sources match; four raw file hashes differed
only because of line endings (verified text equality after CRLF normalization).
However, the original run used CPU/JAX 0.12.0.1 and the uploaded run used the
cluster setup/JAX 0.5.2. These are single training seeds and backend/version
differences can alter learned solutions. The table is an observed comparison,
not a geometry-only causal estimate.

## Learning over checkpoints: doubled doorways

| Epoch | Same, H=30 | Adjacent, H=30 | Diagonal, H=30 | Adjacent, H=150 | Diagonal, H=150 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 200 | 95.5% | 54.0% | 12.5% | 68.0% | 28.0% |
| 400 | 96.0% | 49.0% | 24.0% | 67.5% | 37.5% |
| 600 | 95.5% | 43.5% | 17.0% | 63.5% | 32.5% |
| 800 | 94.0% | 47.5% | 12.0% | 64.0% | 26.5% |

There is clear deterioration on this fixed diagonal evaluation sample after
epoch 400. From epoch 400 to 800, H=30 loses 38 formerly successful diagonal
trials while gaining 14; H=150 loses 43 while gaining 21. This is not merely a
fresh set of goals at the later checkpoint. It does not by itself diagnose
overfitting or identify a training instability mechanism. Do not select epoch 400
as the final reproduction result retrospectively; report epoch 800 and the
checkpoint trend separately.

## Remaining navigation failures

At epoch 800, H=30 wall-hit actions are 5.1% for same-room, 51.3% adjacent and
53.4% diagonal. At H=150 they reach 72.9% and 73.2% for adjacent and diagonal.
Persistent wall attempts dominate the extra time on difficult trials.

For diagonal H=30 trials, 48.5% complete at least one room crossing, but only
12% reach the goal. At H=150 those values are 58.5% and 26.5%. Finding one doorway
does not reliably produce the rest of the route. The complete-crossing predicate
and goal success predicate differ, since the goal square can extend toward a
doorway; these metrics are complementary, not a strict decomposition of success.

Mean noiseless restricted-lattice BFS lengths are 4.495 / 11.885 / 19.805 for
same/adjacent/diagonal tasks. All same/adjacent pairs and 99.5% of diagonal pairs
fit the 30-step BFS reference. Hence horizon length alone is insufficient to
explain the low cross-room success. On the successful subset, H=30 adjacent and
diagonal paths average 1.17 and 1.12 times BFS, while H=150 successful paths
average 3.05 and 3.18 times BFS. The latter metrics have survivor bias; failures
are excluded, and BFS is noiseless and lattice-restricted.

The evidence supports a bottleneck/long-range navigation weakness despite wider
openings. It does not prove that the loss, representation capacity, exploration
or any particular mechanism is the cause.

## Recommended next checks

1. Inspect matched diagonal trajectories and online action scores at epochs 400
   and 800, particularly the 38 H=30 successes lost later. Distinguish aiming into
   walls, failing to find the first door, and failing after the first crossing.
2. Run width 1 and width 2 on the same cluster/software stack, then seeds 1 and 2,
   retaining the same evaluation suite. This removes the current backend/version
   confound before attributing improvements to doorway width alone.
3. Keep the reference hyperparameters fixed while diagnosing the late decline.
   Any capacity, target-update or objective ablations should be separately named.
