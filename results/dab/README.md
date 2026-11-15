# Selected DAB results

Seven figures retained; large arrays, repeated epochs and less informative variants were removed.

## Main example: symmetric width 1.5, epoch 800

- [Original vs first-hit clipping](symmetric_width1.5_replay_continuing_seed0/epoch_0800/comparison_01.png)
- [Original vs first-hit compression](symmetric_width1.5_replay_continuing_seed0/epoch_0800/comparison_02.png)
- [First-hit frequency above 1.05](symmetric_width1.5_replay_continuing_seed0/epoch_0800/first_hit_above_1p05.png)
- [Original frequency above 1.05](symmetric_width1.5_replay_continuing_seed0/epoch_0800/original_above_1p05.png)
- [First-hit upper cap, exclude negatives, then mean](symmetric_width1.5_replay_continuing_seed0/epoch_0800/first_hit_cap_exclude_negative/comparison.png)

## Useful contrasting examples, epoch 800

- [Orthonormal width 2: original vs first-hit compression](width2_replay_continuing_orthonormal_seed0_20261109_084240/epoch_0800/comparison_02.png)
- [Stopped width 2: weak doorway localisation](width2_replay_stopped_new/epoch_0800/comparison_02.png)

JSON summaries alongside the figures retain checkpoint, geometry and policy provenance. The sweep used matched sampled pairs and checkpoint softmax temperatures. Paired comparison columns have matched colour scales; standalone maps auto-scale. These processed maps are exploratory and do not establish statistical reliability of the raw mean; see the raw DAB diagnostic examples for that limitation.

Generation scripts remain in the project for reproduction.
