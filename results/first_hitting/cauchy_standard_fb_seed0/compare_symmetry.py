"""Compare symmetry-balanced first-hit estimates without symmetrizing outputs."""
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dab_first_hitting import square_symmetries
from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms
from fb_navigation.environments.continuous_four_rooms import visualize_environment


root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--independent-10000', action='store_true')
args = parser.parse_args()
folders = (('epoch_0800_10000_softmax_cap2', 'epoch_0800_10000_greedy_cap2')
           if args.independent_10000 else
           ('epoch_0800_symmetry_softmax_cap2', 'epoch_0800_symmetry_greedy_cap2'))
stem = 'policy_comparison_10000_cap2' if args.independent_10000 else 'policy_comparison_symmetry_cap2'
archives = [np.load(root / folder / 'first_hitting_results.npz') for folder in folders]
for key in ('u', 'v', 'X', 'Y', 'valid_mask'):
    assert np.array_equal(archives[0][key], archives[1][key]), key
if args.independent_10000:
    for archive, folder in zip(archives, folders):
        summary = json.loads((root / folder / 'summary.json').read_text())
        assert not summary['symmetry_pairs'] and summary['total_pairs'] == 10000
else:
    original = np.load(root / 'epoch_0800' / 'first_hitting_results.npz')
    for archive in archives:
        for key in ('u', 'v'):
            expected = square_symmetries(original[key], 1.0).astype(archive[key].dtype)
            np.testing.assert_array_equal(archive[key], expected)

env = AdjustableFourRooms(doorway_scale=1.0, include_stay_action=False)
metadata = json.loads(Path('experiments/fourrooms_doorways/runs/cauchy_standard_fb_seed0/metadata.json').read_text())
env._wall_blocks = [type(env.wall_blocks[0])(*b) for b in metadata['wall_blocks']]
means = []
for archive in archives:
    mean = archive['mean_clipped']
    valid_mask = archive['valid_mask']
    means.append(mean[valid_mask] if mean.shape == valid_mask.shape else mean)
vmin = min(np.nanmin(mean) for mean in means)
vmax = max(np.nanmax(mean) for mean in means)
fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
stats = {}
for ax, archive, mean, folder, label in zip(axes, archives, means, folders, ('Softmax (temperature 200)', 'Greedy')):
    grid = np.full(archive['X'].size, np.nan)
    grid[archive['valid_mask']] = mean
    grid = grid.reshape(archive['X'].shape)
    mesh = ax.pcolormesh(archive['X'], archive['Y'], np.ma.masked_invalid(grid), shading='nearest', cmap='viridis', vmin=vmin, vmax=vmax)
    visualize_environment(env, ax, draw_grid=False, draw_initial_position=False, draw_goal=False)
    ax.set(title=label, xlim=(0, env.size), ylim=(0, env.size), aspect='equal')
    transforms = [np.rot90(grid, k) for k in range(4)]
    transforms += [np.fliplr(t) for t in transforms.copy()]
    counts = np.sum(np.isfinite(transforms), axis=0)
    symmetric_mean = np.where(counts > 0, np.nansum(transforms, axis=0) / np.maximum(counts, 1), np.nan)
    summary = json.loads((root / folder / 'summary.json').read_text())
    stats[label] = dict(mean=float(np.nanmean(mean)), retained_fraction=summary['retained_fraction'], mean_absolute_asymmetry=float(np.nanmean(abs(grid-symmetric_mean))))
fig.colorbar(mesh, ax=axes, label='Mean first-hit estimate clipped to [0, 2]')
fig.suptitle(('10,000 independent pairs; no rotations or reflections' if args.independent_10000 else '8,000 symmetry-balanced pairs; unchanged stability filters'))
fig.savefig(root / f'{stem}.png', dpi=170)
stats['comparison'] = dict(identical_pairs_and_grid=True, symmetry_pairs=not args.independent_10000, mean_absolute_map_difference=float(np.nanmean(abs(means[1]-means[0]))), map_correlation=float(np.corrcoef(means)[0, 1]))
(root / f'{stem}.json').write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
