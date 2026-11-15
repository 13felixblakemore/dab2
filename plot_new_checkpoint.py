"""Plot successor densities and original-formulation DAB for a saved FB run."""
import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

from dab import visualise_dab, visualise_successor_measure
from dab_first_hitting import build_valid_grid, compute_density_terms
from fb_navigation.agents.forward_backward_jax import FBLearner
from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('experiments/fourrooms_doorways/runs/symmetric_width1_replay_continuing_orthonormal_embed150_seed0/epoch_0800.pkl'))
    parser.add_argument('--pairs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=4)
    args = parser.parse_args()
    run = args.checkpoint.parent
    config = json.loads((run/'config.json').read_text())
    recorded = json.loads((run/'metadata.json').read_text())
    env = AdjustableFourRooms(**config['environment'], seed=args.seed)
    block_type = type(env.wall_blocks[0])
    env._wall_blocks = [block_type(*box) for box in recorded['wall_blocks']]
    learner = FBLearner.from_checkpoint(str(args.checkpoint))
    assert learner.temperature == config['learner']['temperature']
    X, Y, mask, states = build_valid_grid(env, 50)
    u, v = env.sample_points(args.pairs), env.sample_points(args.pairs)
    terms = compute_density_terms(learner, states, learner.embed_B(states), u, v,
                                  learner.embed_B(v), 'softmax', learner.temperature, 4)
    m_us, _, _, _, m_ss, _ = terms
    assert m_us.shape == m_ss.shape == (args.pairs, int(mask.sum()))
    with np.errstate(divide='ignore', invalid='ignore'):
        raw = m_us/m_ss
        raw_mean = np.mean(raw, axis=0)
    # Keep the raw arithmetic mean unfiltered; mark numerically undefined states.
    undefined = ~np.isfinite(raw).all(axis=0)
    raw_mean[undefined] = np.nan
    accepted = np.isfinite(raw) & (m_us >= 0) & (m_ss > 50) & (raw <= 2)
    counts = accepted.sum(axis=0)
    filtered_mean = np.divide(np.where(accepted, raw, 0).sum(axis=0), counts,
                              out=np.full(counts.shape, np.nan), where=counts>0)
    # First adjacent-room sample, with rooms inferred from the recorded wall slabs.
    walls = np.array(recorded['wall_blocks'])
    vertical = walls[(walls[:,3]-walls[:,1])>(walls[:,2]-walls[:,0])]
    horizontal = walls[(walls[:,2]-walls[:,0])>(walls[:,3]-walls[:,1])]
    vx = Counter(map(tuple, vertical[:,[0,2]])).most_common(1)[0][0]
    hy = Counter(map(tuple, horizontal[:,[1,3]])).most_common(1)[0][0]
    def rooms(points):
        x, y = np.asarray(points).T
        inside = ((x<vx[0])|(x>vx[1])) & ((y<hy[0])|(y>hy[1]))
        return np.where(inside, (x>vx[1]).astype(int)+2*(y>hy[1]).astype(int), -1)
    room_u, room_v = rooms(u), rooms(v)
    delta = room_u ^ room_v
    examples = np.flatnonzero((room_u>=0)&(room_v>=0)&((delta==1)|(delta==2)))
    if not len(examples):
        raise ValueError('No adjacent-room example; increase --pairs')
    pair = int(examples[0])
    output = Path('results/dab')/run.name/args.checkpoint.stem
    output.mkdir(parents=True, exist_ok=True)
    def grid(values):
        full = np.full(X.size, np.nan)
        full[mask] = values
        return full.reshape(X.shape)
    density_grid = grid(m_us[pair])
    fig, _ = visualise_successor_measure(density_grid, X, Y, env, start=u[pair], goal=v[pair],
                  title=f'Successor density — adjacent-room pair {pair}\nEmbedding {learner.embed_dim}; softmax T={learner.temperature:g}')
    fig.savefig(output/'successor_density_example.png', dpi=170)
    plt.close(fig)
    images = [('dab_raw_mean', raw_mean, 'Original DAB: raw arithmetic mean (no filtering)'),
              ('dab_filtered_mean', filtered_mean, 'Original DAB: calculate_dab.py filters'),
              ('successor_density_mean', m_us.mean(axis=0), 'Successor density: mean over sampled tasks')]
    for name, values, title in images:
        fig, _ = visualise_dab(grid(values), X, Y, env,
                    title=f'{title}\n{args.pairs} pairs; embedding {learner.embed_dim}; softmax T={learner.temperature:g}',
                    colorbar_label='Successor density' if name.startswith('successor') else 'Mean successor-density ratio')
        fig.savefig(output/(name+'.png'), dpi=170)
        plt.close(fig)
    summary = dict(checkpoint=str(args.checkpoint), wall_blocks=recorded['wall_blocks'],
                   seed=args.seed, n_pairs=args.pairs, grid_size=50, embed_dim=learner.embed_dim,
                   hidden_dim=config['learner']['hidden_dim'], policy='softmax', temperature=float(learner.temperature),
                   formulation='original successor-density ratio m_us/m_ss under pi_v',
                   sampling='independent uniform free-space endpoints, including same-room tasks',
                   example_pair=pair, example_selection='first adjacent-room sampled pair; no density-based selection',
                   nonfinite_values=int((~np.isfinite(raw)).sum()), undefined_raw_mean_states=int(undefined.sum()),
                   filtered_definition='finite ratio AND numerator>=0 AND denominator>50 AND ratio<=2; mean over retained samples, matching calculate_dab.py filters',
                   retained_fraction=float(accepted.mean()),
                   raw_mean_range=[float(np.nanmin(raw_mean)),float(np.nanmax(raw_mean))],
                   filtered_mean_range=[float(np.nanmin(filtered_mean)),float(np.nanmax(filtered_mean))])
    (output/'plot_metadata.json').write_text(json.dumps(summary, indent=2))
    np.savez_compressed(output/'plot_data.npz', X=X,Y=Y,valid_mask=mask,starts=np.asarray(u),goals=np.asarray(v),
                        raw_mean=raw_mean,filtered_mean=filtered_mean,retained_counts=counts,
                        mean_successor_density=m_us.mean(axis=0),example_density=m_us[pair],example_pair=pair)
    print(json.dumps(summary, indent=2))
    print(f'Saved four plots to {output.resolve()}')


if __name__ == '__main__':
    main()
