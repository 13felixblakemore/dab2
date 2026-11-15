"""Save six different-room successor-density examples using the DAB config."""
import ast
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

from dab import visualise_successor_measure
from fb_navigation.agents.forward_backward_jax import FBLearner
from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', help='Override the checkpoint in the DAB config')
    parser.add_argument('--doorway-scale', type=float, help='Override the DAB doorway width')
    args = parser.parse_args()
    # Read settings without executing the full DAB calculation.
    tree = ast.parse(Path("calculate_dab_different_rooms.py").read_text(encoding="utf-8-sig"))
    config = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "config" for t in node.targets))
    env_call = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "env" for t in node.targets))
    width = next(ast.literal_eval(k.value) for k in env_call.keywords if k.arg == "doorway_scale")
    if args.checkpoint is not None:
        config['path'] = args.checkpoint
    if args.doorway_scale is not None:
        width = args.doorway_scale
    env = AdjustableFourRooms(doorway_scale=width, include_stay_action=False,
                              seed=int(config["seed"]) + 4)
    # Archived checkpoints may predate the current symmetric wall layout.
    training_metadata = Path(config['path']).parent / 'metadata.json'
    geometry_source = 'current AdjustableFourRooms'
    if training_metadata.exists():
        recorded = json.loads(training_metadata.read_text())
        if 'wall_blocks' in recorded:
            block_type = type(env.wall_blocks[0])
            env._wall_blocks = [block_type(*block) for block in recorded['wall_blocks']]
            geometry_source = str(training_metadata)
    learner = FBLearner.from_checkpoint(config["path"])
    # Use the same room-interior sampling helpers as the DAB script.
    helpers = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                              and n.name in ("room_ids", "sample_different_room_pairs")],
                         type_ignores=[])
    scope = {"np": np}
    exec(compile(helpers, "<DAB sampling helpers>", "exec"), scope)
    u, v = scope["sample_different_room_pairs"](env, 1000)
    difference = scope["room_ids"](u, env.size) ^ scope["room_ids"](v, env.size)
    indices = np.concatenate([np.flatnonzero(difference == d)[:2] for d in (1, 2, 3)])
    u, v = u[indices], v[indices]
    X, Y = np.meshgrid(np.linspace(0, env.size, 80), np.linspace(0, env.size, 80))
    valid = np.ones(X.shape, dtype=bool)
    for b in env.wall_blocks:
        valid &= ~((X >= b.xmin) & (X <= b.xmax) & (Y >= b.ymin) & (Y <= b.ymax))
    states = np.stack([X[valid], Y[valid]], axis=-1)
    z = learner.embed_B(v)
    f = learner.embed_F(u, z)
    actions = jnp.argmax(jnp.einsum("nda,nd->na", f, z), axis=-1)
    f_pi = jnp.take_along_axis(f, actions[:, None, None], axis=-1).squeeze(-1)
    densities = np.asarray(f_pi @ learner.embed_B(states).T)
    output = Path("successor_examples") / Path(config["path"]).parent.name
    output.mkdir(parents=True, exist_ok=True)
    panels = []
    for i, density in enumerate(densities):
        grid = np.full(X.shape, np.nan)
        grid[valid] = density
        kind = "Diagonal" if difference[indices[i]] == 3 else "Adjacent"
        fig, _ = visualise_successor_measure(grid, X, Y, env, start=u[i], goal=v[i],
                                             title=f"{kind} rooms — example {i + 1}")
        path = output / f"example_{i + 1}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        panels.append(path)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, path in zip(axes.flat, panels):
        ax.imshow(plt.imread(path))
        ax.axis("off")
    fig.suptitle(f"{Path(config['path']).parent.name} / {Path(config['path']).name}")
    fig.tight_layout()
    fig.savefig(output / "overview.png", dpi=130)
    plt.close(fig)
    np.savez(output / "examples.npz", starts=u, goals=v, densities=densities, states=states)
    (output / 'metadata.json').write_text(json.dumps({
        'checkpoint': config['path'], 'doorway_scale': width,
        'environment_seed': int(config['seed']) + 4,
        'policy': 'greedy', 'grid_size': 80,
        'geometry_source': geometry_source,
        'wall_blocks': [[b.xmin, b.ymin, b.xmax, b.ymax] for b in env.wall_blocks],
        'sampling': 'six different-room pairs: two per room XOR category',
    }, indent=2))
    print(f"Checkpoint: {config['path']}")
    print(f"Saved {len(panels)} examples to {output.resolve()}")


if __name__ == "__main__":
    main()
