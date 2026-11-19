import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agents import OptionAgent
from fb_navigation.agents.forward_backward_jax import FBLearner
from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms
from fb_navigation.environments.continuous_four_rooms_jax import Point, _move


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--doorway_scale", type=float, default=1.0)
    parser.add_argument("--FB_path", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--goal", nargs=2, type=float, action="append", required=True,
                        metavar=("X", "Y"), help="Option target. Repeat for each option.")
    parser.add_argument("--step_limit", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--trials_per_start", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.02,
                        help="Probability of choosing a uniformly random action (default: 0.02).")
    args = parser.parse_args()
    if args.step_limit < 0 or args.num_samples < 1 or args.trials_per_start < 1:
        parser.error("step_limit must be nonnegative and sample and trial counts must be positive")
    if not 0.0 <= args.epsilon <= 1.0:
        parser.error("epsilon must be between 0 and 1")
    return args


def rollout(learner, env, starts, goals, step_limit, threshold, key, trials_per_start=1,
            *, epsilon=0.02):
    """Return success flags indexed by (option, start, trial)."""
    walls = tuple(env.wall_blocks)
    actions = jnp.asarray([(action.x, action.y) for action in env.actions])
    shape = (len(goals), len(starts), trials_per_start)
    agent = OptionAgent(learner, epsilon)

    def move(position, action, move_key):
        next_position, _, _ = _move(
            Point(*position), Point(*actions[action]), move_key, env._noise, env.size, walls
        )
        return jnp.stack((next_position.x, next_position.y))

    @jax.jit
    def run(params, starts, goals, key):
        tasks = agent.embed_goals(params["b"], goals)
        tasks = jnp.repeat(tasks, shape[1] * shape[2], axis=0)
        targets = jnp.repeat(goals, shape[1] * shape[2], axis=0)
        positions = jnp.broadcast_to(starts[None, :, None, :], (*shape, 2)).reshape(-1, 2)

        def reached(position):
            return jnp.all(jnp.abs(position - targets) <= threshold, axis=-1)

        def step(_, carry):
            positions, successes, key = carry
            key, action_key, move_key = jax.random.split(key, 3)
            selected = agent.act(params["f"], positions, tasks, action_key)
            move_keys = jax.random.split(move_key, positions.shape[0])
            next_positions = jax.vmap(move)(positions, selected, move_keys)
            positions = jnp.where(successes[:, None], positions, next_positions)
            return positions, successes | reached(positions), key

        _, successes, _ = jax.lax.fori_loop(
            0, step_limit, step, (positions, reached(positions), key)
        )
        return successes.reshape(shape)

    return run(learner.state["params"], starts, goals, key)

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = AdjustableFourRooms(
        doorway_scale=args.doorway_scale,
        threshold_distance=args.threshold,
        include_stay_action=False,
        seed=args.seed + 3,
    )

    run = Path(args.FB_path)
    learner = FBLearner.from_checkpoint(run / "epoch_0800.pkl", device=args.device)

    starts = env.sample_points(args.num_samples)
    goals = jnp.asarray(args.goal, dtype=starts.dtype)
    successes = rollout(
        learner, env, starts, goals, args.step_limit, env.threshold_distance,
        jax.random.PRNGKey(args.seed), args.trials_per_start, epsilon=args.epsilon,
    )
    goal_coordinates = np.asarray(goals)
    success_rate_by_option = np.asarray(successes.mean(axis=(1, 2)))
    np.savez(
        output_dir / "option_success.npz",
        starts=np.asarray(starts),
        goals=goal_coordinates,
        epsilon=args.epsilon,
        successes=np.asarray(successes),
        success_rate_by_start=np.asarray(successes.mean(axis=2)),
        success_rate_by_option=success_rate_by_option,
    )
    for i, ((x, y), rate) in enumerate(zip(goal_coordinates, success_rate_by_option), start=1):
        print(f"Option {i} ({x:.4f}, {y:.4f}): {rate:.1%}")

if __name__ == "__main__":
    args = arguments()
    main(args)
