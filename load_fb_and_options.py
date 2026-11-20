# example for Sam of how to load and use the FB learner
# python load_fb_and_options.py --FB_path experiments/fourrooms_doorways/runs/cauchy_standard_fb_seed0 --output_dir results/fb_options_demo --start 0.36 0.227 --goal 0.64 0.227 --goal 0.18 0.73
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agents import OptionAgent
from fb_navigation.agents.forward_backward_jax import FBLearner
from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms
from fb_navigation.environments.continuous_four_rooms_jax import Point


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--doorway_scale", type=float, default=1.0)
    parser.add_argument("--FB_path", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--start", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--goal", nargs=2, type=float, action="append", required=True,
                        metavar=("X", "Y"), help="Option target. Repeat for each option.")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--step_limit", type=int, default=30)
    parser.add_argument("--epsilon", type=float, default=0.02,
                        help="Probability of choosing a uniformly random action (default: 0.02).")
    args = parser.parse_args()
    if not 0.0 <= args.epsilon <= 1.0:
        parser.error("epsilon must be between 0 and 1")
    if args.step_limit < 1:
        parser.error("--step_limit must be positive")
    return args


if __name__ == "__main__":
    args = arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = AdjustableFourRooms(
        doorway_scale=args.doorway_scale,
        threshold_distance=args.threshold,
        include_stay_action=False,
        seed=args.seed + 3,
    )

    # to load the fb
    run = Path(args.FB_path)
    learner = FBLearner.from_checkpoint(run / "epoch_0800.pkl", device=args.device)

    # reset samples the outer task goal. The option goals are separate.
    start = env.reset(state=Point(*args.start) if args.start is not None else None)
    goals = jnp.asarray(args.goal, dtype=start.dtype)
    for coords in [start, *goals]:
        if not bool(jnp.all((0 < coords) & (coords < env.size))):
            raise SystemExit(f"Start and option goals must be inside (0, {env.size}).")
        if any(bool(wall.contains(Point(*coords))) for wall in env.wall_blocks):
            raise SystemExit(f"Point {coords} is inside a wall.")

    # note that the learner applies the rbf features itself

    # backwards
    embedded_goals = learner.embed_B(goals)

    # A discrete SAC policy would pick an option ID here.
    option_id = 0
    goal = goals[option_id]
    z = embedded_goals[option_id:option_id + 1]
    threshold = env.threshold_distance
    if bool(jnp.all(jnp.abs(start - goal) <= threshold)):
        raise SystemExit("Choose a start outside the first option's goal tolerance.")

    # forward
    f = learner.embed_F(start[None, :], z)

    # Q values for each action
    values = jnp.einsum("bda,bd->ba", f, z)
    print(f"Start {start}, option {option_id} goal {goal}, task goal {env.goal}")
    print(f"Initial action scores {values[0]}, greedy action {int(jnp.argmax(values[0]))}")

    # Run the option with frozen FB parameters and collect outer task rewards.
    agent = OptionAgent(learner, args.epsilon)
    forward_params = learner.state["params"]["f"]
    key = jax.device_put(jax.random.PRNGKey(args.seed), learner.device)
    obs = start
    gamma = 0.99  # SAC discount per primitive step
    discounted_return = 0.0
    reason = "option step limit"
    for step in range(args.step_limit):
        key, action_key = jax.random.split(key)
        states = jax.device_put(obs[None, :], learner.device)
        action = agent.act(forward_params, states, z, action_key)[0]
        obs, reward, terminated, _ = env.step(int(action))
        discounted_return += gamma ** step * float(reward)
        if terminated:
            reason = "task ended"
            break
        if bool(jnp.all(jnp.abs(obs - goal) <= threshold)):
            reason = "option goal reached"
            break

    # Include task_goal in SAC's observation when it varies between episodes.
    # Bootstrap with gamma ** duration if the task continues.
    duration = step + 1
    np.savez(output_dir / "option_transition.npz", start_obs=start, task_goal=env.goal,
             option_id=option_id,
             discounted_task_return=discounted_return, next_obs=obs,
             duration=duration, terminated=bool(terminated))
    print(f"Stopped at {obs} after {duration} steps ({reason}).")
