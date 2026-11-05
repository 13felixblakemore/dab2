"""Four Rooms FB training with the same schedule as run_fb.py.

Only pure functions run inside scans. Collection, replay, updates and target
averaging stay on the learner device, with one loss transfer per epoch.
Random streams differ from PyTorch. Run with --device cpu for a CPU backend.
"""

import argparse
import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange

from continuous_four_rooms_jax import ContinuousFourRooms, Point, _move, _sample_point
from forward_backward_jax import FBLearner, RadialBasisFunction2D


def make_collector(env, episodes=4, steps=30):
    """Pure, sequential rollouts for the original epsilon=1 exploration policy."""
    walls = tuple(env.wall_blocks)
    actions = jnp.asarray([(a.x, a.y) for a in env.actions], dtype=jnp.float32)

    def reset(key):
        key, position_key = jax.random.split(key)
        position = _sample_point(position_key, env.size, walls)
        if env.explorable:
            return key, position, position
        key, goal_key = jax.random.split(key)
        goal = _sample_point(goal_key, env.size, walls)
        return key, position, goal

    def collect(env_key, action_key, learner_key):
        def episode(carry, _):
            env_key, action_key, learner_key = carry
            env_key, position, goal = reset(env_key)
            # Torch draws w once per rollout, but epsilon=1 means it never
            # influences actions. Advance its key without computing unused w.
            learner_key, _ = jax.random.split(learner_key)

            def step(carry, _):
                env_key, action_key, position, goal = carry
                action_key, draw_key = jax.random.split(action_key)
                action = jax.random.randint(draw_key, (), 0, len(env.actions))
                env_key, move_key = jax.random.split(env_key)
                delta = actions[action]
                next_position, _, _ = _move(
                    position, Point(delta[0], delta[1]), move_key,
                    env._noise, env.size, walls,
                )
                done = (jnp.asarray(False) if env.explorable else
                        next_position.is_close_to(goal, diff=env.threshold_distance))
                transition = dict(
                    obs=jnp.asarray((position.x, position.y)), action=action,
                    obs_next=jnp.asarray((next_position.x, next_position.y)),
                    next_actions=jnp.ones((len(env.actions),), dtype=bool),
                )
                # Save the pre-reset transition, including a terminal next_obs.
                env_key, position, goal = jax.lax.cond(
                    done, reset, lambda key: (key, next_position, goal), env_key,
                )
                return (env_key, action_key, position, goal), transition

            (env_key, action_key, _, _), batch = jax.lax.scan(
                step, (env_key, action_key, position, goal), None, length=steps,
            )
            return (env_key, action_key, learner_key), batch

        keys, batch = jax.lax.scan(
            episode, (env_key, action_key, learner_key), None, length=episodes,
        )
        batch = jax.tree_util.tree_map(lambda a: a.reshape((-1,) + a.shape[2:]), batch)
        return keys, batch

    return collect


def make_epoch(env, learner, cycles=25, episodes=4, steps=30, updates=40):
    """Compile a full epoch; replay donation allows storage reuse in place."""
    collect = make_collector(env, episodes, steps)
    capacity = learner.buffer.max_size
    count = episodes * steps
    skip = max(0, count - capacity)

    def epoch(state, storage, ptr, size, env_key, action_key):
        def cycle(carry, _):
            state, storage, ptr, size, env_key, action_key = carry
            (env_key, action_key, learner_key), batch = collect(env_key, action_key, state["key"])
            index = (ptr + jnp.arange(skip, count)) % capacity
            storage = jax.tree_util.tree_map(lambda a, b: a.at[index].set(b[skip:]), storage, batch)
            ptr = (ptr + count) % capacity
            size = jnp.minimum(size + count, capacity)
            state, losses = learner._train_many_impl(dict(state, key=learner_key), storage, size, updates)
            # Exactly one Polyak update after each group of 40 SGD updates.
            target = jax.tree_util.tree_map(
                lambda t, p: (1 - learner.polyak_coef)*p + learner.polyak_coef*t,
                state["target"], state["params"],
            )
            state = dict(state, target=target)
            return (state, storage, ptr, size, env_key, action_key), losses

        return jax.lax.scan(cycle, (state, storage, ptr, size, env_key, action_key), None, length=cycles)

    return jax.jit(epoch, donate_argnums=(1,))


def run_fb(epochs=800, device="cuda", output_dir=".", seed=1, *,
           cycles=25, episodes=4, steps=30, updates=40, env=None, learner=None,
           progress=True):
    """Train and save all per-update losses plus a reloadable JAX checkpoint.

    Optional schedule/environment/learner arguments allow small validation runs.
    Defaults reproduce the Torch runner's hyperparameters and update ordering.
    """
    if min(epochs, cycles, episodes, steps, updates) < 1:
        raise ValueError("Training counts must all be positive")
    env = ContinuousFourRooms(include_stay_action=False, seed=seed) if env is None else env
    if learner is None:
        learner = FBLearner(
            obs_dim=2, n_actions=len(env.get_action_space()), embed_dim=50,
            learning_rate=5e-4, discount_factor=0.99, reg_coef=1,
            polyak_coef=0.95, batch_size=128, hidden_dim=256, num_layers=2,
            buffer_size=1_000_000, device=device, buffer_device=device,
            temperature=200.0, zero_z_p=0.25, seed=seed,
            featuriser=RadialBasisFunction2D(
                x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0), dim=21,
                sigma=0.05, device=device,
            ),
        )
    if learner.n_actions != len(env.actions):
        raise ValueError("Learner and environment action counts must agree")
    if learner.buffer.device != learner.device:
        raise ValueError("Compiled runner requires replay on the learner device")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with jax.default_device(learner.device):
        # CPU validation also exercises the compiled replay path.
        learner.buffer.storage = jax.device_put(learner.buffer.storage, learner.device)
        learner.buffer._host = False
        epoch_fn = make_epoch(env, learner, cycles, episodes, steps, updates)
        carry = (learner.state, learner.buffer.storage,
                 jnp.asarray(learner.buffer.ptr, jnp.int32),
                 jnp.asarray(learner.buffer.size, jnp.int32),
                 jax.device_put(env._rng, learner.device), jax.random.PRNGKey(seed))
        loss_history = np.empty((epochs, cycles, updates), dtype=np.float32)
        pbar = trange(epochs, disable=not progress)
        for epoch in pbar:
            carry, losses = epoch_fn(*carry)
            # One synchronization per epoch, retaining every loss in order.
            loss_history[epoch] = jax.device_get(losses)
            pbar.set_description_str(f"loss {loss_history[epoch, -1, -1]:8.3f}")

        learner.state, learner.buffer.storage, ptr, size, env._rng, _ = carry
        learner.buffer.ptr, learner.buffer.size = int(ptr), int(size)
        learner._sync_networks()

    loss_history = loss_history.reshape(-1)
    np.save(output_dir / "fb_fourrooms_jax_loss.npy", loss_history)
    learner.save(output_dir / "fb_fourrooms_jax.pkl")
    # Preserve the reference measure separately without altering training/RNG.
    reference_rng = np.random.default_rng(seed + 104729)
    indices = reference_rng.integers(learner.buffer.size, size=100_000)
    reference = jax.device_get(learner.buffer.obs[indices])
    np.savez_compressed(
        output_dir / "fb_fourrooms_jax_reference.npz", observations=reference,
        source="sampled_final_replay", seed=seed, replay_size=learner.buffer.size,
        training_updates=int(learner.state["optim"][0]),
        size=env.size, movement_noise=env._noise,
        actions=np.asarray([(a.x, a.y) for a in env.actions]),
        checkpoint_sha256=hashlib.sha256((output_dir / "fb_fourrooms_jax.pkl").read_bytes()).hexdigest(),
        wall_blocks=np.asarray([(b.xmin,b.ymin,b.xmax,b.ymax) for b in env.wall_blocks]),
    )
    return learner, loss_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    run_fb(**vars(args))
