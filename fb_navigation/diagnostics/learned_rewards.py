"""Frozen learned-reward bounds and continuing forced-action rollouts.

Rewards are evaluated at the current state. Goal entry is recorded without
stopping or resetting the continuing return.
"""
import jax
import jax.numpy as jnp
import numpy as np

from fb_navigation.environments.continuous_four_rooms_jax import Point, _move


def reward_bounds(learner, tasks):
    """Conservative global interval bounds: normalized RBF inputs form a simplex."""
    layers = jax.device_get(learner.state['params']['b'])
    lower = layers[0]['weight'].min(0) + layers[0]['bias']
    upper = layers[0]['weight'].max(0) + layers[0]['bias']
    lower, upper = np.maximum(lower, 0), np.maximum(upper, 0)
    for layer in layers[1:-1]:
        w, b = layer['weight'].astype(np.float64), layer['bias']
        pos, neg = np.maximum(w, 0), np.minimum(w, 0)
        lower, upper = np.maximum(lower @ pos + upper @ neg + b, 0), np.maximum(upper @ pos + lower @ neg + b, 0)
    w = layers[-1]['weight'].astype(np.float64) @ tasks.T
    b = layers[-1]['bias'].astype(np.float64) @ tasks.T
    pos, neg = np.maximum(w, 0), np.minimum(w, 0)
    return np.stack([lower @ pos + upper @ neg + b, upper @ pos + lower @ neg + b], axis=1)


def make_rollout(learner, env, horizons, policy, trace=False):
    """Reward at s_t, forced a_0, then frozen policy; never stop/reset on a goal."""
    horizons = jnp.asarray(horizons, jnp.int32)
    horizon = int(np.max(horizons))
    actions = jnp.asarray([(a.x, a.y) for a in env.actions], jnp.float32)
    walls = tuple(env.wall_blocks)
    gamma = learner.discount_factor

    @jax.jit
    def rollout(params, starts, goals, first, keys, corrected_tasks):
        tasks = learner.backward_network.apply(params['b'], learner.featuriser.transform(goals))
        n = len(starts)
        def step(carry, t):
            position, keys, total, compensation, saved, collision_count, first_blocked, hits, max_reward = carry
            features = learner.featuriser.transform(position)
            b = learner.backward_network.apply(params['b'], features)
            reward = jnp.stack([jnp.sum(b*tasks, axis=1), jnp.sum(b*corrected_tasks, axis=1)], axis=1)
            # Compensated float32 summation without changing native movement RNG precision.
            addition = gamma**t * reward - compensation
            new_total = total + addition
            compensation = (new_total-total) - addition
            total = new_total
            saved = jnp.where((t+1 == horizons)[None, :, None], total[:, None, :], saved)
            max_reward = jnp.maximum(max_reward, jnp.abs(reward))
            splits = jax.vmap(lambda k: jax.random.split(k, 4))(keys)
            keys, _, ak, mk = [splits[:, i] for i in range(4)]
            f = learner.forward_network.apply(params['f'], features, tasks)
            q = jnp.einsum('bda,bd->ba', f, tasks)
            if policy == 'softmax200':
                selected = jax.vmap(lambda k, logits: jax.random.categorical(k, logits))(ak, q/learner.temperature)
            elif policy == 'wall_masked':
                def mask(p):
                    return jax.vmap(lambda a: _move(Point(*p), Point(*a), jax.random.PRNGKey(0),
                        0., env.size, walls)[2])(actions)
                blocked = jax.vmap(mask)(position)
                scores = jnp.where(blocked, -jnp.inf, q)
                selected = jnp.argmax(jnp.where(jnp.all(blocked, axis=1)[:, None], q, scores), axis=1)
            else:
                selected = jnp.argmax(q, axis=1)
            selected = jnp.where(t == 0, first, selected)
            def move(p, a, key):
                nxt, _, blocked = _move(Point(*p), Point(*actions[a]), key, env._noise, env.size, walls)
                return jnp.asarray([nxt.x, nxt.y]), blocked
            following, blocked = jax.vmap(move)(position, selected, mk)
            first_blocked = jnp.where(t == 0, blocked, first_blocked)
            collision_count += blocked.astype(jnp.int32)
            reached = jnp.all(abs(following-goals) <= env.threshold_distance, axis=1)
            hits = jnp.where((hits > horizon) & reached, t+1, hits)
            carry = (following, keys, total, compensation, saved, collision_count, first_blocked, hits, max_reward)
            return carry, dict(position=position, following=following, reward=reward, action=selected, blocked=blocked) if trace else None
        zeros = jnp.zeros((n, 2), jnp.float32)
        carry = (starts, keys, zeros, zeros, jnp.zeros((n, len(horizons), 2), jnp.float32),
            jnp.zeros(n, jnp.int32), jnp.zeros(n, bool), jnp.full(n, horizon+1, jnp.int32), zeros)
        final, history = jax.lax.scan(step, carry, jnp.arange(horizon))
        result = dict(returns=final[4], wall_actions=final[5], first_blocked=final[6], hits=final[7], max_abs_reward=final[8])
        if trace: result['trace'] = history
        return result
    return rollout
