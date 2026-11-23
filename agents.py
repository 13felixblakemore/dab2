import math

import jax
import jax.numpy as jnp

from fb_navigation.environments.continuous_four_rooms_jax import sample_displacements


class OptionAgent:
    def __init__(self, learner, epsilon, *, policy="epsilon-greedy", temperature=None):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        self.learner = learner
        self.epsilon = float(epsilon)
        if policy not in ("epsilon-greedy", "boltzmann"):
            raise ValueError("Unknown option policy")
        self.policy = policy
        self.temperature = temperature
        if policy == "boltzmann":
            self.temperature = learner.temperature if temperature is None else temperature
            if not math.isfinite(self.temperature) or self.temperature <= 0:
                raise ValueError("Boltzmann temperature must be finite and positive")

    def embed_goals(self, backward_params, goals):
        return self.learner.backward_network.apply(
            backward_params, self.learner.featuriser.transform(goals)
        )

    def act(self, forward_params, states, tasks, key):
        """Select one action per state and its matching goal embedding."""
        features = self.learner.featuriser.transform(states)
        if getattr(self.learner, "continuous", False):
            def choose(feature, task, key):
                if self.policy == "boltzmann":
                    return self.learner.select_displacements(
                        forward_params, feature[None], task[None], key,
                        softmax=True, temperature=self.temperature,
                    )[0]
                greedy_key, explore_key, action_key = jax.random.split(key, 3)
                greedy = self.learner.select_displacements(
                    forward_params, feature[None], task[None], greedy_key
                )[0]
                if self.epsilon == 0.0:
                    return greedy
                explore = jax.random.bernoulli(explore_key, self.epsilon)
                random_action = sample_displacements(action_key, self.learner.action_limit)
                return jnp.where(explore, random_action, greedy)

            return jax.vmap(choose)(features, tasks, jax.random.split(key, len(states)))

        forward = self.learner.forward_network.apply(forward_params, features, tasks)
        scores = jnp.einsum("bda,bd->ba", forward, tasks)
        if self.policy == "boltzmann":
            return jax.random.categorical(key, scores / self.temperature, axis=-1)
        greedy = jnp.argmax(scores, axis=-1)
        if self.epsilon == 0.0:
            return greedy

        explore_key, action_key = jax.random.split(key)
        explore = jax.random.bernoulli(explore_key, self.epsilon, shape=greedy.shape)
        random_actions = jax.random.randint(action_key, greedy.shape, 0, scores.shape[-1])
        return jnp.where(explore, random_actions, greedy)
