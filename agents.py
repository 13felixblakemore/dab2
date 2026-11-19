import jax
import jax.numpy as jnp


class OptionAgent:
    def __init__(self, learner, epsilon):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        self.learner = learner
        self.epsilon = float(epsilon)

    def embed_goals(self, backward_params, goals):
        return self.learner.backward_network.apply(
            backward_params, self.learner.featuriser.transform(goals)
        )

    def act(self, forward_params, states, tasks, key):
        """Select one action per state and its matching goal embedding."""
        features = self.learner.featuriser.transform(states)
        forward = self.learner.forward_network.apply(forward_params, features, tasks)
        scores = jnp.einsum("bda,bd->ba", forward, tasks)
        greedy = jnp.argmax(scores, axis=-1)
        if self.epsilon == 0.0:
            return greedy

        explore_key, action_key = jax.random.split(key)
        explore = jax.random.bernoulli(explore_key, self.epsilon, shape=greedy.shape)
        random_actions = jax.random.randint(action_key, greedy.shape, 0, scores.shape[-1])
        return jnp.where(explore, random_actions, greedy)
