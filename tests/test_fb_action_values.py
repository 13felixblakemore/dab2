"""Regression checks for noisy Bellman integration and forced-action MC."""
import math
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from fb_navigation.environments.continuous_four_rooms_jax import ContinuousWorld, ContinuousFourRooms
from fb_navigation.evaluation.evaluate_fb_action_values import make_expected_backup, make_returns, detailed_collector
from fb_navigation.training.run_fb_jax import make_collector


def linear_learner():
    # B(x,y)=x; F(x,y,z,a)=x, identically across actions.
    return SimpleNamespace(n_actions=4, embed_dim=1, temperature=200., discount_factor=.99,
        featuriser=SimpleNamespace(transform=lambda x: x),
        backward_network=SimpleNamespace(apply=lambda params, x: x[:, :1]),
        forward_network=SimpleNamespace(apply=lambda params, x, z:
            jnp.broadcast_to(x[:, :1, None], (len(x), 1, 4))))


class ActionValueTests(unittest.TestCase):
    def test_expected_backup_integrates_clipped_gaussian_noise(self):
        learner = linear_learner()
        env = ContinuousWorld(size=1., step_size=.05, movement_noise=.01, include_stay_action=False)
        fn = make_expected_backup(learner, env, 8192, np.eye(1), np.eye(1))
        params = dict(f=None, b=None)
        result = fn(params, params, jnp.asarray([[.005, .5]]), jnp.asarray([[1., .5]]), jax.random.split(jax.random.PRNGKey(20), 1))
        # For UP, next x is max(0,N(.005,.01^2)); upper clipping negligible.
        mu, sigma = .005, .01
        alpha = mu/sigma
        expected_x = sigma*math.exp(-alpha**2/2)/math.sqrt(2*math.pi)+mu*.5*(1+math.erf(alpha/math.sqrt(2)))
        self.assertAlmostEqual(float(result['backup'][0, 2]), mu+.99*expected_x, delta=.0003)
        self.assertGreater(float(result['clipped_probability'][0, 2]), .28)
        self.assertGreater(float(result['backup'][0, 2]), mu+.99*mu+.001)
        self.assertAlmostEqual(float(result['backup'][0, 1]), .005+.99*.055, delta=.0004)

    def test_mc_forces_action_and_counts_current_state_first(self):
        learner = linear_learner()
        env = ContinuousWorld(size=1., step_size=.05, movement_noise=0., include_stay_action=False)
        fn = make_returns(learner, env, 2)
        params = dict(f=None, b=None)
        result = fn(params, params, jnp.asarray([[.4, .5], [.4, .5]]), jnp.asarray([[.45], [.45]]),
                    jnp.asarray([1, 0]), jax.random.split(jax.random.PRNGKey(2), 2))
        expected = np.asarray([.4*.45+.99*.45*.45, .4*.45+.99*.35*.45])
        np.testing.assert_allclose(result['returns'], expected, atol=1e-7)
        np.testing.assert_allclose(result['half_horizon_returns'], [.4*.45]*2, atol=1e-7)

    def test_instrumentation_preserves_training_collector(self):
        env = ContinuousFourRooms(include_stay_action=False)
        keys = tuple(jax.random.split(jax.random.PRNGKey(9), 3))
        old_keys, old = jax.jit(make_collector(env, 4, 30))(*keys)
        new_keys, new = detailed_collector(env, 4)(*keys)
        for a, b in zip(old_keys, new_keys):
            np.testing.assert_array_equal(a, b)
        for name in old:
            np.testing.assert_array_equal(old[name], new[name])
        self.assertFalse(np.any(np.any(np.asarray(new['obs']) != np.asarray(new['obs_next']), axis=1)&np.asarray(new['collision'])))


if __name__ == '__main__':
    unittest.main()
