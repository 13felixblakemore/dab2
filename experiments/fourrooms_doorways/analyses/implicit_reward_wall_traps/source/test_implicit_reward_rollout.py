"""Analytic checks for the continuing learned-reward Monte Carlo diagnostic."""
import types
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from adjustable_four_rooms import AdjustableFourRooms
from forward_backward_jax import IdentityFeaturiser
from experiments.maintenance.diagnose_implicit_reward_traps_20261106 import make_rollout


class IdentityBackward:
    def apply(self, params, observations):
        return observations


class PreferUpForward:
    def apply(self, params, observations, tasks):
        return jnp.zeros((len(observations), 2, 4), jnp.float32).at[:, 0, 2].set(1.)


class LearnedRewardRolloutTests(unittest.TestCase):
    def setUp(self):
        self.learner = types.SimpleNamespace(discount_factor=.5, temperature=200.,
            featuriser=IdentityFeaturiser(), backward_network=IdentityBackward(),
            forward_network=PreferUpForward())
        self.env = AdjustableFourRooms(doorway_scale=2, movement_noise=0.,
            threshold_distance=.2, include_stay_action=False)
        self.params = dict(f=None, b=None)

    def rollout(self, states, goals, actions, policy='greedy'):
        fn = make_rollout(self.learner, self.env, [1, 3], policy)
        keys = jax.random.split(jax.random.PRNGKey(123), len(states))
        return jax.device_get(fn(self.params, jnp.array(states, jnp.float32),
            jnp.array(goals, jnp.float32), jnp.array(actions, jnp.int32), keys,
            jnp.array(goals, jnp.float32)))

    def test_current_state_reward_and_forced_initial_action(self):
        result = self.rollout([[.1, .1], [.1, .1]], [[1., 0.], [1., 0.]], [1, 0])
        np.testing.assert_allclose(result['returns'][:, 0, 0], [.1, .1], atol=1e-6)
        # After RIGHT/LEFT, the learned greedy policy chooses UP, holding x fixed.
        np.testing.assert_allclose(result['returns'][:, 1, 0], [.2125, .1375], atol=1e-6)
        np.testing.assert_array_equal(result['returns'][:, :, 0], result['returns'][:, :, 1])

    def test_goal_entry_does_not_stop_the_return(self):
        result = self.rollout([[.1, .1]], [[.1, .1]], [1])
        self.assertEqual(int(result['hits'][0]), 1)
        # Current rewards .020, .025, .030 continue after the first goal hit.
        self.assertAlmostEqual(float(result['returns'][0, 1, 0]), .04, places=6)

    def test_rejected_initial_move_preserves_current_reward(self):
        result = self.rollout([[.543, .55]], [[1., 0.]], [1])
        self.assertTrue(bool(result['first_blocked'][0]))
        self.assertAlmostEqual(float(result['returns'][0, 1, 0]), .543*1.75, places=6)

    def test_all_continuations_force_the_supplied_first_action(self):
        for policy in ('greedy', 'softmax200', 'wall_masked'):
            with self.subTest(policy=policy):
                fn = make_rollout(self.learner, self.env, [2], policy, trace=True)
                result = fn(self.params, jnp.array([[.1, .1]], jnp.float32),
                    jnp.array([[1., 0.]], jnp.float32), jnp.array([1]),
                    jax.random.split(jax.random.PRNGKey(123), 1), jnp.array([[1., 0.]], jnp.float32))
                self.assertEqual(int(result['trace']['action'][0, 0]), 1)
                np.testing.assert_allclose(result['trace']['following'][0, 0], [.15, .1], atol=1e-6)
                self.assertAlmostEqual(float(result['returns'][0, 0, 0]), .175, places=6)


if __name__ == '__main__':
    unittest.main()
