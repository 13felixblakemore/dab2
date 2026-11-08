"""Bellman backups, learned-reward returns, and failure classification."""
from types import SimpleNamespace
import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from fb_navigation.agents.forward_backward_jax import IdentityFeaturiser
from fb_navigation.diagnostics.analyse_fb_rewards_failures import classify_tail
from fb_navigation.diagnostics.learned_rewards import make_rollout
from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms
from fb_navigation.environments.continuous_four_rooms_jax import (
    ContinuousFourRooms,
    ContinuousWorld,
)
from fb_navigation.evaluation.evaluate_fb_action_values import (
    detailed_collector,
    make_expected_backup,
    make_returns,
)
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


def record(points, actions):
    p = np.asarray(points, dtype=np.float32)
    return dict(position=p[None, :-1], next_position=p[None, 1:],
                action=np.asarray(actions, dtype=np.int32)[None],
                collision=np.zeros((1, len(actions)), bool))


class FailureAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.env = ContinuousFourRooms(include_stay_action=False)

    def test_outward_boundary_action_is_pinning(self):
        p = np.repeat([[0., .2]], 51, axis=0)
        labels, d = classify_tail(record(p, np.zeros(50, int)), self.env, 50)
        self.assertEqual(labels[0], 'boundary_pinned')
        self.assertEqual(d['boundary_fraction'][0], 1.)

    def test_repeated_square_route_is_cycling(self):
        corners = np.asarray([[.2, .2], [.25, .2], [.25, .25], [.2, .25]])
        p = corners[np.arange(53) % 4]
        a = np.asarray([1, 2, 0, 3])[np.arange(52) % 4]
        labels, d = classify_tail(record(p, a), self.env, 52)
        self.assertEqual(labels[0], 'cycling')
        self.assertGreater(d['revisit_fraction'][0], .5)

    def test_straight_slow_progress_is_not_cycling(self):
        p = np.column_stack((.1+np.arange(51)*.01, np.repeat(.2, 51)))
        labels, d = classify_tail(record(p, np.ones(50, int)), self.env, 50)
        self.assertEqual(labels[0], 'wandering_or_slow')
        self.assertGreater(d['tail_path_efficiency'][0], .99)

    def test_stationary_interior_is_confined(self):
        p = np.repeat([[.2, .2]], 51, axis=0)
        labels, _ = classify_tail(record(p, np.ones(50, int)), self.env, 50)
        self.assertEqual(labels[0], 'confined_without_cycle')


class IdentityBackward:
    def apply(self, params, observations):
        return observations


class PreferUpForward:
    def apply(self, params, observations, tasks):
        return jnp.zeros((len(observations), 2, 4), jnp.float32).at[:, 0, 2].set(1.)


class LearnedRewardRolloutTests(unittest.TestCase):
    def setUp(self):
        self.learner = SimpleNamespace(discount_factor=.5, temperature=200.,
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
