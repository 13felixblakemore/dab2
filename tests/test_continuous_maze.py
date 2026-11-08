"""Self-contained maze dynamics, evaluation, and diagnostics."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import jax.numpy as jnp
import numpy as np

from fb_navigation.diagnostics.diagnose_continuous_maze import (
    bootstrap_residual,
    heldout_transitions,
    wilson_interval,
)
from fb_navigation.environments.continuous_maze import ContinuousMaze, goal_reached, intersects
from fb_navigation.training.run_continuous_maze import evaluate


class MazeEnvironmentTests(unittest.TestCase):
    def test_actions_clipping_and_goal_tolerance(self):
        env = ContinuousMaze(movement_noise=0)
        env.goal = np.array([.5, .5])
        for action, expected in enumerate(([.4, .5], [.6, .5], [.5, .6], [.5, .4], [.5, .5])):
            with self.subTest(action=action):
                env.position = [.5, .5]
                position, _, done, info = env.step(action)
                np.testing.assert_allclose(position, expected, atol=np.finfo(np.float32).eps, rtol=0)
                self.assertFalse(done)
                self.assertFalse(info['goes_through_wall'])
        for start, action, expected in (([0., 0.], 0, [0., 0.]), ([1., 1.], 1, [1., 1.])):
            env.position = start
            np.testing.assert_array_equal(env.step(action)[0], expected)
        self.assertTrue(goal_reached([.54, .54], [.5, .5]))
        self.assertFalse(goal_reached([.551, .5], [.5, .5]))
        self.assertTrue(goal_reached([.05, .05], [0, 0]))
        env.position = [.5, .5]
        _, reward, done, _ = env.step(4)
        self.assertEqual(reward, 1.)
        self.assertFalse(done)

    def test_wall_crossings_are_rejected_but_openings_are_passable(self):
        env = ContinuousMaze(movement_noise=0)
        env.goal = np.array([.5, .5])
        for start, expected, collision in (([.2, .2], [.2, .2], True),
                                           ([.7, .8], [.7, .8], True),
                                           ([.2, .5], [.3, .5], False),
                                           ([.7, .5], [.8, .5], False)):
            with self.subTest(start=start):
                env.position = start
                position, _, _, info = env.step(1)
                np.testing.assert_allclose(position, expected, atol=2e-8, rtol=0)
                self.assertEqual(info['goes_through_wall'], collision)

    def test_intersection_handles_crossings_endpoints_and_separation(self):
        for a, b, c, d, expected in (
            ([0, 0], [1, 1], [0, 1], [1, 0], True),
            ([0, 0], [1, 0], [1, 0], [1, 1], True),
            ([0, 0], [1, 0], [.5, 0], [2, 0], True),
            ([0, 0], [1, 0], [0, 1], [1, 1], False),
        ):
            with self.subTest(segments=(a, b, c, d)):
                self.assertEqual(intersects(a, b, c, d), expected)

    def test_stay_noise(self):
        env = ContinuousMaze(4)
        samples = []
        env.goal = np.array([0, 0])
        for _ in range(5000):
            env.position = np.array([.5, .5])
            samples.append(env.step(4)[0]-.5)
        samples = np.array(samples)
        np.testing.assert_allclose(samples.std(axis=0), [.01, .01], atol=.0004)
        self.assertLess(abs(np.corrcoef(samples.T)[0, 1]), .05)


class DiagnosticTests(unittest.TestCase):
    def test_evaluation_uses_online_task_and_actions_and_stops(self):
        calls = []
        class Environment:
            def __init__(self, seed):
                self.goal = np.array([.6, .7])
                self.steps = 0
            def reset(self):
                return np.array([.1, .2])
            def step(self, action):
                self.steps += 1
                return np.array([.3, .4]), float(self.steps == 2), False, dict(goes_through_wall=False)
        class Learner:
            def embed_B(self, goal):
                np.testing.assert_array_equal(goal, [[.6, .7]])
                calls.append('online_B')
                return np.array([[2., 3.]])
            def act(self, obs, z, target_network=False):
                self_target = target_network
                assert not self_target
                np.testing.assert_array_equal(z, [[2., 3.]])
                calls.append('online_F')
                return np.array([1])
        # Control epsilon draw only; .02 remains the argument under test.
        with patch('fb_navigation.training.run_continuous_maze.ContinuousMaze', Environment), patch('random.Random.random', return_value=.5):
            result, paths = evaluate(Learner(), 1, 42)
        self.assertEqual(calls, ['online_B', 'online_F', 'online_F'])
        self.assertEqual(result['success_rate'], 1)
        self.assertEqual(result['mean_successful_path_length'], 2)
        self.assertEqual(len(paths[0]['path']), 3)

    def test_failure_uses_exactly_30_steps(self):
        class Environment:
            def __init__(self, seed):
                self.goal = np.zeros(2)
            def reset(self):
                return np.ones(2)
            def step(self, action):
                return np.ones(2), 0., False, dict(goes_through_wall=True)
        learner = SimpleNamespace(embed_B=lambda g: np.ones((1, 2)), act=lambda *a: np.array([0]))
        with patch('fb_navigation.training.run_continuous_maze.ContinuousMaze', Environment):
            result, paths = evaluate(learner, 2, 42)
        self.assertEqual(result['wall_collisions'], 60)
        self.assertIsNone(result['mean_successful_path_length'])
        self.assertTrue(all(len(p['path']) == 31 for p in paths))

    def test_bootstrap_residual_scaling_and_target_choice(self):
        learner = SimpleNamespace(
            state=dict(params=dict(f=2., b=3.), target=dict(f=1., b=1.)),
            featuriser=SimpleNamespace(transform=lambda x: x),
            forward_network=SimpleNamespace(apply=lambda p, x, w: jnp.full((len(x), 1, 5), p)),
            backward_network=SimpleNamespace(apply=lambda p, x: jnp.full((len(x), 1), p)),
            temperature=200., discount_factor=.99)
        batch = dict(obs=jnp.zeros((2, 2)), obs_next=jnp.ones((2, 2)), action=jnp.array([0, 4]))
        result = bootstrap_residual(learner, batch, jnp.ones((2, 2)), jnp.ones((2, 1)))
        self.assertAlmostEqual(float(result['residual_mean']), 5.01, places=5)
        self.assertAlmostEqual(float(result['residual_mse']), 5.01**2, places=4)
        self.assertAlmostEqual(float(result['backup_entropy']), np.log(5), places=6)

    def test_fixed_probes_and_intervals(self):
        first, second = heldout_transitions(3, 65), heldout_transitions(3, 65)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(first['obs'].shape, (65, 2))
        self.assertTrue(np.isin(first['action'], range(5)).all())
        low, high = wilson_interval(830, 1000)
        self.assertLess(low, .83)
        self.assertGreater(high, .83)
        self.assertLess(high-low, .05)


if __name__ == '__main__':
    unittest.main()
