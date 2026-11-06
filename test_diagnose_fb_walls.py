"""Checks for the diagnostic mask and trajectory bookkeeping."""
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from continuous_four_rooms_jax import ContinuousWorld, ContinuousFourRooms, Point, WallBlock, _move
from diagnose_fb_walls import nominal_action_mask, make_trace, collision_runs


class WallDiagnosticTests(unittest.TestCase):
    def test_mask_matches_environment_noiseless_collisions(self):
        env = ContinuousFourRooms(include_stay_action=False)
        rng = np.random.default_rng(12)
        positions = jnp.asarray(rng.uniform(0, 1, (150, 2)), dtype=jnp.float32)
        actions = jnp.asarray([(a.x, a.y) for a in env.actions])
        walls = tuple(env.wall_blocks)
        actual = jax.jit(lambda p: nominal_action_mask(p, actions, env.size, walls))(positions)
        def reference(p):
            return jax.vmap(lambda a: ~_move(Point(*p), Point(*a), jax.random.PRNGKey(0),
                                            0., env.size, walls)[2])(actions)
        expected = jax.jit(jax.vmap(reference))(positions)
        np.testing.assert_array_equal(actual, expected)

    def test_mask_changes_wall_action_without_changing_dynamics(self):
        # RIGHT is maximal; UP is the best permitted action at this wall.
        learner = SimpleNamespace(n_actions=4,
            featuriser=SimpleNamespace(transform=lambda x: x),
            forward_network=SimpleNamespace(apply=lambda params, obs, tasks:
                jnp.broadcast_to(jnp.asarray([.1, 1., .5, .2]), (len(obs), 1, 4))))
        env = ContinuousWorld(size=1., step_size=.1, movement_noise=0.,
                              threshold_distance=.05, include_stay_action=False,
                              wall_blocks=[WallBlock(.4, 0., .5, .6)])
        args = (None, jnp.asarray([[.35, .3]]), jnp.asarray([[.8, .3]]),
                jnp.ones((1, 1)), jax.random.split(jax.random.PRNGKey(8), 1))
        hits, original = make_trace(learner, env, 10, False)(*args)
        _, masked = make_trace(learner, env, 10, True)(*args)
        self.assertEqual(int(hits[0]), 11)
        self.assertTrue(np.all(original['action'] == 1))
        self.assertTrue(np.all(original['collision']))
        np.testing.assert_array_equal(original['position'], original['next_position'])
        self.assertEqual(int(masked['action'][0, 0]), 2)
        self.assertFalse(bool(masked['collision'][0, 0]))
        self.assertEqual(int(collision_runs(jax.tree_util.tree_map(np.asarray, original)).max()), 10)

    def test_collision_run_resets_when_action_changes(self):
        record = dict(action=np.asarray([[1, 1, 2, 2, 2, 2]]),
                      collision=np.asarray([[True, True, True, False, True, True]]))
        np.testing.assert_array_equal(collision_runs(record), [[1, 2, 1, 0, 1, 2]])


if __name__ == '__main__':
    unittest.main()
