"""Wall masks, collision streaks, and Four Rooms rollout metrics."""
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from fb_navigation.diagnostics.compare_navigation_stages import mc_rollout
from fb_navigation.diagnostics.diagnose_fb_walls import (
    collision_runs,
    make_trace,
    nominal_action_mask,
)
from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms
from fb_navigation.environments.continuous_four_rooms_jax import (
    ContinuousFourRooms,
    ContinuousWorld,
    Point,
    WallBlock,
    _move,
)


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


class MonteCarloRolloutTests(unittest.TestCase):
    def test_monte_carlo_collision_windows_and_streaks(self):
        env=AdjustableFourRooms(doorway_scale=2,include_stay_action=False);env._noise=0
        learner=SimpleNamespace(featuriser=SimpleNamespace(transform=lambda x:x),
            backward_network=SimpleNamespace(apply=lambda p,x:jnp.ones((len(x),1))),
            forward_network=SimpleNamespace(apply=lambda p,x,z:jnp.tile(jnp.array([[[0.,2.,0.,0.]]]),(len(x),1,1))))
        fn=mc_rollout(learner,env,2,epsilon=0)
        record=fn(dict(f=0,b=0),jnp.array([[.525,.325]]),jnp.array([[.9,.325]]),jnp.array([1]),jax.random.split(jax.random.PRNGKey(0),1))
        np.testing.assert_array_equal(record['collisions'],[[2,2,1]])
        np.testing.assert_array_equal(record['repeats'],[[1,1,0]])
        np.testing.assert_array_equal(record['longest'],[[2,2,1]])
        np.testing.assert_array_equal(record['hits'],[3])


if __name__ == '__main__':
    unittest.main()
