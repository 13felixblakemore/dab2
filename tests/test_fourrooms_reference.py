import json
import unittest
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from fb_navigation.environments.continuous_four_rooms_jax import ContinuousFourRooms
from fb_navigation.evaluation.fourrooms_reference_diagnostics import make_pairs, coordinate_room, make_rollout, summarize
from fb_navigation.training.run_fourrooms_reference import CONFIG, create_learner


class ReferenceFourRoomsTests(unittest.TestCase):
    def test_config_preserves_dynamics_and_uses_released_objective(self):
        config = json.loads(CONFIG.read_text())
        env = ContinuousFourRooms(**config['environment'])
        self.assertEqual(len(env.actions), 4)
        self.assertEqual(float(env.actions[0].x), -.05)
        self.assertEqual(env._noise, .005)
        self.assertEqual(env.threshold_distance, .1)
        self.assertFalse(env.explorable)
        self.assertEqual(len(env.wall_blocks), 7)
        config['learner']['buffer_size'] = 512
        learner = create_learner(config, 0, 'cpu')
        self.assertEqual(learner.b_regularization, 'touati_continuous')
        self.assertFalse(learner.goal_terminated)
        self.assertEqual(learner.zero_z_p, 0)
        self.assertIsNone(learner.max_grad_norm)
        self.assertEqual(learner.embed_F([[.2, .2]], np.ones((1, 50))).shape, (1, 50, 4))
        self.assertEqual(config['epochs']*config['cycles']*config['updates'], 800000)

    def test_pairs_strata_deterministic_and_bfs(self):
        env = ContinuousFourRooms(include_stay_action=False)
        first, second = make_pairs(env, 8, 9), make_pairs(env, 8, 9)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(np.bincount(first['categories']), [8, 8, 8])
        self.assertTrue((first['bfs_lengths'] >= 1).all())
        self.assertTrue((np.max(abs(first['starts']-first['goals']), axis=1) > .1).all())
        np.testing.assert_array_equal(coordinate_room(jnp.asarray([[.1, .1], [.1, .8], [.8, .1], [.8, .8], [.58, .125]])), [0, 1, 2, 3, -1])

    def test_completed_door_crossings_and_wall_rejections(self):
        env = ContinuousFourRooms(include_stay_action=False)
        env._noise = 0  # Deterministic test only; runtime env remains unchanged.
        learner = SimpleNamespace(
            state=dict(params=dict(f=0, b=0)),
            featuriser=SimpleNamespace(transform=lambda x: x),
            forward_network=SimpleNamespace(apply=lambda p, x, z: jnp.tile(jnp.array([[[0., 1., 0., 0.]]]), (len(x), 1, 1))),
            backward_network=SimpleNamespace(apply=lambda p, x: jnp.ones((len(x), 1))))
        rollout = make_rollout(learner, env, 10, epsilon=0)
        starts = jnp.array([[.525, .125], [.525, .325]])
        goals = jnp.array([[.9, .125], [.9, .325]])
        hit, collisions, doors, path = rollout(learner.state['params'], starts, goals, jax.random.split(jax.random.PRNGKey(0), 2))
        self.assertLessEqual(int(hit[0]), 10)
        self.assertEqual(int(doors[0, 0]), 1)
        self.assertEqual(int(collisions[0]), 0)
        self.assertEqual(int(hit[1]), 11)
        self.assertEqual(int(collisions[1]), 10)
        self.assertEqual(int(doors[1].sum()), 0)

    def test_wall_percentage_and_bfs_conditioning(self):
        pairs = dict(categories=np.array([0, 1, 2]), bfs_lengths=np.array([2, 10, 40]))
        rows = summarize(np.array([4, 31, 31]), np.array([1, 15, 0]), np.zeros((3, 4), int), pairs, 30)
        self.assertEqual(rows[0]['wall_action_percent'], 25)
        self.assertEqual(rows[0]['mean_success_steps_over_bfs'], 2)
        self.assertEqual(rows[1]['success_on_bfs_feasible'], 0)
        self.assertIsNone(rows[2]['success_on_bfs_feasible'])


if __name__ == '__main__':
    unittest.main()
