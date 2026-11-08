"""Four Rooms geometry, goal validity, and reference evaluation."""
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from fb_navigation.diagnostics.audit_doorway_goals import component_sizes, point_status
from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms, fixed_pairs
from fb_navigation.environments.continuous_four_rooms_jax import ContinuousFourRooms, Point, _move
from fb_navigation.evaluation.fourrooms import summarize_width2
from fb_navigation.evaluation.fourrooms import (
    coordinate_room,
    make_pairs,
    make_rollout,
    summarize,
)
from fb_navigation.training.run_fourrooms_doorways import retained_checkpoint
from fb_navigation.training.run_fourrooms_reference import CONFIG, create_learner


class EvaluationCommandTests(unittest.TestCase):
    def test_fourrooms_modes_forward_evaluation_arguments(self):
        from fb_navigation.evaluation.fourrooms import main

        for command, entry in (('reference', 'reference_main'), ('width2', 'width2_main')):
            with self.subTest(command=command), patch(
                f'fb_navigation.evaluation.fourrooms.{entry}'
            ) as evaluate:
                options = ['--output', 'new-results', '--help']
                main([command, *options])
                evaluate.assert_called_once_with(options)


class DoorwayTests(unittest.TestCase):
    def test_scale_one_exactly_preserves_baseline(self):
        base = ContinuousFourRooms(include_stay_action=False)
        env = AdjustableFourRooms(include_stay_action=False)
        self.assertEqual(env.wall_blocks, base.wall_blocks)
        self.assertEqual(env.actions, base.actions)
        self.assertEqual(env._noise, base._noise)
        self.assertEqual(env.threshold_distance, base.threshold_distance)
        self.assertFalse(env.explorable)

    def test_larger_door_passes_a_previously_blocked_move(self):
        narrow = AdjustableFourRooms(doorway_scale=1, include_stay_action=False)
        wide = AdjustableFourRooms(doorway_scale=2, include_stay_action=False)
        # y=1/11 lies on the narrow doorway edge, inside the larger opening.
        for env, expected in [(narrow, True), (wide, False)]:
            _, _, collision = _move(Point(.525, 1/11), Point(.15, 0),
                                     jax.random.PRNGKey(0), 0., env.size, tuple(env.wall_blocks))
            self.assertEqual(bool(collision), expected)
        self.assertAlmostEqual(wide.wall_blocks[1].ymin-wide.wall_blocks[0].ymax, 2/11)
        self.assertAlmostEqual(wide.wall_blocks[4].xmin-wide.wall_blocks[3].xmax, 2/11)

    def test_pairs_identical_bfs_recomputed_and_schedule(self):
        narrow = fixed_pairs(AdjustableFourRooms(doorway_scale=1), 10, 271828)
        wide = fixed_pairs(AdjustableFourRooms(doorway_scale=2), 10, 271828)
        for key in ['starts', 'goals', 'categories', 'start_rooms', 'goal_rooms']:
            np.testing.assert_array_equal(narrow[key], wide[key])
        self.assertTrue(np.all(wide['bfs_lengths'] <= narrow['bfs_lengths']))
        self.assertEqual([e for e in range(1, 801) if retained_checkpoint(e, 800, 200)], [200, 400, 600, 800])
        self.assertTrue(retained_checkpoint(250, 250, 200))

    def test_invalid_sizes(self):
        for scale in [0, -1, 3, 4, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                AdjustableFourRooms(doorway_scale=scale)

    def test_smaller_threshold_rejects_success_across_full_wall(self):
        goal = Point(.638, .55)
        for threshold, expected in [(.1, True), (.025, False)]:
            env = AdjustableFourRooms(doorway_scale=2, threshold_distance=threshold,
                                      include_stay_action=False, movement_noise=0.)
            env.reset(state=Point(.543, .50), goal=goal)
            _, reward, done, info = env.step(2)  # UP, alongside the wall.
            self.assertFalse(bool(info['goes_through_wall']))
            self.assertFalse(bool(env._check_on_wall(env._current_position)))
            self.assertFalse(bool(env._check_on_wall(goal)))
            self.assertTrue(bool(env._check_goes_through_wall(env._current_position, goal)))
            self.assertEqual(bool(done), expected)
            self.assertEqual(float(reward), float(expected))

    def test_threshold_preserves_dynamics_and_fixed_endpoints(self):
        original = AdjustableFourRooms(doorway_scale=2, include_stay_action=False)
        strict = AdjustableFourRooms(doorway_scale=2, threshold_distance=.025, include_stay_action=False)
        self.assertEqual(original.wall_blocks, strict.wall_blocks)
        self.assertEqual(original.actions, strict.actions)
        self.assertEqual(original._noise, strict._noise)
        a, b = fixed_pairs(original, 10, 271828), fixed_pairs(strict, 10, 271828)
        for key in ['starts', 'goals', 'categories', 'start_rooms', 'goal_rooms']:
            np.testing.assert_array_equal(a[key], b[key])
        self.assertTrue(np.all(b['bfs_lengths'] >= a['bfs_lengths']))
        self.assertTrue(np.any(b['bfs_lengths'] > a['bfs_lengths']))
        for threshold in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                AdjustableFourRooms(threshold_distance=threshold)


class GoalAuditTests(unittest.TestCase):
    def test_detects_walls_without_confusing_doors(self):
        env=AdjustableFourRooms(doorway_scale=2,include_stay_action=False)
        status=point_status([[.6,.3],[.05,.4],[.6,.125],[.2,.2],[1.01,.5]],env)
        self.assertEqual(status['inside_wall'],2)
        self.assertEqual(status['native_float32_inside_wall'],2)
        self.assertEqual(status['outside_world'],1)
        self.assertEqual(status['bad_indices'],[0,1,4])

    def test_detects_disconnected_graph(self):
        graph=np.array([[1,-1],[0,-1],[-1,-1]])
        self.assertEqual(component_sizes(graph),[2,1])


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


class GoalReachingReportTests(unittest.TestCase):
    def test_censoring_and_wall_denominators(self):
        record=dict(hits=np.array([[1,31],[2,151],[30,31]]),collisions=np.zeros((3,2,3)),repeats=np.zeros((3,2,3)))
        record['collisions'][0,1,0]=3
        pairs=dict(categories=np.arange(3),bfs_lengths=np.array([1,2,30]))
        rows=summarize_width2(record,pairs,'test','epsilon002',30)
        self.assertEqual(rows[0]['success_rate'],.5)
        self.assertEqual(rows[0]['mean_success_steps'],1)
        self.assertAlmostEqual(rows[0]['wall_action_percent'],300/31)
        self.assertEqual(rows[2]['success_rate'],.5)
        self.assertEqual(summarize_width2(record,pairs,'test','epsilon002',150)[2]['success_rate'],1)

    def test_no_success_reports_none(self):
        record=dict(hits=np.full((3,2),151),collisions=np.zeros((3,2,3)),repeats=np.zeros((3,2,3)))
        pairs=dict(categories=np.arange(3),bfs_lengths=np.array([1,2,30]))
        for row in summarize_width2(record,pairs,'test','greedy',150):
            self.assertEqual(row['success_rate'],0)
            self.assertIsNone(row['mean_success_steps'])
            self.assertEqual(row['success_interval95'],[0.,0.])


if __name__ == '__main__':
    unittest.main()
