import unittest
import numpy as np
import jax
from continuous_four_rooms_jax import ContinuousFourRooms, Point, _move
from adjustable_four_rooms import AdjustableFourRooms, fixed_pairs
from run_fourrooms_doorways import retained_checkpoint


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


if __name__ == '__main__':
    unittest.main()
