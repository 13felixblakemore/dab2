"""Synthetic trajectory checks for the failure-analysis classifications."""
import unittest

import numpy as np

from fb_navigation.environments.continuous_four_rooms_jax import ContinuousFourRooms
from fb_navigation.diagnostics.analyse_fb_rewards_failures import classify_tail


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


if __name__ == '__main__':
    unittest.main()
