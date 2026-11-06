import unittest
import numpy as np
from fb_navigation.diagnostics.investigate_doorway_decline import failure_stages, room_distance, fixed_probes
from fb_navigation.diagnostics.investigate_doorway_decline import trace_function


class DeclineTests(unittest.TestCase):
    def test_room_distance_accepts_both_diagonal_routes(self):
        self.assertEqual(room_distance(0, 3), 2)
        self.assertEqual(room_distance(1, 3), 1)
        self.assertEqual(room_distance(2, 3), 1)

    def test_failure_stages_and_repeated_actions(self):
        # Three adjacent BL -> BR tasks: no approach, approach without crossing,
        # and crossing into the goal room then failing to reach the goal.
        position = np.array([[[.1,.1],[.525,.125],[.525,.125]],
                             [[.1,.1],[.525,.125],[.60,.125]],
                             [[.1,.1],[.525,.125],[.675,.125]]], np.float32)
        following = position.copy(); following[1:,2] = [.675,.125]
        trace = dict(position=position, following=following, hits=np.array([4,4,4]),
                     action=np.ones((3,3),int), blocked=np.array([[0,1,0],[0,1,0],[0,1,1]],bool))
        pairs = dict(start_rooms=np.array([0,0,0]),goal_rooms=np.array([2,2,2]))
        rows = failure_stages(trace,pairs,3,2)
        self.assertEqual(rows[0]['stage'],'never_approached_useful_door')
        self.assertEqual(rows[1]['stage'],'approached_no_progress_crossing')
        self.assertEqual(rows[1]['longest_identical_wall_run'],3)
        self.assertEqual(rows[1]['repeated_wall_actions'],2)
        self.assertEqual(rows[2]['stage'],'entered_goal_room_failed_goal_control')
        trace['hits'][1] = 2
        self.assertEqual(failure_stages(trace,pairs,3,2)[1]['stage'],'success')

    def test_common_probes_are_reproducible_and_use_both_paths(self):
        first = dict(hits=np.array([2]),position=np.zeros((4,1,2),np.float32))
        second = dict(hits=np.array([4]),position=np.ones((4,1,2),np.float32))
        a=fixed_probes(first,second)
        np.testing.assert_array_equal(a,fixed_probes(first,second))
        np.testing.assert_array_equal(a[0,:8],0)
        np.testing.assert_array_equal(a[0,8:],1)

    def test_oracle_mask_is_separate_and_changes_blocked_greedy_choice(self):
        from types import SimpleNamespace
        import jax
        import jax.numpy as jnp
        from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms
        env = AdjustableFourRooms(doorway_scale=2, include_stay_action=False)
        env._noise = 0
        learner = SimpleNamespace(featuriser=SimpleNamespace(transform=lambda x:x),
            backward_network=SimpleNamespace(apply=lambda p,x:jnp.ones((len(x),1))),
            forward_network=SimpleNamespace(apply=lambda p,x,z:jnp.tile(jnp.array([[[1.,2.,0.,0.]]]),(len(x),1,1))))
        args = (dict(f=0,b=0),jnp.array([[.525,.325]]),jnp.array([[.9,.325]]),jax.random.split(jax.random.PRNGKey(0),1))
        normal = trace_function(learner,env,1,epsilon=0)(*args)
        masked = trace_function(learner,env,1,mask_wall_actions=True,epsilon=0)(*args)
        self.assertTrue(bool(normal['blocked'][0,0]))
        self.assertFalse(bool(masked['blocked'][0,0]))
        self.assertEqual(int(normal['action'][0,0]),1)
        self.assertEqual(int(masked['action'][0,0]),0)


if __name__ == '__main__':
    unittest.main()
