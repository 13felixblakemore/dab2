import unittest
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from adjustable_four_rooms import AdjustableFourRooms
from compare_navigation_stages import extract_probes,mc_rollout,summary_for


class NavigationStageTests(unittest.TestCase):
    def test_stage_states_and_remaining_budget(self):
        path=np.array([[.225,.325],[.225,.475],[.25,.525],[.3,.55],[.35,.6]]+[[.35,.6]]*26,np.float32)
        trace=dict(position=path[:30,None],following=path[1:31,None],hits=np.array([151]))
        pairs=dict(categories=np.array([2]),start_rooms=np.array([0]),goal_rooms=np.array([3]),goals=np.array([[.9,.8]]))
        rows,availability=extract_probes({400:trace,800:trace},pairs,2)
        selected=[r for r in rows if r['source_epoch']==400]
        self.assertEqual([r['source_step'] for r in selected],[0,1,4])
        self.assertEqual([r['remaining_budget'] for r in selected],[30,29,26])
        np.testing.assert_array_equal(selected[1]['state'],path[1])
        self.assertEqual(availability[400]['diagonal_trials'],1)
        self.assertEqual(selected[2]['committed_room'],1)

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

    def test_summary_uses_remaining_budget_without_extending_it(self):
        record=dict(q=np.array([[1.,0.,0.,0.]]),advantages=np.array([[1.,-1.,-1.,-1.]]),
            hits=np.array([[3,5]]),collisions=np.array([[[1,1,1],[2,2,1]]]),
            repeats=np.zeros((1,2,3)),longest=np.zeros((1,2,3)))
        rows=[dict(remaining_budget=4,trial=400)]
        result=summary_for(record,rows,np.array([True]),np.array([[False,False,False,False]]),2)
        self.assertEqual(result['success_rate'],.5)
        self.assertAlmostEqual(result['wall_action_fraction'],2/7)
        self.assertAlmostEqual(result['discounted_return'],.99**2/2)


if __name__=='__main__':unittest.main()
