import unittest
import numpy as np
from compare_bottleneck_policies import estimate,simulate_fb,path_statistics


class PolicyDiscountControlTests(unittest.TestCase):
    def test_discounts_reuse_first_hits_and_exclusions(self):
        first=np.array([[0,2,-1],[1,3,4]])
        eligible=np.array([[True,True,True],[True,False,True]])
        one,pair_one=estimate(first,eligible,1,1.)
        discount,pair_discount=estimate(first,eligible,1,.99)
        np.testing.assert_allclose(one,[1.,1.,.5])
        np.testing.assert_allclose(discount,[(1+.99)/2,.99**2,.99**4/2])
        self.assertTrue(np.all(pair_discount<=pair_one))
        np.testing.assert_array_equal(first,[[0,2,-1],[1,3,4]])

    def test_tabular_rollout_and_goal_exclusion(self):
        graph=np.array([[-1,1],[0,2],[1,-1]])
        terminal=np.array([[False,False,True]])
        policy=np.array([[[0.,1.],[0.,1.],[0.,1.]]])
        first,hit,blocked=simulate_fb(graph,terminal,np.array([0]),np.array([0]),np.arange(3),policy,4,5,3)
        np.testing.assert_array_equal(first,np.broadcast_to([0,1,-1],first.shape))
        np.testing.assert_array_equal(hit,2);np.testing.assert_array_equal(blocked,0)
        stats=path_statistics(hit,5,np.array([2]),4)
        self.assertEqual(stats['success_rate'],1.)
        self.assertEqual(stats['mean_successful_length_ratio_to_bfs'],1.)

    def test_censoring_is_reported(self):
        graph=np.array([[-1,1],[0,2],[1,-1]])
        terminal=np.array([[False,False,True]])
        policy=np.array([[[1.,0.],[1.,0.],[1.,0.]]])
        first,hit,blocked=simulate_fb(graph,terminal,np.array([0]),np.array([0]),np.arange(3),policy,4,5,3)
        np.testing.assert_array_equal(hit,6)
        stats=path_statistics(hit,5,np.array([2]),4)
        self.assertEqual(stats['success_rate'],0.)
        self.assertIsNone(stats['mean_successful_steps'])
        self.assertEqual(stats['mean_capped_steps'],5.)
        np.testing.assert_array_equal(blocked,5)


if __name__=='__main__':unittest.main()
