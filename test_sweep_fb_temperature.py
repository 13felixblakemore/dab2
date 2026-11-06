import unittest
import numpy as np
from sweep_fb_temperature import probabilities,simulate


class TemperatureSweepTests(unittest.TestCase):
    def test_stable_softmax_and_entropy(self):
        q=np.array([[[100001.,100000.,99999.,99998.]]])
        warm=probabilities(q,'softmax',200.);cold=probabilities(q,'softmax',.2)
        np.testing.assert_allclose(warm.sum(axis=-1),1.)
        self.assertTrue(np.isfinite(cold).all())
        self.assertGreater(cold.max(),warm.max())
        np.testing.assert_allclose(probabilities(q,'epsilon'),[[[.985,.005,.005,.005]]])
        np.testing.assert_array_equal(probabilities(q,'greedy'),[[[1,0,0,0]]])

    def test_common_random_numbers_survive_other_trial_completion(self):
        graph=np.array([[-1,1],[0,2],[1,-1]])
        terminal=np.array([[False,False,True],[False,False,True]])
        policy=np.full((2,3,2),.5)
        policy[0]=[0,1]  # first trial finishes immediately along the chain
        slow=policy.copy();slow[0]=[1,0]  # first trial never leaves the wall
        args=(graph,terminal,np.array([0,0]),np.array([0,1]),np.arange(3))
        fast_first,fast_goal,_=simulate(*args,policy,1,30,4)
        slow_first,slow_goal,_=simulate(*args,slow,1,30,4)
        np.testing.assert_array_equal(fast_first[1],slow_first[1])
        self.assertEqual(fast_goal[1],slow_goal[1])
        self.assertEqual(fast_goal[0],2);self.assertEqual(slow_goal[0],31)


if __name__=='__main__':unittest.main()
