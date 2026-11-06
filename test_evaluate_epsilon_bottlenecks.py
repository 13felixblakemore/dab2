import unittest
import jax.numpy as jnp
import numpy as np
from compare_bottleneck_policies import policy_probabilities
from evaluate_epsilon_bottlenecks import doorway_arrivals


class EpsilonBottleneckTests(unittest.TestCase):
    def test_exact_epsilon_policy(self):
        scores=jnp.array([[1.,4.,3.,2.],[5.,1.,0.,-1.]])
        pi=policy_probabilities(scores,'epsilon',200.,.02)
        np.testing.assert_allclose(pi,[[.005,.985,.005,.005],[.985,.005,.005,.005]],rtol=1e-6)
        np.testing.assert_allclose(np.asarray(pi).sum(axis=1),1.)
        np.testing.assert_allclose(policy_probabilities(scores,'epsilon',200.,1.),.25)
        np.testing.assert_array_equal(policy_probabilities(scores,'epsilon',200.,0.),[[0,1,0,0],[1,0,0,0]])
        expected=np.exp(np.asarray(scores)/200);expected/=expected.sum(axis=1,keepdims=True)
        np.testing.assert_allclose(policy_probabilities(scores,'softmax',200.,.02),expected,rtol=1e-6)

    def test_arrival_definition_and_exclusions(self):
        points=np.array([[6.3,1.4],[6.7,1.4],[3.,8.]])/11
        first=np.array([[5,8,0],[10,-1,1]])
        eligible=np.array([[False,True,True],[True,True,True]])
        stats=doorway_arrivals(first,eligible,points,1)
        self.assertEqual(stats['any_door']['mean_first_any_cell'],9.)
        self.assertEqual(stats['any_door']['mean_pooled_cell_arrival'],9.)
        self.assertEqual(stats['any_door']['arrival_rate'],1.)
        self.assertEqual(stats['per_door']['top']['arrival_rate'],0.)
        self.assertIsNone(stats['per_door']['top']['mean_first_any_cell'])


if __name__=='__main__':unittest.main()
