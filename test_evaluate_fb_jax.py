"""Fast analytical checks of evaluation conventions (no checkpoint required)."""
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from continuous_four_rooms_jax import ContinuousWorld
from evaluate_fb_jax import make_mc, bin_ids, predict
from forward_backward_jax import IdentityFeaturiser


class EvaluationTests(unittest.TestCase):
    def learner(self, scores):
        return SimpleNamespace(
            n_actions=5, embed_dim=1, discount_factor=.99, temperature=200.,
            featuriser=IdentityFeaturiser(),
            forward_network=SimpleNamespace(apply=lambda p,x,z: jnp.broadcast_to(jnp.asarray(scores)[None,None,:],(len(x),1,5))),
        )

    def test_forced_action_is_after_immediate_state(self):
        env=ContinuousWorld(size=11,step_size=1,movement_noise=0)
        learner=self.learner([0.]*5)
        hist,_=make_mc(learner,env,4,2,11)(None,jnp.array([5.5,5.5]),jnp.array(1),jnp.ones(1),jax.random.PRNGKey(4))
        expected=np.zeros(121)
        expected[5*11+5]=1
        expected[5*11+6]=.99
        np.testing.assert_allclose(hist,np.broadcast_to(expected,hist.shape),rtol=1e-6)

    def test_target_softmax_temperature_mask_and_sampling(self):
        env=ContinuousWorld(size=11,step_size=1,movement_noise=0)
        env.get_available_actions=lambda s:[0,1,4]
        learner=self.learner([0.,200.,10000.,10000.,0.])
        hist,_=make_mc(learner,env,6000,3,11)(None,jnp.array([5.5,5.5]),jnp.array(4),jnp.ones(1),jax.random.PRNGKey(9))
        mean=np.asarray(hist).mean(0)
        mean[5*11+5]-=1+.99
        observed=np.array([mean[5*11+4],mean[5*11+6],mean[5*11+5]])/.99**2
        expected=np.exp([0.,1.,0.]);expected/=expected.sum()
        np.testing.assert_allclose(observed,expected,atol=.025,rtol=0)
        self.assertEqual(float(mean[6*11+5]),0.)
        self.assertEqual(float(mean[4*11+5]),0.)

    def test_reference_average_is_signed_and_not_normalised(self):
        learner=SimpleNamespace(embed_F=lambda s,z,action:np.array([[2.]]))
        reference=np.array([[.1,.1],[.9,.9]],np.float32)
        hist,returns,se,scores=predict(learner,reference,np.array([.1,.1]),0,np.ones(1),2,1.,backward=np.array([[-3.],[1.]]))
        np.testing.assert_array_equal(scores,[-6.,2.])
        np.testing.assert_array_equal(hist,[-3.,0.,0.,1.])
        self.assertEqual(hist.sum(),-2.)


if __name__=='__main__':
    unittest.main()
