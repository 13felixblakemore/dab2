"""Analytical checks of the vectorised FB/MC bottleneck comparison."""
import unittest
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from continuous_four_rooms_jax import ContinuousWorld, ContinuousFourRooms
from forward_backward_jax import IdentityFeaturiser
from estimate_fb_bottlenecks_jax import (
    grid_regions,sample_pairs,eligibility,make_fb_estimator,make_mc_estimator,aggregate)


class BottleneckEstimatorTests(unittest.TestCase):
    def test_geometry_and_matched_exclusions(self):
        env=ContinuousFourRooms(include_stay_action=False)
        centers,lo,hi,keep=grid_regions(env,20)
        self.assertEqual(len(centers),400)
        for w in env.wall_blocks:
            overlap=(lo[:,0]<w.xmax)&(hi[:,0]>w.xmin)&(lo[:,1]<w.ymax)&(hi[:,1]>w.ymin)
            self.assertFalse(np.any(overlap & keep))
        u,g=sample_pairs(env,100,.1,2)
        u2,g2=sample_pairs(env,100,.1,2)
        np.testing.assert_array_equal(u,u2);np.testing.assert_array_equal(g,g2)
        self.assertTrue(np.all(np.sum((u-g)**2,axis=1)>.1**2))
        for p in np.concatenate((u,g)):
            self.assertFalse(bool(env._check_on_wall(SimpleNamespace(x=p[0],y=p[1]))))
        # Cell center is outside goal, but its corner overlaps: exclude it.
        result=eligibility(np.array([[0.,0.]]),np.array([[.05,.05],[.2,.2]]),np.array([[.15,.15],[.3,.3]]),.1)
        np.testing.assert_array_equal(result,[[False,True]])

    def test_invalid_denominators_not_clamped_or_silently_averaged(self):
        numerator=np.array([[1.,2.,-1.],[3.,4.,2.]])
        denom=np.array([[1.,-1.,1.],[0.,2.,1.]])
        eligible=np.array([[True,True,True],[True,False,True]])
        mc=np.array([[.2,.3,.4],[.5,.6,.7]])
        result=aggregate(numerator,denom,eligible,mc)
        np.testing.assert_allclose(result['validity'],[.5,0.,1.])
        self.assertTrue(np.isnan(result['fb_strict'][:2]).all())
        self.assertTrue(np.isnan(result['ratios'][1,0]))
        self.assertEqual(result['ratios'][0,2],-1.)
        self.assertEqual(result['ratios'][1,2],2.)
        np.testing.assert_allclose(result['mc_matched'][[0,2]],[.2,.55])

    def test_vectorised_fb_matches_scalar_policy_average(self):
        def forward(p,x,w):
            base=x[:,0]+2*x[:,1]+w[:,0]
            return jnp.stack((base,2*base),axis=-1)[:,None,:]
        learner=SimpleNamespace(n_actions=2,embed_dim=1,temperature=2.,
            featuriser=IdentityFeaturiser(),forward_network=SimpleNamespace(apply=forward))
        c=jnp.array([[.2,.3],[.5,.6]])
        b=jnp.array([[2.],[3.]])
        starts=jnp.array([[.1,.2],[.3,.4]]);tasks=jnp.array([[.5],[1.]])
        num,den=make_fb_estimator(learner,c,b)(None,None,starts,tasks)
        for i in range(2):
            def scalar(x):
                base=float(x[0]+2*x[1]+tasks[i,0]);f=np.array([base,2*base])
                pi=np.exp(f*float(tasks[i,0])/2);pi/=pi.sum()
                return (f*pi).sum()
            np.testing.assert_allclose(num[i],np.array([2,3])*scalar(starts[i]),rtol=1e-6)
            np.testing.assert_allclose(den[i],[2*scalar(c[0]),3*scalar(c[1])],rtol=1e-6)

    def test_mc_first_entry_goal_stop_and_batch_invariance(self):
        learner=SimpleNamespace(n_actions=4,embed_dim=1,temperature=200.,goal_threshold=.06,
            featuriser=IdentityFeaturiser(),forward_network=SimpleNamespace(apply=lambda p,x,z:
                jnp.broadcast_to(jnp.array([0.,1e6,0.,0.])[None,None,:],(len(x),1,4))))
        env=ContinuousWorld(size=1.,step_size=.1,movement_noise=0.)
        fn=make_mc_estimator(learner,env,20,8)
        starts=jnp.array([[.2,.2],[.2,.2]]);goals=jnp.array([[.55,.2],[.55,.2]])
        keys=jax.random.split(jax.random.PRNGKey(1),2);tasks=jnp.ones((2,1))
        hits,goal_hits=fn(None,starts,goals,tasks,keys)
        np.testing.assert_array_equal(goal_hits,[3,3])
        self.assertTrue(np.all(np.asarray(hits)[:,4*20+10]==9))  # entry cell excluded
        self.assertTrue(np.all(np.asarray(hits)[:,4*20+4]==0))
        h,g=fn(None,starts[:1],goals[:1],tasks[:1],keys[:1])
        np.testing.assert_array_equal(h,hits[:1]);np.testing.assert_array_equal(g,goal_hits[:1])


if __name__=='__main__':unittest.main()
