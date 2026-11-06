"""Regression tests for the optional pre-goal-entry FB Bellman target."""
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from fb_navigation.agents.forward_backward_jax import FBLearner, RadialBasisFunction2D, IdentityFeaturiser
from fb_navigation.environments.continuous_four_rooms_jax import ContinuousWorld
from fb_navigation.training.run_fb_jax import run_fb
from fb_navigation.training.smoke_goal_terminated_fb_jax import make_stopped_mc


def learner(**kwargs):
    config=dict(obs_dim=2,n_actions=4,embed_dim=3,hidden_dim=8,num_layers=0,
                batch_size=4,buffer_size=24,device='cpu',buffer_device='cpu',
                z_sampling='replay',goal_terminated=True,goal_threshold=.1,
                featuriser=RadialBasisFunction2D((0,1),(0,1),3,.2))
    config.update(kwargs)
    return FBLearner(**config)


def batch():
    transitions=dict(obs=jnp.array([[.1,.1],[.2,.3],[.7,.8],[.5,.2]]),
        obs_next=jnp.array([[.3,.3],[.5,.5],[.8,.8],[.2,.2]]),
        action=jnp.array([0,1,2,3]),next_actions=jnp.array([[1,1,0,0],[1,0,1,0],[1,1,1,1],[0,1,0,1]],bool))
    other=dict(obs=jnp.array([[.2,.1],[.4,.3],[.6,.8],[.5,.9]]))
    goals=jnp.array([[.31,.3],[.8,.8],[.8,.8],[.28,.28]])
    zero=jnp.array([False,False,True,False])
    return transitions,other,goals,zero


def oracle(l,t,other,w,zero,goals,terminated):
    """Independent NumPy loss calculation including immediate/regularizer terms."""
    p=l.state['params'];target=l.state['target']
    obs=l.featuriser.transform(t['obs']);nxt=l.featuriser.transform(t['obs_next']);oth=l.featuriser.transform(other['obs'])
    fnext=np.asarray(l.forward_network.apply(target['f'],nxt,w))
    logits=np.einsum('sda,sd->sa',fnext,np.asarray(w))/l.temperature
    logits[np.asarray(zero)]=0
    logits=np.where(t['next_actions'],logits,-np.inf)
    pi=np.exp(logits-logits.max(axis=1,keepdims=True));pi/=pi.sum(axis=1,keepdims=True)
    target_matrix=np.einsum('sda,sa->sd',fnext,pi) @ np.asarray(l.backward_network.apply(target['b'],oth)).T
    if terminated:
        reached=(np.sum((np.asarray(t['obs_next'])-np.asarray(goals))**2,axis=-1)<=l.goal_threshold**2)&~np.asarray(zero)
        target_matrix[reached]=0
    fall=np.asarray(l.forward_network.apply(p['f'],obs,w))
    f=fall[np.arange(4),:,np.asarray(t['action'])]
    b=np.asarray(l.backward_network.apply(p['b'],obs));bo=np.asarray(l.backward_network.apply(p['b'],oth))
    fb=.5*np.mean((f@bo.T-l.discount_factor*target_matrix)**2)-np.sum(f*b,axis=1).mean()
    reg=np.mean((b@bo.T)**2)-np.sum(b*b,axis=1).mean()
    return fb+l.reg_coef*reg


class GoalTerminationTests(unittest.TestCase):
    def test_linf_matches_environment_and_preserves_default_l2(self):
        t,o,g,zero=batch();matrix=jnp.ones((4,4))
        square=learner(goal_norm='linf')
        actual=jax.jit(square._mask_goal_bootstrap)(matrix,t['obs_next'],g,zero)
        expected=np.ones((4,4));expected[[0,3]]=0
        np.testing.assert_array_equal(actual,expected)
        circle=learner()
        self.assertEqual(circle.goal_norm,'l2')
        self.assertTrue(np.asarray(circle._mask_goal_bootstrap(matrix,t['obs_next'],g,zero))[3].all())
        self.assertEqual(square.config['goal_norm'],'linf')
        with self.assertRaises(ValueError):learner(goal_norm='invalid')

    def test_rows_raw_coordinates_and_zero_tasks(self):
        l=learner();t,o,g,zero=batch();matrix=jnp.arange(16,dtype=jnp.float32).reshape(4,4)+1
        actual=jax.jit(l._mask_goal_bootstrap)(matrix,t['obs_next'],g,zero)
        expected=np.asarray(matrix).copy();expected[0]=0
        np.testing.assert_array_equal(actual,expected)
        # Row 3 is inside the environment's square tolerance but outside L2.
        self.assertGreater(np.linalg.norm(np.asarray(t['obs_next'][3]-g[3])),l.goal_threshold)

    def test_inclusive_boundary(self):
        l=learner(goal_threshold=.125)
        actual=l._mask_goal_bootstrap(jnp.ones((1,4)),jnp.array([[.125,0.]]),jnp.zeros((1,2)),jnp.array([False]))
        np.testing.assert_array_equal(actual,0)

    def test_cauchy_and_disabled_modes_are_continuing(self):
        t,o,g,zero=batch();matrix=jnp.ones((4,4))
        for l in (learner(z_sampling='cauchy'),learner(goal_terminated=False)):
            np.testing.assert_array_equal(l._mask_goal_bootstrap(matrix,t['obs_next'],g,zero),matrix)
            w=jnp.ones((4,3))
            actual=l._loss(l.state['params'],l.state['target'],t,o,w,zero,g)
            np.testing.assert_allclose(actual,oracle(l,t,o,w,zero,g,False),rtol=1e-5,atol=1e-6)

    def test_loss_integration_leaves_immediate_and_regularizer_unchanged(self):
        l=learner();t,o,g,zero=batch();w=l._training_tasks(l.state['params'],jax.random.PRNGKey(1),g)
        w=jnp.where(zero[:,None],0.,w)
        actual=l._loss(l.state['params'],l.state['target'],t,o,w,zero,g)
        np.testing.assert_allclose(actual,oracle(l,t,o,w,zero,g,True),rtol=1e-5,atol=1e-6)

    def test_missing_goals_raise_when_termination_enabled(self):
        l=learner()
        with self.assertRaises(ValueError):l._mask_goal_bootstrap(jnp.ones((4,4)),jnp.ones((4,2)),None,jnp.zeros(4,bool))

    def test_target_gradients_zero_goal_embedding_detached_online_finite(self):
        l=learner();t,o,g,zero=batch();w=l._training_tasks(l.state['params'],jax.random.PRNGKey(1),g)
        grads=jax.grad(l._loss,argnums=1)(l.state['params'],l.state['target'],t,o,w,zero,g)
        self.assertTrue(all(np.all(np.asarray(x)==0) for x in jax.tree_util.tree_leaves(grads)))
        grads=jax.grad(lambda p:l._training_tasks(p,jax.random.PRNGKey(1),g).sum())(l.state['params'])
        self.assertTrue(all(np.all(np.asarray(x)==0) for x in jax.tree_util.tree_leaves(grads)))
        value,grads=jax.value_and_grad(l._loss)(l.state['params'],l.state['target'],t,o,w,zero,g)
        self.assertTrue(np.isfinite(value))
        self.assertTrue(all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(grads)))
        self.assertTrue(any(np.any(np.asarray(x)!=0) for x in jax.tree_util.tree_leaves(grads)))

    def test_single_update_multi_update_and_compiled_runner(self):
        l=learner(zero_z_p=.25)
        for i in range(12):l.update([i/15,.2],i%4,[i/15,.3],False,[0,1,2,3])
        self.assertTrue(np.isfinite(l.update_network()))
        l.buffer.storage=jax.device_put(l.buffer.storage);l.buffer._host=False
        self.assertTrue(np.isfinite(l.update_networks(3)).all())
        with tempfile.TemporaryDirectory() as tmp:
            l,loss=run_fb(epochs=1,cycles=1,episodes=1,steps=4,updates=2,learner=l,output_dir=tmp,progress=False)
            self.assertTrue(np.isfinite(loss).all())
            restored=FBLearner.from_checkpoint(Path(tmp)/'fb_fourrooms_jax.pkl')
            self.assertTrue(restored.goal_terminated);self.assertEqual(restored.goal_threshold,.1)
            self.assertEqual(restored.discount_factor,.99)
            for a,b in zip(jax.tree_util.tree_leaves(l.state),jax.tree_util.tree_leaves(restored.state)):
                np.testing.assert_array_equal(a,b)

    def test_old_checkpoints_remain_continuing(self):
        l=learner(goal_terminated=False)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'old.pkl';l.save(path)
            with path.open('rb') as f:c=pickle.load(f)
            c['config'].pop('goal_terminated');c['config'].pop('goal_threshold')
            with path.open('wb') as f:pickle.dump(c,f)
            restored=FBLearner.from_checkpoint(path)
            self.assertFalse(restored.goal_terminated);self.assertEqual(restored.goal_threshold,.1)

    def test_option_validation(self):
        for threshold in (0.,-1.,np.inf,np.nan):
            with self.assertRaises(ValueError):learner(goal_threshold=threshold)

    def test_stopped_mc_excludes_entry_and_rejects_start_in_goal(self):
        l=SimpleNamespace(n_actions=4,embed_dim=1,temperature=200.,
            discount_factor=.99,goal_threshold=.06,featuriser=IdentityFeaturiser(),
            forward_network=SimpleNamespace(apply=lambda p,x,z:
                jnp.broadcast_to(jnp.array([0.,1e6,0.,0.])[None,None],(len(x),1,4))))
        env=ContinuousWorld(size=1.,step_size=.1,movement_noise=0.)
        mc=make_stopped_mc(l,env,rollouts=4,horizon=8,bins=20)
        start=jnp.array([.2,.2]);goal=jnp.array([.55,.2])
        hist,active=mc(None,start,1,jnp.ones(1),goal,jax.random.PRNGKey(1))
        # Visits .2, .3, .4; the next position .5 enters the goal and is excluded.
        np.testing.assert_allclose(np.asarray(hist).sum(axis=1),1+.99+.99**2,rtol=1e-6)
        np.testing.assert_array_equal(active,False)
        self.assertTrue(np.all(np.asarray(hist)[:,4*20+10]==0))
        with self.assertRaises(ValueError):mc(None,goal,1,jnp.ones(1),goal,jax.random.PRNGKey(1))


if __name__=='__main__':unittest.main()
