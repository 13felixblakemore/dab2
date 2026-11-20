"""Goal-terminated training and checkpoint compatibility."""
from pathlib import Path
import pickle
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from fb_navigation.agents.forward_backward_jax import (
    FBLearner,
    RadialBasisFunction2D,
)
from fb_navigation.paths import PROJECT_ROOT
from fb_navigation.diagnostics.estimate_fb_bottlenecks_jax import make_fb_estimator, make_mc_estimator
from fb_navigation.environments.continuous_four_rooms_jax import ContinuousFourRooms, ContinuousWorld, Point
from fb_navigation.training.run_continuous_maze import evaluate
from fb_navigation.training.run_fb_jax import run_fb


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


class TrainingEntryPointTests(unittest.TestCase):
    def test_importing_torch_runner_does_not_start_training(self):
        import runpy
        from unittest.mock import patch

        with patch('fb_navigation.agents.forward_backward.FBLearner') as create_learner:
            namespace = runpy.run_path(
                str(PROJECT_ROOT / 'fb_navigation/training/run_fb.py'),
                run_name='torch_runner_import_check',
            )
        create_learner.assert_not_called()
        self.assertTrue(callable(namespace['main']))


class ContinuousActionTests(unittest.TestCase):
    def test_training_checkpoint_and_bottleneck_estimators(self):
        model = learner(action_space='continuous', continuous_action_samples=4)
        env = ContinuousFourRooms(action_space='continuous', include_stay_action=False)
        with tempfile.TemporaryDirectory() as output:
            model, losses = run_fb(epochs=1, cycles=1, episodes=1, steps=8, updates=2,
                                   device='cpu', output_dir=output, env=env, learner=model,
                                   progress=False)
            self.assertTrue(np.isfinite(losses).all())
            actions = np.asarray(model.buffer.action[:model.buffer.size])
            self.assertEqual(actions.shape, (8, 2))
            self.assertTrue(np.all(np.linalg.norm(actions, axis=-1) <= env.step_size))
            self.assertTrue(np.any(np.all(actions != 0, axis=-1)))
            model = FBLearner.from_checkpoint(Path(output) / 'fb_fourrooms_jax.pkl')

        self.assertTrue(model.continuous)
        env = ContinuousWorld.from_movement_config(model.environment_config)
        starts = jnp.asarray([[.2, .2], [.3, .3]])
        goals = jnp.asarray([[.7, .2], [.8, .25]])
        tasks = model.embed_B(goals)
        key = jax.random.PRNGKey(5)
        moves = model.act(starts, tasks, key=key)
        self.assertEqual(moves.shape, (2, 2))
        np.testing.assert_array_equal(moves, model.act(starts, tasks, key=key))
        self.assertEqual(model.embed_F(starts, tasks, moves).shape, tasks.shape)
        features = model.featuriser.transform(starts)
        params = model.state['params']['f']
        candidate_embeddings = model.forward_candidates(params, features, tasks, moves)
        np.testing.assert_allclose(candidate_embeddings[jnp.arange(2), :, jnp.arange(2)],
                                   model.embed_F(starts, tasks, moves), atol=1e-6)
        env.reset(Point(*starts[0]), Point(*goals[0]))
        with self.assertRaises(ValueError):
            env.step((env.step_size, env.step_size))
        self.assertEqual(env.step(moves[0])[0].shape, (2,))

        candidates = jnp.asarray([[.12, .12], [.32, .32], [.75, .75]])
        estimate_fb = make_fb_estimator(model, candidates, model.embed_B(candidates))
        numerator, denominator = estimate_fb(
            params, model.state['target']['f'], features, tasks)
        self.assertEqual(numerator.shape, (2, 3))
        self.assertTrue(np.isfinite(denominator).all())
        estimate_mc = make_mc_estimator(model, env, 10, 3)
        first, hit = estimate_mc(model.state['target']['f'], starts, goals, tasks,
                                 jax.random.split(key, 2))
        self.assertEqual(first.shape, (2, 100))
        np.testing.assert_array_equal(first[jnp.arange(2), [22, 33]], [0, 0])
        self.assertEqual(hit.shape, (2,))


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





class ContinuingLearnerTests(unittest.TestCase):
    def test_target_update_and_checkpoint(self):
        learner = FBLearner(obs_dim=2, n_actions=5, embed_dim=3, hidden_dim=8, num_layers=0,
                            batch_size=4, buffer_size=256, seed=1, polyak_coef=.95,
                            device='cpu', buffer_device='cpu')
        rng = np.random.RandomState(1)
        learner.buffer.add_batch(dict(obs=rng.rand(200, 2), action=rng.randint(5, size=200),
                                      obs_next=rng.rand(200, 2), next_actions=np.ones((200, 5), bool)))
        old = jax.tree_util.tree_map(np.asarray, learner.state['target'])
        self.assertTrue(np.isfinite(learner.update_networks(2)).all())
        for a, b in zip(jax.tree_util.tree_leaves(old), jax.tree_util.tree_leaves(learner.state['target'])):
            np.testing.assert_array_equal(a, b)
        learner.soft_update_target_networks()
        expected = jax.tree_util.tree_map(lambda t, p: .95*t+.05*p, old, learner.state['params'])
        for a, b in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(learner.state['target'])):
            np.testing.assert_allclose(a, b, atol=1e-8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pkl'
            learner.save(path)
            loaded = FBLearner.from_checkpoint(path)
            np.testing.assert_array_equal(learner.embed_B([[.5, .5]]), loaded.embed_B([[.5, .5]]))
        result, paths = evaluate(learner, 3, 7)
        self.assertEqual(result['goals'], 3)
        self.assertTrue(all(2 <= len(p['path']) <= 31 for p in paths))
        _, random_paths = evaluate(learner, 3, 7, epsilon=1)
        for actual, expected in zip(paths, random_paths):
            self.assertEqual(actual['goal'], expected['goal'])
            self.assertEqual(actual['path'][0], expected['path'][0])


if __name__ == '__main__':
    unittest.main()
