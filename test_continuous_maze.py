"""Parity against unmodified released sources, plus schedule and checkpoint checks."""
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import torch

from continuous_maze import ACTIONS, WALLS, ContinuousMaze, goal_reached, intersects
from forward_backward_jax import FBLearner, _cauchy_ball
from run_continuous_maze import DEFAULT_CONFIG, make_learner, evaluate

REFERENCE = Path(__file__).parent / 'experiments/continuous_maze/reference'
sys.path.insert(0, str(REFERENCE))
from continuous_world_modules.env import ContinuousWorld
from continuous_world_modules.geometry import Point, intersect
from continuous_world_modules.featurizer import RadialBasisFunction2D as TorchRBF
from continuous_world_modules.fb_agent import FBAgent
from discrete_action_robots_modules.models import ForwardMap, BackwardMap


class EnvironmentParity(unittest.TestCase):
    def test_seeded_trajectories_and_resets(self):
        reference = ContinuousWorld(1, wall_pairs=[(Point(*a), Point(*b)) for a, b in WALLS],
                                    movement_noise=.01, threshold_distance=.05)
        env = ContinuousMaze(17)
        np.random.seed(17)
        collisions = 0
        for episode in range(20):
            np.testing.assert_array_equal(env.reset(), reference.reset())
            np.testing.assert_array_equal(env.goal, reference.goal)
            for t in range(50):
                a = (t+episode) % 5
                actual, reward, done, info = env.step(a)
                expected, r, d, i = reference.step(a)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual((reward, done, info['goes_through_wall']), (r, d, i['goes_through_wall']))
                collisions += info['goes_through_wall']
        self.assertGreater(collisions, 0)

    def test_actions_clipping_collisions_and_goal(self):
        env = ContinuousMaze(movement_noise=0)
        reference = ContinuousWorld(1, wall_pairs=[(Point(*a), Point(*b)) for a, b in WALLS],
                                    movement_noise=0, threshold_distance=.05)
        starts = [[.5, .5], [0, 0], [1, 1], [.2, .2], [.8, .8], [.25, .4], [.75, .6]]
        for start in starts:
            for action in range(5):
                env.position, env.goal = start, np.array([.5, .5])
                reference.set_initial_position(Point(*start))
                reference.set_goal(Point(.5, .5))
                np.testing.assert_array_equal(env.step(action)[0], reference.step(action)[0])
        self.assertTrue(goal_reached([.54, .54], [.5, .5]))
        self.assertFalse(goal_reached([.551, .5], [.5, .5]))
        self.assertTrue(goal_reached([.05, .05], [0, 0]))
        for action in range(5):
            env.position = np.array([.5, .5])
            np.testing.assert_allclose(env.step(action)[0], .5+ACTIONS[action], atol=2e-8, rtol=0)

    def test_geometry_parity(self):
        rng = np.random.RandomState(3)
        cases = [rng.uniform(0, 1, (4, 2)) for _ in range(1000)]
        cases += [np.array([a, b, c, d]) for a, b in WALLS for c, d in WALLS]
        for a, b, c, d in cases:
            self.assertEqual(intersects(a, b, c, d), intersect((Point(*a), Point(*b)), (Point(*c), Point(*d))))

    def test_stay_noise(self):
        env = ContinuousMaze(4)
        samples = []
        env.goal = np.array([0, 0])
        for _ in range(5000):
            env.position = np.array([.5, .5])
            samples.append(env.step(4)[0]-.5)
        samples = np.array(samples)
        np.testing.assert_allclose(samples.std(axis=0), [.01, .01], atol=.0004)
        self.assertLess(abs(np.corrcoef(samples.T)[0, 1]), .05)


class FBParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(DEFAULT_CONFIG.read_text())
        cls.config['learner']['buffer_size'] = 500
        cls.learner = make_learner(cls.config, 0)

    def test_shapes_rbf_schedule_and_tasks(self):
        learner = self.learner
        obs = np.array([[0, 0], [.4, .7], [1, 1]], np.float32)
        features = learner.featuriser.transform(jnp.asarray(obs))
        expected = TorchRBF(1, 21, .05).transform(torch.tensor(obs)).numpy()
        np.testing.assert_allclose(features, expected, atol=1e-7)
        self.assertEqual(features.shape, (3, 441))
        self.assertEqual(learner.embed_B(obs).shape, (3, 50))
        self.assertEqual(learner.embed_F(obs, np.ones((3, 50))).shape, (3, 50, 5))
        self.assertEqual(len(learner.state['params']['b']), 4)
        config = json.loads(DEFAULT_CONFIG.read_text())
        self.assertEqual(config['epochs']*config['cycles']*config['updates'], 800000)
        self.assertEqual(config['epochs']*config['cycles']*config['episodes']*config['steps'], 2400000)
        tasks = np.asarray(_cauchy_ball(jax.random.PRNGKey(2), 10000, 50))
        self.assertTrue(np.isfinite(tasks).all())
        self.assertAlmostEqual(np.median(np.linalg.norm(tasks, axis=1))/np.sqrt(50), .5, delta=.025)

    def test_released_loss_and_all_gradients(self):
        torch.set_num_threads(1)
        learner = self.learner
        ref = FBAgent.__new__(FBAgent)
        ref.args = SimpleNamespace(cuda=False, w_sampling='cauchy_ball', batch_size=128,
                                   embed_dim=50, gamma=.99, reg_coef=1., soft_update=True, temp=200.)
        ref.featuriser = TorchRBF(1, 21, .05)
        env_params = dict(obs=441, goal=441, action=5)
        for name, model, key, state_key in [
            ('forward_network', ForwardMap, 'f', 'params'),
            ('backward_network', BackwardMap, 'b', 'params'),
            ('forward_target_network', ForwardMap, 'f', 'target'),
            ('backward_target_network', BackwardMap, 'b', 'target')]:
            network = model(env_params, 50)
            for layer, values in zip([m for m in network.modules() if isinstance(m, torch.nn.Linear)], learner.state[state_key][key]):
                with torch.no_grad():
                    layer.weight.copy_(torch.tensor(np.asarray(values['weight']).T.copy()))
                    layer.bias.copy_(torch.tensor(np.asarray(values['bias']).copy()))
            setattr(ref, name, network)
        rng = np.random.RandomState(9)
        def batch():
            return dict(obs=rng.uniform(0, 1, (128, 2)).astype('float32'),
                        obs_next=rng.uniform(0, 1, (128, 2)).astype('float32'),
                        action=rng.randint(5, size=128), g=np.zeros((128, 2), np.float32))
        transitions, other = batch(), batch()
        ref.buffer = SimpleNamespace(sample=iter([transitions, other]).__next__)
        # Adapt only the sampler signature; execute the actual released update method.
        batches = iter([transitions, other])
        ref.buffer.sample = lambda n: next(batches)
        w = _cauchy_ball(jax.random.PRNGKey(11), 128, 50)
        ref.sample_cauchy_ball = lambda n: torch.tensor(np.asarray(w).copy())
        params = list(ref.forward_network.parameters())+list(ref.backward_network.parameters())
        ref.fb_optim = SimpleNamespace(zero_grad=lambda: None, step=lambda: None)
        expected_loss, entropy = ref._update_network()
        trans = {k: jnp.asarray(v) for k, v in transitions.items() if k != 'g'}
        trans['next_actions'] = jnp.ones((128, 5), bool)
        loss, grads = jax.value_and_grad(learner._loss)(learner.state['params'], learner.state['target'], trans,
                    dict(obs=jnp.asarray(other['obs'])), w, jnp.zeros(128, bool))
        np.testing.assert_allclose(loss, expected_loss, atol=1e-6, rtol=1e-5)
        for key, network in [('f', ref.forward_network), ('b', ref.backward_network)]:
            for actual, layer in zip(grads[key], [m for m in network.modules() if isinstance(m, torch.nn.Linear)]):
                for field, expected in [('weight', layer.weight.grad.numpy().T), ('bias', layer.bias.grad.numpy())]:
                    self.assertTrue(np.isfinite(actual[field]).all())
                    np.testing.assert_allclose(actual[field], expected, atol=2e-7, rtol=2e-4)
        components = learner._loss(learner.state['params'], learner.state['target'], trans,
                    dict(obs=jnp.asarray(other['obs'])), w, jnp.zeros(128, bool), return_components=True)
        np.testing.assert_allclose(components['entropy'], entropy, atol=1e-6)

    def test_target_update_and_checkpoint(self):
        learner = make_learner(self.config, 1)
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
