"""Native goal rollouts and PyTorch checkpoint portability."""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import torch

from fb_navigation.agents.forward_backward import FBLearner, RadialBasisFunction2D
from fb_navigation.agents.forward_backward_jax import IdentityFeaturiser
from fb_navigation.environments.continuous_four_rooms_jax import ContinuousWorld, WallBlock
from fb_navigation.evaluation.evaluate_original_goals_jax import make_rollout
from fb_navigation.evaluation.evaluate_torch_goals import (
    load_learner,
    make_environment_step,
    movement_equivalence,
)


class NativeGoalEvaluationTests(unittest.TestCase):
    def test_first_hit_and_native_collision_handling(self):
        learner=SimpleNamespace(n_actions=4,embed_dim=1,temperature=200.,featuriser=IdentityFeaturiser(),
            forward_network=SimpleNamespace(apply=lambda p,x,z:jnp.broadcast_to(jnp.array([0.,1e6,0.,0.])[None,None],(len(x),1,4))))
        env=ContinuousWorld(size=1.,step_size=.1,movement_noise=0.,threshold_distance=.06)
        starts=jnp.array([[.2,.2]]);goals=jnp.array([[.55,.2]]);keys=jax.random.split(jax.random.PRNGKey(1),1)
        hit,blocked=make_rollout(learner,env,10,'greedy_online')(None,starts,goals,jnp.ones((1,1)),keys)
        np.testing.assert_array_equal(hit,3);np.testing.assert_array_equal(blocked,0)
        env=ContinuousWorld(size=1.,step_size=.1,movement_noise=0.,threshold_distance=.06,
                            wall_blocks=[WallBlock(.3,0.,.4,1.)])
        hit,blocked=make_rollout(learner,env,10,'greedy_online')(None,starts,jnp.array([[.8,.2]]),jnp.ones((1,1)),keys)
        np.testing.assert_array_equal(hit,11)
        self.assertGreaterEqual(int(blocked[0]),8)


class TorchGoalTests(unittest.TestCase):
    def test_original_and_shared_environment_movement(self):
        result = movement_equivalence(count=128)
        self.assertTrue(result['collision_flags_identical'])
        self.assertLessEqual(result['max_coordinate_difference'], 1e-6)

    def test_diagnostic_mask_changes_action_without_changing_physics(self):
        env = ContinuousWorld(size=1., step_size=.1, movement_noise=0., include_stay_action=False,
                              wall_blocks=[WallBlock(.4, 0., .5, .6)])
        args = (jnp.asarray([[.35, .3]]), jnp.asarray([51]), jax.random.split(jax.random.PRNGKey(8), 1),
                jnp.asarray([[1., 10., 5., 0.]]), 0)
        baseline = make_environment_step(env, 50, 'greedy_online', 200)(*args)
        masked = make_environment_step(env, 50, 'wall_aware_online', 200)(*args)
        self.assertEqual(int(baseline[2][0]), 1)
        self.assertTrue(bool(baseline[3][0]))
        np.testing.assert_allclose(baseline[0], [[.35, .3]], atol=1e-7)
        self.assertEqual(int(masked[2][0]), 2)
        self.assertFalse(bool(masked[3][0]))
        np.testing.assert_allclose(masked[0], [[.35, .4]], atol=1e-7)

    def test_load_repairs_legacy_feature_module_and_device(self):
        torch.manual_seed(3)
        feature = RadialBasisFunction2D((0, 1), (0, 1), 3, .2, device='cpu')
        learner = FBLearner(2, 4, 3, hidden_dim=8, num_layers=0, buffer_size=16,
                            device='cpu', buffer_device='cpu', featuriser=feature)
        points = np.asarray([[.2, .4], [.8, .7]], np.float32)
        expected = learner.embed_B(points)
        # Emulate the stale device string in a checkpoint mapped to CPU.
        feature.device = 'cuda:99'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.pt'
            legacy = ModuleType('forward_backward')
            legacy.RadialBasisFunction2D = RadialBasisFunction2D
            with patch.dict(sys.modules, {'forward_backward': legacy}), patch.object(
                RadialBasisFunction2D, '__module__', 'forward_backward'
            ):
                learner.save(path)
            loaded = load_learner(path, 'cpu')
        self.assertEqual(loaded.featuriser.device, 'cpu')
        torch.testing.assert_close(loaded.embed_B(points), expected)


if __name__ == '__main__':
    unittest.main()
