"""Checks for PyTorch checkpoint portability and the shared goal benchmark."""
from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch

from continuous_four_rooms_jax import ContinuousWorld, WallBlock
from forward_backward import FBLearner, RadialBasisFunction2D
from evaluate_torch_goals import load_learner, movement_equivalence, make_environment_step


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

    def test_load_repairs_pickled_feature_device_only(self):
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
            learner.save(path)
            loaded = load_learner(path, 'cpu')
        self.assertEqual(loaded.featuriser.device, 'cpu')
        torch.testing.assert_close(loaded.embed_B(points), expected)


if __name__ == '__main__':
    unittest.main()
