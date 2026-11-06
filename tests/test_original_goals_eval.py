import unittest
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from fb_navigation.agents.forward_backward_jax import IdentityFeaturiser
from fb_navigation.environments.continuous_four_rooms_jax import ContinuousWorld,WallBlock
from fb_navigation.evaluation.evaluate_original_goals_jax import make_rollout


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


if __name__=='__main__':unittest.main()
