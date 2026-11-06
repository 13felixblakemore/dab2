import unittest
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
from track_doorway_advantages import advantages,discounted_hits,make_probes,mc_function,comparison
from adjustable_four_rooms import AdjustableFourRooms


class AdvantageTests(unittest.TestCase):
    def test_advantages_exclude_current_action_and_handle_ties(self):
        q=np.array([[1.,4.,3.,2.],[2.,2.,0.,1.]])
        np.testing.assert_array_equal(advantages(q),[[-3,1,-1,-2],[0,0,-2,-1]])
        np.testing.assert_array_equal(discounted_hits(np.array([1,2,31]),30,.5),[1,.5,0])

    def test_probes_fixed_free_and_cross_room(self):
        env=AdjustableFourRooms(doorway_scale=2,include_stay_action=False)
        first,second=make_probes(env,5),make_probes(env,5)
        self.assertEqual(first['states'].shape,(160,2))
        for key in first:np.testing.assert_array_equal(first[key],second[key])
        self.assertTrue((np.max(abs(first['states']-first['goals']),axis=1)>.1).all())
        self.assertEqual(set(first['goal_kind']),{'adjacent','diagonal'})

    def test_first_action_is_forced_and_first_hit_discount(self):
        env=AdjustableFourRooms(doorway_scale=2,include_stay_action=False);env._noise=0
        learner=SimpleNamespace(featuriser=SimpleNamespace(transform=lambda x:x),
            backward_network=SimpleNamespace(apply=lambda p,x:jnp.ones((len(x),1))),
            forward_network=SimpleNamespace(apply=lambda p,x,z:jnp.tile(jnp.array([[[1.,2.,0.,0.]]]),(len(x),1,1))))
        fn=mc_function(learner,env,1)
        hits,_=fn(dict(f=0,b=0),jnp.array([[.2,.2],[.2,.2]]),jnp.array([[.35,.2],[.35,.2]]),
                  jnp.array([0,1]),jax.random.split(jax.random.PRNGKey(0),2))
        np.testing.assert_array_equal(hits,[2,1])

    def test_switch_effect_fixes_current_continuation(self):
        old=dict(q=np.array([[2.,1.,0.,0.]]),hits=np.array([[[1,1],[2,2],[4,4],[4,4]]]),blocked=np.array([[False,False,True,True]]))
        new=dict(q=np.array([[1.,2.,0.,0.]]),hits=np.array([[[1,1],[31,31],[31,31],[31,31]]]),blocked=old['blocked'])
        result=comparison(old,new,30)
        self.assertEqual(result['changed_probes'],1)
        self.assertEqual(result['switch_gain_current_continuation'],-1)
        self.assertEqual(result['switch_return_worsened'],1)


if __name__=='__main__':unittest.main()
