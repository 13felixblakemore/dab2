"""Wall masks, doorway advantages, and navigation failure stages."""
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from fb_navigation.diagnostics.compare_navigation_stages import (
    extract_probes,
    mc_rollout,
    summary_for,
)
from fb_navigation.diagnostics.diagnose_fb_walls import (
    collision_runs,
    make_trace,
    nominal_action_mask,
)
from fb_navigation.diagnostics.investigate_doorway_decline import (
    failure_stages,
    fixed_probes,
    trace_function,
)
from fb_navigation.diagnostics.track_doorway_advantages import (
    advantages,
    comparison,
    discounted_hits,
    make_probes,
    mc_function,
)
from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms
from fb_navigation.environments.continuous_four_rooms_jax import (
    ContinuousFourRooms,
    ContinuousWorld,
    Point,
    WallBlock,
    _move,
)


class WallDiagnosticTests(unittest.TestCase):
    def test_mask_matches_environment_noiseless_collisions(self):
        env = ContinuousFourRooms(include_stay_action=False)
        rng = np.random.default_rng(12)
        positions = jnp.asarray(rng.uniform(0, 1, (150, 2)), dtype=jnp.float32)
        actions = jnp.asarray([(a.x, a.y) for a in env.actions])
        walls = tuple(env.wall_blocks)
        actual = jax.jit(lambda p: nominal_action_mask(p, actions, env.size, walls))(positions)
        def reference(p):
            return jax.vmap(lambda a: ~_move(Point(*p), Point(*a), jax.random.PRNGKey(0),
                                            0., env.size, walls)[2])(actions)
        expected = jax.jit(jax.vmap(reference))(positions)
        np.testing.assert_array_equal(actual, expected)

    def test_mask_changes_wall_action_without_changing_dynamics(self):
        # RIGHT is maximal; UP is the best permitted action at this wall.
        learner = SimpleNamespace(n_actions=4,
            featuriser=SimpleNamespace(transform=lambda x: x),
            forward_network=SimpleNamespace(apply=lambda params, obs, tasks:
                jnp.broadcast_to(jnp.asarray([.1, 1., .5, .2]), (len(obs), 1, 4))))
        env = ContinuousWorld(size=1., step_size=.1, movement_noise=0.,
                              threshold_distance=.05, include_stay_action=False,
                              wall_blocks=[WallBlock(.4, 0., .5, .6)])
        args = (None, jnp.asarray([[.35, .3]]), jnp.asarray([[.8, .3]]),
                jnp.ones((1, 1)), jax.random.split(jax.random.PRNGKey(8), 1))
        hits, original = make_trace(learner, env, 10, False)(*args)
        _, masked = make_trace(learner, env, 10, True)(*args)
        self.assertEqual(int(hits[0]), 11)
        self.assertTrue(np.all(original['action'] == 1))
        self.assertTrue(np.all(original['collision']))
        np.testing.assert_array_equal(original['position'], original['next_position'])
        self.assertEqual(int(masked['action'][0, 0]), 2)
        self.assertFalse(bool(masked['collision'][0, 0]))
        self.assertEqual(int(collision_runs(jax.tree_util.tree_map(np.asarray, original)).max()), 10)

    def test_collision_run_resets_when_action_changes(self):
        record = dict(action=np.asarray([[1, 1, 2, 2, 2, 2]]),
                      collision=np.asarray([[True, True, True, False, True, True]]))
        np.testing.assert_array_equal(collision_runs(record), [[1, 2, 1, 0, 1, 2]])


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


class DeclineTests(unittest.TestCase):

    def test_failure_stages_and_repeated_actions(self):
        # Three adjacent BL -> BR tasks: no approach, approach without crossing,
        # and crossing into the goal room then failing to reach the goal.
        position = np.array([[[.1,.1],[.525,.125],[.525,.125]],
                             [[.1,.1],[.525,.125],[.60,.125]],
                             [[.1,.1],[.525,.125],[.675,.125]]], np.float32)
        following = position.copy(); following[1:,2] = [.675,.125]
        trace = dict(position=position, following=following, hits=np.array([4,4,4]),
                     action=np.ones((3,3),int), blocked=np.array([[0,1,0],[0,1,0],[0,1,1]],bool))
        pairs = dict(start_rooms=np.array([0,0,0]),goal_rooms=np.array([2,2,2]))
        rows = failure_stages(trace,pairs,3,2)
        self.assertEqual(rows[0]['stage'],'never_approached_useful_door')
        self.assertEqual(rows[1]['stage'],'approached_no_progress_crossing')
        self.assertEqual(rows[1]['longest_identical_wall_run'],3)
        self.assertEqual(rows[1]['repeated_wall_actions'],2)
        self.assertEqual(rows[2]['stage'],'entered_goal_room_failed_goal_control')
        trace['hits'][1] = 2
        self.assertEqual(failure_stages(trace,pairs,3,2)[1]['stage'],'success')

    def test_common_probes_are_reproducible_and_use_both_paths(self):
        first = dict(hits=np.array([2]),position=np.zeros((4,1,2),np.float32))
        second = dict(hits=np.array([4]),position=np.ones((4,1,2),np.float32))
        a=fixed_probes(first,second)
        np.testing.assert_array_equal(a,fixed_probes(first,second))
        np.testing.assert_array_equal(a[0,:8],0)
        np.testing.assert_array_equal(a[0,8:],1)

    def test_oracle_mask_is_separate_and_changes_blocked_greedy_choice(self):
        from types import SimpleNamespace
        import jax
        import jax.numpy as jnp
        from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms
        env = AdjustableFourRooms(doorway_scale=2, include_stay_action=False)
        env._noise = 0
        learner = SimpleNamespace(featuriser=SimpleNamespace(transform=lambda x:x),
            backward_network=SimpleNamespace(apply=lambda p,x:jnp.ones((len(x),1))),
            forward_network=SimpleNamespace(apply=lambda p,x,z:jnp.tile(jnp.array([[[1.,2.,0.,0.]]]),(len(x),1,1))))
        args = (dict(f=0,b=0),jnp.array([[.525,.325]]),jnp.array([[.9,.325]]),jax.random.split(jax.random.PRNGKey(0),1))
        normal = trace_function(learner,env,1,epsilon=0)(*args)
        masked = trace_function(learner,env,1,mask_wall_actions=True,epsilon=0)(*args)
        self.assertTrue(bool(normal['blocked'][0,0]))
        self.assertFalse(bool(masked['blocked'][0,0]))
        self.assertEqual(int(normal['action'][0,0]),1)
        self.assertEqual(int(masked['action'][0,0]),0)


class NavigationStageTests(unittest.TestCase):
    def test_stage_states_and_remaining_budget(self):
        path=np.array([[.225,.325],[.225,.475],[.25,.525],[.3,.55],[.35,.6]]+[[.35,.6]]*26,np.float32)
        trace=dict(position=path[:30,None],following=path[1:31,None],hits=np.array([151]))
        pairs=dict(categories=np.array([2]),start_rooms=np.array([0]),goal_rooms=np.array([3]),goals=np.array([[.9,.8]]))
        rows,availability=extract_probes({400:trace,800:trace},pairs,2)
        selected=[r for r in rows if r['source_epoch']==400]
        self.assertEqual([r['source_step'] for r in selected],[0,1,4])
        self.assertEqual([r['remaining_budget'] for r in selected],[30,29,26])
        np.testing.assert_array_equal(selected[1]['state'],path[1])
        self.assertEqual(availability[400]['diagonal_trials'],1)
        self.assertEqual(selected[2]['committed_room'],1)

    def test_monte_carlo_collision_windows_and_streaks(self):
        env=AdjustableFourRooms(doorway_scale=2,include_stay_action=False);env._noise=0
        learner=SimpleNamespace(featuriser=SimpleNamespace(transform=lambda x:x),
            backward_network=SimpleNamespace(apply=lambda p,x:jnp.ones((len(x),1))),
            forward_network=SimpleNamespace(apply=lambda p,x,z:jnp.tile(jnp.array([[[0.,2.,0.,0.]]]),(len(x),1,1))))
        fn=mc_rollout(learner,env,2,epsilon=0)
        record=fn(dict(f=0,b=0),jnp.array([[.525,.325]]),jnp.array([[.9,.325]]),jnp.array([1]),jax.random.split(jax.random.PRNGKey(0),1))
        np.testing.assert_array_equal(record['collisions'],[[2,2,1]])
        np.testing.assert_array_equal(record['repeats'],[[1,1,0]])
        np.testing.assert_array_equal(record['longest'],[[2,2,1]])
        np.testing.assert_array_equal(record['hits'],[3])

    def test_summary_uses_remaining_budget_without_extending_it(self):
        record=dict(q=np.array([[1.,0.,0.,0.]]),advantages=np.array([[1.,-1.,-1.,-1.]]),
            hits=np.array([[3,5]]),collisions=np.array([[[1,1,1],[2,2,1]]]),
            repeats=np.zeros((1,2,3)),longest=np.zeros((1,2,3)))
        rows=[dict(remaining_budget=4,trial=400)]
        result=summary_for(record,rows,np.array([True]),np.array([[False,False,False,False]]),2)
        self.assertEqual(result['success_rate'],.5)
        self.assertAlmostEqual(result['wall_action_fraction'],2/7)
        self.assertAlmostEqual(result['discounted_return'],.99**2/2)


if __name__ == '__main__':
    unittest.main()
