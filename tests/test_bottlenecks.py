"""BFS controls, bottleneck estimators, and policy comparisons."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from fb_navigation.agents.forward_backward_jax import IdentityFeaturiser
from fb_navigation.diagnostics.bottleneck_policies import (
    estimate,
    path_statistics,
    policy_probabilities,
    simulate_fb,
)
from fb_navigation.diagnostics.control_bfs_bottlenecks import (
    bfs_distances,
    build_graph,
    room_ids,
    sample_cross_room_pairs,
    simulate as simulate_bfs,
)
from fb_navigation.diagnostics.estimate_fb_bottlenecks_jax import (
    aggregate,
    eligibility,
    grid_regions,
    make_fb_estimator,
    make_mc_estimator,
    sample_pairs,
)
from fb_navigation.diagnostics.sweep_fb_temperature import (
    probabilities,
    simulate as simulate_temperature,
)
from fb_navigation.environments.continuous_four_rooms_jax import (
    ContinuousFourRooms,
    ContinuousWorld,
)
from fb_navigation.diagnostics.bottleneck_policies import doorway_arrivals


class BottleneckCommandTests(unittest.TestCase):
    def test_policy_modes_keep_distinct_defaults_and_options(self):
        from fb_navigation.diagnostics.bottleneck_policies import main

        for command, entry, output in (
            ('compare', 'run_comparison', 'evaluation_policy_discount_control'),
            ('epsilon', 'run_epsilon', 'evaluation_epsilon_bottlenecks'),
        ):
            with self.subTest(command=command), patch(
                f'fb_navigation.diagnostics.bottleneck_policies.{entry}'
            ) as evaluate:
                options = ['--horizon', '15', '--rollouts', '2']
                if command == 'epsilon':
                    options += ['--epsilon', '.1']
                main([command, *options])
                evaluate.assert_called_once()
                args = evaluate.call_args.args[0]
                self.assertEqual(args.horizon, 15)
                self.assertEqual(args.rollouts, 2)
                self.assertEqual(args.output, output)
                if command == 'epsilon':
                    self.assertEqual(args.epsilon, .1)


class BFSControlTests(unittest.TestCase):
    def test_chain_excludes_goal_and_is_undiscounted(self):
        graph=np.array([[1,-1,-1,-1],[0,2,-1,-1],[1,3,-1,-1],[2,-1,-1,-1]])
        d=bfs_distances(graph,np.array([0,0,0,1],bool))
        np.testing.assert_array_equal(d,[3,2,1,0])
        first,done,length=simulate_bfs(graph,d[None],np.array([0]),np.array([0]),np.arange(4),32,3,1)
        np.testing.assert_array_equal(first,np.broadcast_to([0,1,2,-1],first.shape))
        np.testing.assert_array_equal(length,3);self.assertTrue(done.all())
        np.testing.assert_array_equal((first>=0).mean(axis=0),[1,1,1,0])

    def test_uniform_shortest_action_ties(self):
        graph=np.array([[1,2,-1,-1],[0,3,-1,-1],[0,3,-1,-1],[1,2,-1,-1]])
        d=bfs_distances(graph,np.array([0,0,0,1],bool))
        first,done,length=simulate_bfs(graph,d[None],np.array([0]),np.array([0]),np.arange(4),10000,2,4)
        self.assertTrue(done.all());np.testing.assert_array_equal(length,2)
        self.assertAlmostEqual(np.mean(first[:,1]>=0),.5,delta=.02)
        self.assertAlmostEqual(np.mean(first[:,2]>=0),.5,delta=.02)

    def test_graph_connectivity_symmetry_cross_room_pairs(self):
        env=ContinuousFourRooms(include_stay_action=False)
        nodes,graph=build_graph(env,44)
        for i,row in enumerate(graph):
            for j in row[row>=0]:self.assertIn(i,graph[j])
        goal=np.zeros(len(nodes),bool);goal[0]=True
        d=bfs_distances(graph,goal);self.assertTrue(np.all(d>=0))
        for w in env.wall_blocks:
            inside=(nodes[:,0]>=w.xmin)&(nodes[:,0]<=w.xmax)&(nodes[:,1]>=w.ymin)&(nodes[:,1]<=w.ymax)
            self.assertFalse(inside.any())
        rooms=room_ids(nodes,env.size)
        u,g=sample_cross_room_pairs(nodes,rooms,100,.1,2026)
        self.assertTrue(np.all(rooms[u]!=rooms[g]))
        self.assertTrue(np.all(np.sum((nodes[u]-nodes[g])**2,axis=1)>.1**2))
        u2,g2=sample_cross_room_pairs(nodes,rooms,100,.1,2026)
        np.testing.assert_array_equal(u,u2);np.testing.assert_array_equal(g,g2)


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


class PolicyDiscountControlTests(unittest.TestCase):
    def test_discounts_reuse_first_hits_and_exclusions(self):
        first=np.array([[0,2,-1],[1,3,4]])
        eligible=np.array([[True,True,True],[True,False,True]])
        one,pair_one=estimate(first,eligible,1,1.)
        discount,pair_discount=estimate(first,eligible,1,.99)
        np.testing.assert_allclose(one,[1.,1.,.5])
        np.testing.assert_allclose(discount,[(1+.99)/2,.99**2,.99**4/2])
        self.assertTrue(np.all(pair_discount<=pair_one))
        np.testing.assert_array_equal(first,[[0,2,-1],[1,3,4]])

    def test_tabular_rollout_and_goal_exclusion(self):
        graph=np.array([[-1,1],[0,2],[1,-1]])
        terminal=np.array([[False,False,True]])
        policy=np.array([[[0.,1.],[0.,1.],[0.,1.]]])
        first,hit,blocked=simulate_fb(graph,terminal,np.array([0]),np.array([0]),np.arange(3),policy,4,5,3)
        np.testing.assert_array_equal(first,np.broadcast_to([0,1,-1],first.shape))
        np.testing.assert_array_equal(hit,2);np.testing.assert_array_equal(blocked,0)
        stats=path_statistics(hit,5,np.array([2]),4)
        self.assertEqual(stats['success_rate'],1.)
        self.assertEqual(stats['mean_successful_length_ratio_to_bfs'],1.)

    def test_censoring_is_reported(self):
        graph=np.array([[-1,1],[0,2],[1,-1]])
        terminal=np.array([[False,False,True]])
        policy=np.array([[[1.,0.],[1.,0.],[1.,0.]]])
        first,hit,blocked=simulate_fb(graph,terminal,np.array([0]),np.array([0]),np.arange(3),policy,4,5,3)
        np.testing.assert_array_equal(hit,6)
        stats=path_statistics(hit,5,np.array([2]),4)
        self.assertEqual(stats['success_rate'],0.)
        self.assertIsNone(stats['mean_successful_steps'])
        self.assertEqual(stats['mean_capped_steps'],5.)
        np.testing.assert_array_equal(blocked,5)


class EpsilonBottleneckTests(unittest.TestCase):
    def test_exact_epsilon_policy(self):
        scores=jnp.array([[1.,4.,3.,2.],[5.,1.,0.,-1.]])
        pi=policy_probabilities(scores,'epsilon',200.,.02)
        np.testing.assert_allclose(pi,[[.005,.985,.005,.005],[.985,.005,.005,.005]],rtol=1e-6)
        np.testing.assert_allclose(np.asarray(pi).sum(axis=1),1.)
        np.testing.assert_allclose(policy_probabilities(scores,'epsilon',200.,1.),.25)
        np.testing.assert_array_equal(policy_probabilities(scores,'epsilon',200.,0.),[[0,1,0,0],[1,0,0,0]])
        expected=np.exp(np.asarray(scores)/200);expected/=expected.sum(axis=1,keepdims=True)
        np.testing.assert_allclose(policy_probabilities(scores,'softmax',200.,.02),expected,rtol=1e-6)

    def test_arrival_definition_and_exclusions(self):
        points=np.array([[6.3,1.4],[6.7,1.4],[3.,8.]])/11
        first=np.array([[5,8,0],[10,-1,1]])
        eligible=np.array([[False,True,True],[True,True,True]])
        stats=doorway_arrivals(first,eligible,points,1)
        self.assertEqual(stats['any_door']['mean_first_any_cell'],9.)
        self.assertEqual(stats['any_door']['mean_pooled_cell_arrival'],9.)
        self.assertEqual(stats['any_door']['arrival_rate'],1.)
        self.assertEqual(stats['per_door']['top']['arrival_rate'],0.)
        self.assertIsNone(stats['per_door']['top']['mean_first_any_cell'])


class TemperatureSweepTests(unittest.TestCase):
    def test_stable_softmax_and_entropy(self):
        q=np.array([[[100001.,100000.,99999.,99998.]]])
        warm=probabilities(q,'softmax',200.);cold=probabilities(q,'softmax',.2)
        np.testing.assert_allclose(warm.sum(axis=-1),1.)
        self.assertTrue(np.isfinite(cold).all())
        self.assertGreater(cold.max(),warm.max())
        np.testing.assert_allclose(probabilities(q,'epsilon'),[[[.985,.005,.005,.005]]])
        np.testing.assert_array_equal(probabilities(q,'greedy'),[[[1,0,0,0]]])

    def test_common_random_numbers_survive_other_trial_completion(self):
        graph=np.array([[-1,1],[0,2],[1,-1]])
        terminal=np.array([[False,False,True],[False,False,True]])
        policy=np.full((2,3,2),.5)
        policy[0]=[0,1]  # first trial finishes immediately along the chain
        slow=policy.copy();slow[0]=[1,0]  # first trial never leaves the wall
        args=(graph,terminal,np.array([0,0]),np.array([0,1]),np.arange(3))
        fast_first,fast_goal,_=simulate_temperature(*args,policy,1,30,4)
        slow_first,slow_goal,_=simulate_temperature(*args,slow,1,30,4)
        np.testing.assert_array_equal(fast_first[1],slow_first[1])
        self.assertEqual(fast_goal[1],slow_goal[1])
        self.assertEqual(fast_goal[0],2);self.assertEqual(slow_goal[0],31)


if __name__ == '__main__':
    unittest.main()
