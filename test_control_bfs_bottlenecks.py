import unittest
import numpy as np
from control_bfs_bottlenecks import build_graph,bfs_distances,simulate,room_ids,sample_cross_room_pairs
from continuous_four_rooms_jax import ContinuousFourRooms


class BFSControlTests(unittest.TestCase):
    def test_chain_excludes_goal_and_is_undiscounted(self):
        graph=np.array([[1,-1,-1,-1],[0,2,-1,-1],[1,3,-1,-1],[2,-1,-1,-1]])
        d=bfs_distances(graph,np.array([0,0,0,1],bool))
        np.testing.assert_array_equal(d,[3,2,1,0])
        first,done,length=simulate(graph,d[None],np.array([0]),np.array([0]),np.arange(4),32,3,1)
        np.testing.assert_array_equal(first,np.broadcast_to([0,1,2,-1],first.shape))
        np.testing.assert_array_equal(length,3);self.assertTrue(done.all())
        np.testing.assert_array_equal((first>=0).mean(axis=0),[1,1,1,0])

    def test_uniform_shortest_action_ties(self):
        graph=np.array([[1,2,-1,-1],[0,3,-1,-1],[0,3,-1,-1],[1,2,-1,-1]])
        d=bfs_distances(graph,np.array([0,0,0,1],bool))
        first,done,length=simulate(graph,d[None],np.array([0]),np.array([0]),np.arange(4),10000,2,4)
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


if __name__=='__main__':unittest.main()
