import unittest
import numpy as np
from adjustable_four_rooms import AdjustableFourRooms
from audit_doorway_goals import point_status,component_sizes


class GoalAuditTests(unittest.TestCase):
    def test_detects_walls_without_confusing_doors(self):
        env=AdjustableFourRooms(doorway_scale=2,include_stay_action=False)
        status=point_status([[.6,.3],[.05,.4],[.6,.125],[.2,.2],[1.01,.5]],env)
        self.assertEqual(status['inside_wall'],2)
        self.assertEqual(status['native_float32_inside_wall'],2)
        self.assertEqual(status['outside_world'],1)
        self.assertEqual(status['bad_indices'],[0,1,4])

    def test_detects_disconnected_graph(self):
        graph=np.array([[1,-1],[0,-1],[-1,-1]])
        self.assertEqual(component_sizes(graph),[2,1])


if __name__=='__main__':unittest.main()
