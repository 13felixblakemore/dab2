import unittest
import numpy as np
from fb_navigation.evaluation.evaluate_width2_goals import summarize

class GoalReachingReportTests(unittest.TestCase):
    def test_censoring_and_wall_denominators(self):
        record=dict(hits=np.array([[1,31],[2,151],[30,31]]),collisions=np.zeros((3,2,3)),repeats=np.zeros((3,2,3)))
        record['collisions'][0,1,0]=3
        pairs=dict(categories=np.arange(3),bfs_lengths=np.array([1,2,30]))
        rows=summarize(record,pairs,'test','epsilon002',30)
        self.assertEqual(rows[0]['success_rate'],.5)
        self.assertEqual(rows[0]['mean_success_steps'],1)
        self.assertAlmostEqual(rows[0]['wall_action_percent'],300/31)
        self.assertEqual(rows[2]['success_rate'],.5)
        self.assertEqual(summarize(record,pairs,'test','epsilon002',150)[2]['success_rate'],1)

    def test_no_success_reports_none(self):
        record=dict(hits=np.full((3,2),151),collisions=np.zeros((3,2,3)),repeats=np.zeros((3,2,3)))
        pairs=dict(categories=np.arange(3),bfs_lengths=np.array([1,2,30]))
        for row in summarize(record,pairs,'test','greedy',150):
            self.assertEqual(row['success_rate'],0)
            self.assertIsNone(row['mean_success_steps'])
            self.assertEqual(row['success_interval95'],[0.,0.])

if __name__=='__main__':unittest.main()
