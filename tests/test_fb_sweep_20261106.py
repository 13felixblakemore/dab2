from fb_navigation.paths import PROJECT_ROOT, source_path
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = PROJECT_ROOT
spec=importlib.util.spec_from_file_location('sweep',ROOT/'experiments/fb_width2_lr_polyak_sweep_20261106.py')
sweep=importlib.util.module_from_spec(spec);spec.loader.exec_module(sweep)

class SweepTests(unittest.TestCase):
    def test_grid_changes_only_requested_parameters(self):
        base=json.loads((ROOT/'experiments/fourrooms_reference/config.json').read_text())
        original=json.dumps(base,sort_keys=True)
        cells=sweep.make_plan(base,[5e-4,1e-4,5e-5],[.9,.95,.99],[0])
        self.assertEqual(len(cells),9);self.assertEqual(len({c['name'] for c in cells}),9)
        self.assertEqual(json.dumps(base,sort_keys=True),original)
        for cell in cells:
            expected=dict(base['learner'],learning_rate=cell['learning_rate'],polyak_coef=cell['polyak'])
            self.assertEqual(cell['config']['learner'],expected)
            self.assertEqual(cell['config']['environment']['doorway_scale'],2)
            self.assertEqual(cell['config']['epochs']*cell['config']['cycles']*cell['config']['updates'],800000)
            self.assertEqual(cell['config']['evaluation_seed'],base['evaluation_seed'])

    def test_classic_only_and_dimension_grid(self):
        replay=json.loads((ROOT/'experiments/fourrooms_doorways/replay_goal_terminated.json').read_text())
        with self.assertRaises(ValueError):sweep.make_plan(replay,[1e-4],[.99],[0])
        base=json.loads((ROOT/'experiments/fourrooms_reference/config.json').read_text())
        cells=sweep.make_plan(base,[1e-4],[.95],[0],dimensions=[25,50,100])
        self.assertEqual([c['config']['learner']['embed_dim'] for c in cells],[25,50,100])
        self.assertEqual(len({c['name'] for c in cells}),3)
        self.assertTrue(all(c['config']['learner']['z_sampling']=='cauchy' and not c['config']['learner']['goal_terminated'] for c in cells))
        with self.assertRaises(ValueError):sweep.make_plan(base,[1e-4],[.95],[0],dimensions=[0])
        for lr,p in [([0],[.95]),([1e-4],[1.]),([1e-4,1e-4],[.95])]:
            with self.assertRaises(ValueError):sweep.make_plan(base,lr,p,[0])

    def test_child_command_has_distinct_config_and_resume(self):
        base=json.loads((ROOT/'experiments/fourrooms_reference/config.json').read_text())
        cell=sweep.make_plan(base,[1e-4],[.95],[2])[0]
        cmd=sweep.command(cell,ROOT/'experiments/sweeps/test','cuda',200,20,True)
        self.assertIn('--resume',cmd);self.assertEqual(cmd[cmd.index('--seed')+1],'2')
        self.assertEqual(cmd[cmd.index('--device')+1],'cuda')
        self.assertIn(cell['name'],cmd[cmd.index('--output')+1])

if __name__=='__main__':unittest.main()
