# Forward–Backward navigation experiments

JAX and PyTorch FB implementations, Continuous Maze reproduction, and Four Rooms
doorway experiments. Run commands below from the repository root with your
existing Python environment.

## Project layout

```text
fb_navigation/
  agents/          JAX and PyTorch FB implementations
  environments/    Four Rooms, adjustable doorways, and Continuous Maze
  training/        Training runners, smoke runs, and sweep implementation
  evaluation/      Goal-reaching and action-value evaluation
  diagnostics/     Wall traps, learned rewards, doorway stages, and bottlenecks
tests/             Automated environment, learner, and evaluation tests
notebooks/         Interactive Four Rooms notebook
experiments/       Configurations, saved runs, analyses, reference code, and reports
models/            Existing standalone checkpoints
docs/              Method notes, layout guide, and validation records
scripts/maintenance/  Project organization tools
archive/drafts/    Unfinished historical code, retained for reference
```

The small Python files remaining at the root are compatibility entry points.
Their implementations live in `fb_navigation`; the usual training commands and
legacy learner/environment imports continue to work.

## Train and evaluate

For example, train the strict-tolerance width-2 replay-continuing experiment:

```bash
CUDA_VISIBLE_DEVICES=0 python run_fourrooms_doorways.py \
  --config experiments/fourrooms_doorways/replay_continuing_threshold025.json \
  --device cuda --doorway-scale 2 \
  --checkpoint-every 200 --recovery-every 200 --log-every 20 \
  --seed 0 --output experiments/fourrooms_doorways/runs/NEW_RUN_NAME
```

The package entry point is equivalent:

```bash
python -m fb_navigation.training.run_fourrooms_doorways --help
python -m fb_navigation.evaluation.evaluate_width2_goals --help
python -m fb_navigation.diagnostics.investigate_doorway_decline --help
```

Experiment instructions and results:

- [Four Rooms experiment index](experiments/fourrooms_doorways/INDEX.md)
- [Smaller goal threshold: training and comparison](experiments/fourrooms_doorways/THRESHOLD025.md)
- [Matched strict-threshold results](experiments/fourrooms_doorways/analyses/replay_threshold025_comparison/REPORT.md)
- [Learned-reward versus Q-value wall-trap diagnosis](experiments/fourrooms_doorways/analyses/implicit_reward_wall_traps/REPORT.md)
- [Continuous Maze reproduction](experiments/continuous_maze/README.md)
- [Reference-compatible Four Rooms](experiments/fourrooms_reference/README.md)

## Tests and notebook

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Open [the Four Rooms notebook](notebooks/forward_backwards_4rooms.ipynb). Its first
cell locates the project root so imports and relative data paths work from the
new folder. Historical checkpoint selections in later cells are retained.

## Reproducibility

Saved runs, checkpoint bytes, evaluation arrays, logs, and archived experiment
sources retain their original paths and contents. Active source imports and
locations have changed. Existing source-hash guards still require the original
archived source when resuming an older run; use a fresh output directory for new
experiments or evaluations under the organized package.

See [the layout guide](docs/PROJECT_LAYOUT.md) for module paths and the migration
record. Checkpoints and Python caches remain ignored by Git.
