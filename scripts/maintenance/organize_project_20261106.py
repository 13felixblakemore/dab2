"""One-time project layout migration; no experiment data or archived sources are edited."""
import argparse
import ast
import hashlib
import json
import pprint
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {
    'agents': ['forward_backward', 'forward_backward_jax'],
    'environments': ['continuous_four_rooms', 'continuous_four_rooms_jax', 'continuous_maze', 'adjustable_four_rooms'],
    'training': ['run_fb', 'run_fb_jax', 'run_continuous_maze', 'run_fourrooms_reference',
                 'run_fourrooms_doorways', 'smoke_goal_terminated_fb_jax'],
    'evaluation': ['evaluate_width2_goals', 'evaluate_original_goals_jax', 'evaluate_torch_goals',
                   'evaluate_epsilon_bottlenecks', 'evaluate_fb_action_values', 'fourrooms_reference_diagnostics'],
    'diagnostics': ['analyse_fb_rewards_failures', 'audit_doorway_goals', 'compare_bottleneck_policies',
                    'compare_navigation_stages', 'control_bfs_bottlenecks', 'diagnose_continuous_maze',
                    'diagnose_fb_walls', 'estimate_fb_bottlenecks_jax', 'investigate_doorway_decline',
                    'sweep_fb_temperature', 'track_doorway_advantages'],
}
DATED = ['record_replay_failures_20261106', 'analyse_threshold025_results_20261106',
         'diagnose_implicit_reward_traps_20261106', 'geometry_detour_implicit_reward_20261106',
         'summarize_implicit_reward_traps_20261106']
LAUNCHERS = {'run_fb', 'run_fb_jax', 'run_continuous_maze', 'run_fourrooms_reference',
             'run_fourrooms_doorways', 'evaluate_width2_goals'}
LEGACY_IMPORTS = set(GROUPS['agents'] + GROUPS['environments'])


def within(path):
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f'Path outside project: {resolved}')
    return resolved


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def plan():
    moves = {f'{stem}.py': f'fb_navigation/{group}/{stem}.py'
             for group, stems in GROUPS.items() for stem in stems}
    moves.update({f'experiments/maintenance/{stem}.py': f'fb_navigation/diagnostics/{stem}.py' for stem in DATED})
    moves['experiments/fb_width2_lr_polyak_sweep_20261106.py'] = 'fb_navigation/training/fb_width2_lr_polyak_sweep_20261106.py'
    moves.update({p.name: f'tests/{p.name}' for p in ROOT.glob('test_*.py')})
    moves.update({p.name: f'docs/{p.name}' for p in ROOT.glob('*.md')})
    moves['dab.py'] = 'archive/drafts/dab.py'
    moves['forward_backwards_4rooms.ipynb'] = 'notebooks/forward_backwards_4rooms.ipynb'
    moves['experiments/maintenance/organization_baseline_tests.log'] = 'docs/validation/organization_baseline_tests.log'
    return moves


def module_mapping(moves):
    modules = {}
    for old, new in moves.items():
        if new.startswith('fb_navigation/') and new.endswith('.py'):
            module = new[:-3].replace('/', '.')
            modules[Path(old).stem] = module
            if '/' in old:
                modules[old[:-3].replace('/', '.')] = module
    return modules


def rewrite(text, modules, destination):
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines: offsets.append(offsets[-1]+len(line))
    changes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in modules:
            start = offsets[node.lineno-1]+node.col_offset+5
            changes.append((start, start+len(node.module), modules[node.module]))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in modules:
                    start = offsets[alias.lineno-1]+alias.col_offset
                    value = modules[alias.name] + ('' if alias.asname else f' as {alias.name}')
                    changes.append((start, start+len(alias.name), value))
        elif isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute)):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            if name == 'patch' and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                arg = node.args[0]
                for old in sorted(modules, key=len, reverse=True):
                    if arg.value.startswith(old+'.'):
                        start = offsets[arg.lineno-1]+arg.col_offset
                        end = offsets[arg.end_lineno-1]+arg.end_col_offset
                        changes.append((start, end, repr(modules[old]+arg.value[len(old):])))
                        break
    for start, end, value in sorted(changes, reverse=True): text = text[:start]+value+text[end:]
    needs_paths = False
    for old in ['ROOT = Path(__file__).resolve().parent\n', 'ROOT=Path(__file__).resolve().parent\n',
                'ROOT=Path(__file__).resolve().parents[1]\n']:
        if old in text:
            text = text.replace(old, 'ROOT = PROJECT_ROOT\n'); needs_paths = True
    # Test data and references stay at the project root after tests move.
    text, count = re.subn(r"Path\(__file__\)\.parent\s*/\s*(['\"])experiments/", r"PROJECT_ROOT / \1experiments/", text)
    needs_paths |= bool(count)
    if re.search(r'ROOT\s*/\s*(?:s|source|name)\b', text):
        text = re.sub(r'ROOT\s*/\s*(s|source|name)\b', r'source_path(\1)', text)
        needs_paths = True
    if destination.endswith('summarize_implicit_reward_traps_20261106.py'):
        text = text.replace("ROOT/'test_implicit_reward_rollout.py'", "ROOT/'tests/test_implicit_reward_rollout.py'")
    if needs_paths:
        tree = ast.parse(text)
        first = tree.body[0]
        insert = first.end_lineno if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str) else 0
        while insert < len(text.splitlines()) and text.splitlines()[insert].startswith('from __future__ import'):
            insert += 1
        rows = text.splitlines(keepends=True)
        rows.insert(insert, 'from fb_navigation.paths import PROJECT_ROOT, source_path\n')
        text = ''.join(rows)
    ast.parse(text)
    return text


def wrapper(module, nested=False):
    setup = ('from pathlib import Path as _Path\nimport sys as _sys\n'
             '_root = str(_Path(__file__).resolve().parents[2])\n'
             'if _root not in _sys.path:\n    _sys.path.insert(0, _root)\n') if nested else 'import sys as _sys\n'
    return (f'"""Compatibility entry point; implementation: {module}."""\n'+setup+
        'if __name__ == "__main__":\n    import runpy as _runpy\n'
        f'    _runpy.run_module({module!r}, run_name="__main__")\n'
        'else:\n    import importlib as _importlib\n'
        f'    _implementation = _importlib.import_module({module!r})\n'
        '    globals().update({k: v for k, v in vars(_implementation).items() if not (k.startswith("__") and k.endswith("__"))})\n'
        '    _sys.modules[__name__] = _implementation\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true'); args = parser.parse_args()
    moves = plan(); modules = module_mapping(moves)
    for old, new in moves.items():
        if not within(ROOT/old).is_file(): raise FileNotFoundError(old)
        if within(ROOT/new).exists(): raise FileExistsError(new)
    if not args.apply:
        print(json.dumps(moves, indent=2)); return
    protected = {}
    for folder in ('experiments', 'models'):
        for path in (ROOT/folder).rglob('*'):
            rel = path.relative_to(ROOT).as_posix()
            if not path.is_file() or '__pycache__' in path.parts: continue
            if rel.startswith('experiments/maintenance/') or rel in moves: continue
            if path.suffix == '.md' and len(path.relative_to(ROOT).parts) == 3: continue
            protected[rel] = digest(path)
    source_files = {Path(old).name: new for old, new in moves.items() if new.startswith('fb_navigation/')}
    source_files['paths.py'] = 'fb_navigation/paths.py'
    package = ROOT/'fb_navigation'
    for group in ('', *GROUPS):
        directory = within(package/group); directory.mkdir(parents=True, exist_ok=True)
        (directory/'__init__.py').write_text('"""Forward-backward navigation experiments."""\n', encoding='utf-8')
    paths_source = ('"""Project paths and canonical source locations for reproducible snapshots."""\n'
                    'from pathlib import Path\n\nPROJECT_ROOT = Path(__file__).resolve().parents[1]\n'
                    f'SOURCE_FILES = {pprint.pformat(source_files, sort_dicts=False, width=100)}\n\n'
                    'def source_path(name):\n    """Resolve historical source basenames to active implementations."""\n'
                    '    return PROJECT_ROOT / SOURCE_FILES.get(str(name), str(name))\n')
    (package/'paths.py').write_text(paths_source, encoding='utf-8')
    records = []
    for old, new in moves.items():
        source, target = within(ROOT/old), within(ROOT/new)
        before = digest(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
        if target.suffix == '.py' and new != 'archive/drafts/dab.py':
            target.write_text(rewrite(target.read_text(encoding='utf-8-sig'), modules, new), encoding='utf-8', newline='\n')
        records.append(dict(old=old, new=new, sha256_before=before, sha256_after=digest(target)))
    (ROOT/'tests/__init__.py').write_text('"""Automated navigation and FB validation."""\n', encoding='utf-8')
    for stem in sorted(LAUNCHERS | LEGACY_IMPORTS):
        (ROOT/f'{stem}.py').write_text(wrapper(modules[stem]), encoding='utf-8')
    for stem in DATED:
        (ROOT/f'experiments/maintenance/{stem}.py').write_text(wrapper(modules[stem], nested=True), encoding='utf-8')
    sweep = 'fb_width2_lr_polyak_sweep_20261106'
    # The older sweep path is one directory below the project root.
    (ROOT/f'experiments/{sweep}.py').write_text(wrapper(modules[sweep], nested=True).replace('.parents[2]', '.parents[1]'), encoding='utf-8')
    # Historical sources, configurations and outputs are deliberately outside this rewrite.
    broken = [path for path, sha in protected.items() if not (ROOT/path).is_file() or digest(ROOT/path) != sha]
    manifest = dict(moves=records, protected_files=protected, protected_count=len(protected), protected_unchanged=not broken,
                    package='fb_navigation', compatibility_launchers=sorted(LAUNCHERS | LEGACY_IMPORTS))
    (ROOT/'docs/project_organization_20261106.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if broken: raise RuntimeError(f'Protected files changed: {broken}')
    print(json.dumps(dict(moved=len(records), protected_files_unchanged=len(protected), root_launchers=len(LAUNCHERS | LEGACY_IMPORTS)), indent=2))


if __name__ == '__main__': main()
