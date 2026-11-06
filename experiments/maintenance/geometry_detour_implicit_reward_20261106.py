"""Compatibility entry point; implementation: fb_navigation.diagnostics.geometry_detour_implicit_reward_20261106."""
from pathlib import Path as _Path
import sys as _sys
_root = str(_Path(__file__).resolve().parents[2])
if _root not in _sys.path:
    _sys.path.insert(0, _root)
if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module('fb_navigation.diagnostics.geometry_detour_implicit_reward_20261106', run_name="__main__")
else:
    import importlib as _importlib
    _implementation = _importlib.import_module('fb_navigation.diagnostics.geometry_detour_implicit_reward_20261106')
    globals().update({k: v for k, v in vars(_implementation).items() if not (k.startswith("__") and k.endswith("__"))})
    _sys.modules[__name__] = _implementation
