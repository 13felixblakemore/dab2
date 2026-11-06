"""Compatibility entry point; implementation: fb_navigation.environments.adjustable_four_rooms."""
import sys as _sys
if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module('fb_navigation.environments.adjustable_four_rooms', run_name="__main__")
else:
    import importlib as _importlib
    _implementation = _importlib.import_module('fb_navigation.environments.adjustable_four_rooms')
    globals().update({k: v for k, v in vars(_implementation).items() if not (k.startswith("__") and k.endswith("__"))})
    _sys.modules[__name__] = _implementation
