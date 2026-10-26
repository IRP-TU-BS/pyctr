"""
C++ implementation of pyctr (see cpp/README.md). Same classes and methods as the Python package pyctr.
Every component is a separate extension module, components that are not built are skipped.
"""

import importlib

for _name in ("_rod", "_robot", "_rod_dynamics", "_robot_dynamics", "_config"):
    try:
        _module = importlib.import_module(f".{_name}", __name__)
    except ImportError:
        continue
    globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
