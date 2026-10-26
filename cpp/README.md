# pyctr_cpp: C++ port of pyctr

The Python package `pyctr/` is the reference implementation, written to be read and learned from. This directory
contains the same models in C++17 (Eigen), with Python bindings (pybind11) that have the same classes and
methods, so `pyctr_cpp` can be used in place of `pyctr`, for example in `pyctr.drake_system`.

## Install

Needs a C++17 compiler, CMake ≥ 3.20, Eigen 3.4 or 5 and yaml-cpp on the system. pip builds the package with
scikit-build-core and gets pybind11 itself:

```bash
pip install ./cpp                 # or: rye add pyctr_cpp --path ./cpp
```

The package `pyctr_cpp` is independent of `pyctr`, so `pip install .` of the main package still needs no compiler.

For development, build in place and run the C++ checks:

```bash
cmake -S cpp -B cpp/build -G Ninja -DPython_EXECUTABLE=$(pwd)/.venv/bin/python
cmake --build cpp/build
(cd cpp/build && ctest)
export PYTHONPATH=$(pwd)/cpp/build # the package pyctr_cpp is in the build directory
```

## Use

```python
import pyctr_cpp
from pyctr.yaml_to_model import load_continous_ctcr_model

robot = load_continous_ctcr_model("example_robots/ctr_robot.yaml", pyctr_cpp.DynamicConcentricTubeRobot, dt=2e-3)
robot.init_static()
p, R, wrench = robot.step()
```

The constructors take the same arguments as in Python: pyctr rod objects (their `params`) or parameter dicts.
`pyctr_cpp.load_tubes(path)` reads a YAML robot file in C++ and returns parameter dicts.

## Components

| C++ | Python reference | Python vs C++ per call |
|---|---|---|
| `rod`: `CosseratRod`, `CurvedCosseratRod`, `StraightCosseratRod`, `CurvedCosseratRodExt`, `Force_Model` | `cosserat_rod.py`, `cosserat_rod_force_along.py` | `push_end` 222–289 ms vs 3–5 ms |
| `robot`: `ConcentricTubeContinuumRobot` and the shooting functions | `robots.py` | `fwd_static` 43 vs 0.8 ms, `push_end` 2021 vs 14 ms |
| `rod_dynamics`: `DynamicCosseratRod` | `cosserat_rod_dynamics.py` | `step` 216 vs 0.3 ms |
| `robot_dynamics`: `DynamicConcentricTubeRobot` | `cosserat_rod_dynamics.py` | `step` (186 nodes) 228 vs 11.5 ms |
| `config`: `setup_tubes`, `load_tubes` | `yaml_to_model.py` | 2 vs 0.2 ms |

Shared code: `include/pyctr/core.hpp` (types, `TubeParameters`), `integrate.hpp` (adaptive Dormand-Prince),
`solve.hpp` (Levenberg-Marquardt), `python/convert.hpp` (conversion of Python parameters).

Not ported: the visualization (`pyctr/visualization`), `TendonRobot` (empty in Python), `display_status`, and the
Drake system, which is Python (`pyctr/drake_system.py`) and works with both implementations.

## Agreement with Python

`tests/test_cpp_*.py` compare every class with the Python class on the same inputs.

- Dynamic models: same discretization, so they agree to the solver tolerance (robot: positions within 1e-9 m
  per step). The C++ robot computes the Jacobian per interval and uses its own block elimination, the Newton
  iteration and its tolerance are the same.
- Static models: the integrator is Dormand-Prince (rtol 1e-10) instead of LSODA, the results agree within about
  1e-6. Shooting problems with several equilibria can end at a different equilibrium than Python.
- Interface: attributes that return arrays are copies (`y`, `s`, `params`, ...) or views where Python users write
  into them (`alphas`, `betas`, `p_init`, `R_init`, `base_velocity`). Errors that Python only finds late (or
  never) raise `ValueError` or `RuntimeError` early. The C++ dynamic models raise `RuntimeError` when a solve does
  not converge, also in cases where scipy reports success at a local minimum.
- The YAML expressions are evaluated by a small parser (numbers, `+ - * /`, parentheses) instead of `eval`.

The complete list of deviations is in the docstrings and in the test files.
