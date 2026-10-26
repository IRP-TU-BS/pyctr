import unittest
import numpy as np
from pyctr.cosserat_rod import CurvedCosseratRod
from pyctr.cosserat_rod_dynamics import DynamicCosseratRod

try:
    import pyctr_cpp
except ImportError:
    pyctr_cpp = None

STEEL = {"E": 200e9, "G": 80e9, "r_outer": 1e-3, "rho": 8000, "g": 0, "L": 0.5, "s": 0.0125}
ALONG_X = np.array([[0, 0, 1.0], [0, 1, 0], [-1, 0, 0]])  # rod axis along world x
# Both use MINPACK LM with xtol = 1.5e-8 on the base wrench, Python with the steps eps * |x| of lmdif for the Jacobian,
# C++ with eps * max(1, |x|). They take different iterates and stop at different points within xtol, so the base
# wrenches differ by up to about 3e-8 relative. Velocities
# and the wrenches of the dynamic solution are more sensitive (BDF-alpha divides by dt), measured worst case over the
# tests: 2e-6 of the scale of the quantity, at dt = 5e-4. The stable dynamics does not let the difference grow.
RTOL = 1e-5
# SI units, for quantities that are zero in theory: the static solution of Python is only exact to xtol, the force
# imbalance from this moves a rod "at rest" with up to 1e-10 m/s and rad/s.
ATOL = 1e-9
# C++ raises when the tip residual is above 1e-8 * (1 + wrench scale); measured C++ residuals are below 4e-11 N, Nm
TIP_RESIDUAL = 1e-8
LARGE_LOAD = np.array([3, 2, 1, 0.1, 0.2, 0.05])  # bends the steel rod by more than 90 degrees in 3D


def period(x, dt):
    """Mean time between upward zero crossings, as in tests/test_dynamics.py"""
    return np.mean(np.diff(np.flatnonzero(np.diff(np.sign(x)) > 0) * dt))


@unittest.skipIf(pyctr_cpp is None or not hasattr(pyctr_cpp, "DynamicCosseratRod"), "pyctr_cpp not built")
class TestCppDynamicCosseratRod(unittest.TestCase):
    def assert_close(self, cpp, py):
        """Every group of 3 columns relative to its own largest value"""
        cpp, py = np.asarray(cpp), np.asarray(py)
        self.assertEqual(cpp.shape, py.shape)
        for k in range(0, py.shape[1], 3):
            np.testing.assert_allclose(cpp[:, k : k + 3], py[:, k : k + 3], rtol=0, atol=RTOL * np.abs(py[:, k : k + 3]).max() + ATOL)

    def assert_same(self, cpp, py, tip_wrench):
        np.testing.assert_array_less(np.abs(self.cpp.y[-1, 12:18] - tip_wrench), TIP_RESIDUAL)  # a root, not a local minimum
        for c, p in zip(cpp, py):
            self.assert_close(c, p)
        self.assert_close(self.cpp.y, self.py.y)
        self.assert_close(self.cpp.z, self.py.z)

    def make(self, params, p0=np.zeros((1, 3)), R0=np.eye(3), **kwargs):
        self.py = DynamicCosseratRod(dict(params) if params is not None else None, **kwargs)
        self.cpp = pyctr_cpp.DynamicCosseratRod(dict(params) if params is not None else None, **kwargs)
        for rod in (self.py, self.cpp):
            rod.set_initial_conditions(p0, R0)
        self.assertEqual(self.cpp.N, self.py.N)
        self.assertEqual(self.cpp.dt, self.py.dt)

    def run_both(self, tip_wrench, steps=200, step_wrench=np.zeros(6)):
        """init_static with tip_wrench, then steps with step_wrench; returns the tip x of both"""
        self.assert_same(self.cpp.init_static(tip_wrench), self.py.init_static(tip_wrench), tip_wrench)
        x_cpp, x_py = [], []
        for _ in range(steps):
            c, p = self.cpp.step(step_wrench), self.py.step(step_wrench)
            self.assert_same(c, p, step_wrench)
            x_cpp.append(c[0][-1, 0])
            x_py.append(p[0][-1, 0])
        return np.array(x_cpp), np.array(x_py)

    def test_cantilever(self):
        # tests/test_dynamics.py: static deflection, release, first bending mode
        self.make(STEEL)
        EI, rhoA, L, F = self.cpp.params["Kbt"][0, 0], STEEL["rho"] * self.cpp.params["A"], 0.5, 0.02
        x_cpp, x_py = self.run_both([F, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(self.cpp.init_static([F, 0, 0, 0, 0, 0])[0][-1, 0], F * L**3 / (3 * EI), rtol=0.03)
        np.testing.assert_allclose(period(x_cpp, 2e-3), period(x_py, 2e-3), rtol=1e-12)
        np.testing.assert_allclose(period(x_cpp, 2e-3), 2 * np.pi / 1.8751**2 * np.sqrt(rhoA * L**4 / EI), rtol=0.03)

    def test_damping_and_drag(self):
        prm = {**STEEL, "Bse": np.diag([1e-3, 1e-3, 2e-3]), "Bbt": np.diag([1e-4, 1e-4, 5e-5]), "C": np.diag([1e-3, 1e-3, 0])}
        self.make(prm)
        x_cpp, _ = self.run_both([0.02, 0.005, 0.01, 0, 0.002, 0], steps=150)
        self.assertLess(np.abs(x_cpp[-50:]).max(), 0.9 * np.abs(x_cpp[:50]).max())  # the oscillation decays

    def test_curved_rod_moving_tip_load(self):
        # straight, then curved about y; base moved and turned; tip load switched on during the steps
        self.make({**STEEL, "kappa": 2.0, "straight_length": 0.2}, p0=np.array([[0.1, -0.2, 0.3]]), R0=ALONG_X)
        self.run_both([0, 0.01, 0, 0, 0, 1e-3], steps=100, step_wrench=np.array([0.01, 0, -0.01, 1e-3, 0, 0]))

    def test_gravity(self):
        # horizontal rod sagging under gravity, released from a tip force
        self.make({**STEEL, "g": 9.81, "r_outer": 2e-3}, R0=ALONG_X)
        self.run_both([0, 0, 0.05, 0, 0, 0], steps=100)
        self.assertLess(self.cpp.y[-1, 2], 0)

    def test_edge_cases(self):
        # no params (CurvedCosseratRod defaults: kappa 0.16 and gravity), L not a multiple of s, bdf_alpha 0
        self.make(None, R0=ALONG_X)
        self.run_both(np.zeros(6), steps=5)
        self.make({**STEEL, "s": 0.03}, dt=5e-4, bdf_alpha=0.0)
        self.run_both([0.01, 0, 0, 0, 0, 0], steps=20)
        # init_static again after steps starts from rest
        self.assert_same(self.cpp.init_static([0, 0.01, 0, 0, 0, 0]), self.py.init_static([0, 0.01, 0, 0, 0, 0]), [0, 0.01, 0, 0, 0, 0])

    def test_large_3d_load(self):
        # far from the start value: on this grid both find the root
        self.make({**STEEL, "s": 1e-3})
        self.assert_same(self.cpp.init_static(LARGE_LOAD), self.py.init_static(LARGE_LOAD), LARGE_LOAD)

    def test_rod_object_as_params(self):
        rod = pyctr_cpp.DynamicCosseratRod(CurvedCosseratRod(dict(STEEL)))
        rod.set_initial_conditions(np.zeros((1, 3)), np.eye(3))
        self.make(STEEL)
        np.testing.assert_allclose(rod.init_static([0.02, 0, 0, 0, 0, 0])[0], self.py.init_static([0.02, 0, 0, 0, 0, 0])[0], atol=1e-12)

    def test_errors(self):
        rod = pyctr_cpp.DynamicCosseratRod(dict(STEEL))
        with self.assertRaises(RuntimeError):
            rod.init_static()  # no initial conditions
        rod.set_initial_conditions(np.zeros(3), np.eye(3))
        with self.assertRaises(RuntimeError):
            rod.step()  # no init_static
        # 1000 N bends the Euler discretization far beyond its range: no root. (Python root reports success there
        # with a residual of 940 N, see the open issue.)
        with self.assertRaises(RuntimeError):
            rod.init_static([1000, 0, 0, 0, 0, 0])
        # MINPACK stops at a local minimum of the residual (0.5 N) and reports success, Python returns that point
        with self.assertRaises(RuntimeError):
            rod.init_static(LARGE_LOAD)
        # soft rod: MINPACK runs out of evaluations 4e-6 N away from the 1e-4 N load
        soft = pyctr_cpp.DynamicCosseratRod({**STEEL, "E": 1e6, "G": 4e5, "L": 0.3, "s": 0.01})
        soft.set_initial_conditions(np.zeros(3), np.eye(3))
        with self.assertRaises(RuntimeError):
            soft.init_static([1e-4, 0, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
