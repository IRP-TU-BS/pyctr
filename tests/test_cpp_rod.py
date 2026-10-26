"""
C++ rod models (pyctr_cpp, built from cpp/) against the Python reference pyctr.cosserat_rod and
pyctr.cosserat_rod_force_along.

Tolerances: the ODE right hand sides are the same formulas, they agree to rounding (1e-12 relative).
Python integrates with odeint (LSODA, rtol = atol = 1.5e-8 per step), C++ with dopri5 (rtol 1e-10), so the
integrated states agree to the accumulated LSODA error, 1e-6 relative to the largest value of each quantity.
With the sharp Gaussian loads of the ExtForceRod tests the LSODA error is larger, 5e-6 relative (measured
against odeint with rtol 1e-13, where dopri5 is within 1e-11), so these use 1e-5.
The shooting methods use different solvers (MINPACK lm with ftol 1e-6 in Python, Levenberg-Marquardt to
|F| <= 1e-10 in C++), so solved results are checked by their boundary conditions and against Python to 1e-5.
"""

import unittest

import numpy as np

from pyctr.cosserat_rod import CosseratRod, CurvedCosseratRod
from pyctr.cosserat_rod_force_along import CurvedCosseratRodExt, Force_Model, StraightCosseratRod
from pyctr.utils import Rz

try:
    import pyctr_cpp

    HAVE_CPP = all(
        hasattr(pyctr_cpp, c)
        for c in ("CosseratRod", "CurvedCosseratRod", "StraightCosseratRod", "CurvedCosseratRodExt")
    )
except ImportError:
    HAVE_CPP = False

P0 = np.array([[0.01, -0.02, 0.03]])
R_TILT = Rz(0.4) @ np.array([[1, 0, 0], [0, np.cos(0.3), -np.sin(0.3)], [0, np.sin(0.3), np.cos(0.3)]])
WRENCHES = [np.zeros(6), np.array([0.01, -0.02, 0.003, 1e-4, -2e-4, 5e-5]), np.array([-0.05, 0.0, 0.0, 0.0, 3e-4, 0.0])]
CURVED = {"L": 0.1, "kappa": 8.0, "straight_length": 0.04, "r_inner": 0.5e-3, "r_outer": 1.25e-3}


def pair(name, params=None, p0=P0, R0=R_TILT, **kwargs):
    python_class = {
        "CosseratRod": CosseratRod,
        "CurvedCosseratRod": CurvedCosseratRod,
        "StraightCosseratRod": StraightCosseratRod,
        "CurvedCosseratRodExt": CurvedCosseratRodExt,
    }[name]
    rods = python_class(None if params is None else dict(params), **kwargs), getattr(pyctr_cpp, name)(params, **kwargs)
    for rod in rods:
        rod.set_initial_conditions(p0, R0)
    return rods


def assert_states_close(test, a, b, rtol=1e-6):
    """p, R, wrench of apply_force: same shapes, each within rtol of its largest value"""
    for x, y in zip(a, b):
        test.assertEqual(x.shape, y.shape)
        np.testing.assert_allclose(y, x, rtol=0, atol=rtol * max(np.abs(x).max(), 1e-12))


@unittest.skipIf(not HAVE_CPP, "pyctr_cpp with the rod classes is not built")
class TestCppRodParams(unittest.TestCase):
    def assert_params_equal(self, py_rod, cpp_rod):
        for key, value in py_rod.params.items():
            np.testing.assert_allclose(cpp_rod.params[key], value, rtol=1e-15, err_msg=key)

    def test_defaults_and_derived_values(self):
        self.assert_params_equal(CosseratRod(), pyctr_cpp.CosseratRod())
        self.assert_params_equal(CurvedCosseratRod(), pyctr_cpp.CurvedCosseratRod())
        self.assertEqual(pyctr_cpp.CurvedCosseratRod().params["kappa"], 0.16)  # params=None
        self.assertEqual(pyctr_cpp.CurvedCosseratRod({}).params["kappa"], 0.0)

    def test_custom_params(self):
        params = {**CURVED, "E": 2e9, "G": 7e8, "rho": 1000, "g": 3.0}
        for name in ("CosseratRod", "CurvedCosseratRod", "StraightCosseratRod", "CurvedCosseratRodExt"):
            py_rod, cpp_rod = pair(name, params)
            self.assert_params_equal(py_rod, cpp_rod)
        # vector kappa [0, kappa, 0] is reduced to the scalar
        self.assertEqual(pyctr_cpp.CurvedCosseratRod({"kappa": np.array([0, 5.0, 0])}).params["kappa"], 5.0)

    def test_set_params(self):
        rod = pyctr_cpp.CurvedCosseratRod(CURVED)
        rod.set_params({"beta": 0.25, "alpha": 1.0, "r_outer": 2e-3})
        self.assertEqual((rod.params["beta"], rod.params["alpha"], rod.params["kappa"]), (0.25, 1.0, 8.0))
        # C++ derives Kbt from the new radius (Python keeps the value computed in the constructor)
        np.testing.assert_allclose(rod.params["Kbt"], CurvedCosseratRod({**CURVED, "r_outer": 2e-3}).params["Kbt"])
        with self.assertRaises(ValueError):
            rod.set_params({"Kbt": np.eye(3)})


@unittest.skipIf(not HAVE_CPP, "pyctr_cpp with the rod classes is not built")
class TestCppRodOde(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.states = []
        for _ in range(5):
            state = rng.normal(size=18) * np.r_[np.full(3, 0.1), np.ones(9), np.full(3, 0.05), np.full(3, 1e-3)]
            state[3:12] = (Rz(rng.normal()) @ R_TILT).flatten()
            self.states.append(state)

    def check(self, py_rod, cpp_rod, method):
        for state in self.states:
            for s in (0.0, 0.03, 0.08):
                a = getattr(py_rod, method)(state, s)
                np.testing.assert_allclose(getattr(cpp_rod, method)(state, s), a, rtol=0, atol=1e-12 * np.abs(a).max())

    def test_odes(self):
        for name in ("CosseratRod", "CurvedCosseratRod", "StraightCosseratRod", "CurvedCosseratRodExt"):
            py_rod, cpp_rod = pair(name, CURVED)
            for rod in (py_rod, cpp_rod):
                rod.set_kappa(8.0)
                if "Ext" in name or "Straight" in name:
                    rod._gaussians = [((0.1, -0.05), 1000, 0.05), ((0.02, 0.03), 300, 0.09)]
            with self.subTest(name=name):
                self.check(py_rod, cpp_rod, "cosserate_rod_ode")
                if "Curved" in name:  # Python returns 24 values instead of 18, C++ raises as apply_force does
                    self.assertEqual(py_rod.cosserat_rod_ode_curvature(self.states[0], 0).shape, (24,))
                    with self.assertRaises(NotImplementedError):
                        cpp_rod.cosserat_rod_ode_curvature(self.states[0], 0)
                else:
                    self.check(py_rod, cpp_rod, "cosserat_rod_ode_curvature")
        py_rod, cpp_rod = pair("CosseratRod")
        for n in (12, 21):  # Python raises ValueError for a state with the wrong length
            for method in ("cosserate_rod_ode", "cosserat_rod_ode_curvature"):
                for rod in (py_rod, cpp_rod):
                    with self.subTest(n=n, method=method, rod=type(rod).__module__), self.assertRaises(ValueError):
                        getattr(rod, method)(np.ones(n), 0.0)

    def test_get_u_div(self):
        py_rod, cpp_rod = pair("CosseratRod")
        for m in (np.array([1e-3, -2e-3, 5e-4]), np.array([[1e-3], [-2e-3], [5e-4]])):  # result has the shape of m
            a = py_rod.get_u_div(R_TILT, np.zeros(3), m, np.zeros(3))
            b = cpp_rod.get_u_div(R_TILT, np.zeros(3), m, np.zeros(3))
            self.assertEqual(b.shape, a.shape)
            np.testing.assert_allclose(b, a, rtol=1e-12)

    def test_hookes_law(self):
        py_rod, cpp_rod = pair("CurvedCosseratRod", CURVED)
        for rod in (py_rod, cpp_rod):
            rod.set_kappa(8.0)
        R = np.stack([s[3:12] for s in self.states])
        u = np.stack([s[15:] for s in self.states]) * 100
        for R_in in (R, R.reshape(-1, 3, 3)):
            np.testing.assert_allclose(cpp_rod.calc_m_from_curvature(R_in, u), py_rod.calc_m_from_curvature(R_in, u),
                                       rtol=1e-12)
            np.testing.assert_allclose(cpp_rod.calc_u_from_m(R_in, u), py_rod.calc_u_from_m(R_in, u), rtol=1e-12)
        py_base, cpp_base = pair("CosseratRod")
        np.testing.assert_allclose(cpp_base.calc_m_from_curvature(R, u), py_base.calc_m_from_curvature(R, u),
                                   rtol=1e-12)

    def test_kappa_and_sections(self):
        py_rod, cpp_rod = pair("CurvedCosseratRod", CURVED)
        with self.assertRaises(AttributeError):  # cur_kappa is set by set_kappa or apply_force
            cpp_rod.get_kappa()
        cpp_rod.set_kappa(3.0)
        np.testing.assert_array_equal(cpp_rod.get_kappa(), [0, 3.0, 0])
        py_base, cpp_base = pair("CosseratRod", CURVED)
        for beta in (0.0, 0.3):
            for rod in (py_rod, py_base):
                rod.params["beta"] = beta
            for rod in (cpp_rod, cpp_base):
                rod.set_params({"beta": beta})
            for s in (0.0, 0.01, 0.04, 0.041, 0.07, 0.0701, 0.1, 0.12):
                self.assertEqual(cpp_rod.is_curved_or_at_end(s), py_rod.is_curved_or_at_end(s), (beta, s))
                self.assertEqual(cpp_base.is_curved_or_at_end(s), py_base.is_curved_or_at_end(s), (beta, s))  # None in curve


@unittest.skipIf(not HAVE_CPP, "pyctr_cpp with the rod classes is not built")
class TestCppRodApplyForce(unittest.TestCase):
    def test_straight_rod(self):
        py_rod, cpp_rod = pair("CosseratRod")
        for wrench in WRENCHES:
            for s_l in (1, 2, 20, 57):
                for ci in (True, False):
                    with self.subTest(wrench=wrench, s_l=s_l, ci=ci):
                        assert_states_close(self, py_rod.apply_force(wrench, s_l, ci), cpp_rod.apply_force(wrench, s_l, ci))
        assert_states_close(self, py_rod.apply_force(WRENCHES[1]), cpp_rod.apply_force(WRENCHES[1]))  # defaults

    def test_curved_rod(self):
        cases = [  # (straight_length, beta, steps): straight and curved, curved only, retracted, all straight,
            # one straight point, no exposed length
            (0.04, 0.0, 100), (0.0, 0.0, 30), (0.04, 0.3, 50), (0.04, 0.5, 40), (0.1, 0.0, 10), (0.08, 0.2, 7),
            (0.0015, 0.0, 100), (0.04, 1.0, 5), (0.04, 0.0, 2),
        ]
        for straight_length, beta, steps in cases:
            py_rod, cpp_rod = pair("CurvedCosseratRod", {**CURVED, "straight_length": straight_length})
            py_rod.params["beta"] = beta
            cpp_rod.set_params({"beta": beta})
            for wrench in WRENCHES:
                with self.subTest(straight_length=straight_length, beta=beta, steps=steps, wrench=wrench):
                    assert_states_close(self, py_rod.apply_force(wrench, steps), cpp_rod.apply_force(wrench, steps))
                    np.testing.assert_array_equal(cpp_rod.get_kappa(), py_rod.get_kappa())
        with self.assertRaises(NotImplementedError):
            cpp_rod.apply_force(np.zeros(6), 10, True)
        for rod in (py_rod, cpp_rod):  # Python divides by steps
            with self.assertRaises(ZeroDivisionError):
                rod.apply_force(np.zeros(6), 0)
        # straight part longer than the exposed length: more straight points than steps, numpy raises
        _, cpp_rod = pair("CurvedCosseratRod", {**CURVED, "straight_length": 0.15, "beta": 0})
        cpp_rod.set_params({"beta": 0.2})
        with self.assertRaises(ValueError):
            cpp_rod.apply_force(np.zeros(6), 7)

    def test_curved_rod_is_circular_arc(self):
        """same as tests/test_theory.py: unloaded rod with curvature kappa is an arc of radius 1/kappa"""
        _, rod = pair("CurvedCosseratRod", {**CURVED, "straight_length": 0.0, "g": 0}, p0=np.zeros((1, 3)), R0=np.eye(3))
        p, _, _ = rod.apply_force(np.zeros(6), steps=400)
        np.testing.assert_allclose(np.linalg.norm(p - [1 / 8.0, 0, 0], axis=1), 1 / 8.0, atol=1e-9)

    def test_ext_force_rods(self):
        gaussians = [((0.1, -0.05), 1000, 0.05), ((0.02, 0.03), 300, 0.09)]
        for name in ("StraightCosseratRod", "CurvedCosseratRodExt"):
            py_rod, cpp_rod = pair(name, CURVED)
            for rod in (py_rod, cpp_rod):
                rod._gaussians = gaussians
            for wrench in WRENCHES[:2]:
                for ci in (False,) if "Curved" in name else (True, False):
                    with self.subTest(name=name, wrench=wrench, ci=ci):
                        assert_states_close(self, py_rod.apply_force(wrench, 60, ci), cpp_rod.apply_force(wrench, 60, ci),
                                            rtol=1e-5)


@unittest.skipIf(not HAVE_CPP, "pyctr_cpp with the rod classes is not built")
class TestCppRodExternalForces(unittest.TestCase):
    def test_gaussian_forces(self):
        for gaussians in ([], [((0, 0), 0, 0)], [((0.1, -0.05), 1000, 0.05), ((0.02, 0.03), 300, 0.09)]):
            py_rod, cpp_rod = pair("CurvedCosseratRodExt", CURVED)
            if gaussians != [((0, 0), 0, 0)]:  # the default
                py_rod._gaussians = gaussians
                cpp_rod._gaussians = gaussians
            for s in (0.0, 0.05, 0.08, 0.2):
                np.testing.assert_allclose(cpp_rod.external_gauss_forces(s), py_rod.external_gauss_forces(s), rtol=1e-14)
        np.testing.assert_array_equal(pyctr_cpp.CosseratRod().external_gauss_forces(0.1), np.zeros((3, 1)))
        # _gaussians is a copy: changing it in place raises instead of being ignored, assigning a list works
        with self.assertRaises(AttributeError):
            cpp_rod._gaussians.append(((0.5, 0), 0, 0))
        with self.assertRaises(TypeError):
            cpp_rod._gaussians[0] = ((0.5, 0), 0, 0)

    def test_force_model(self):
        for model in (Force_Model.GAUSSIAN, Force_Model.FOURIER, pyctr_cpp.Force_Model.FOURIER):
            _, cpp_rod = pair("CurvedCosseratRodExt", CURVED, force_model=model)
            self.assertTrue(cpp_rod._force_model == Force_Model[model.name])
            self.assertTrue(Force_Model[model.name] == cpp_rod._force_model)
            other = Force_Model.FOURIER if model.name == "GAUSSIAN" else Force_Model.GAUSSIAN
            self.assertTrue(cpp_rod._force_model != other)
            self.assertFalse(cpp_rod._force_model == other)

    def test_fourier_forces(self):
        py_rod, cpp_rod = pair("CurvedCosseratRodExt", CURVED, force_model=Force_Model.FOURIER)
        with self.assertRaises(AttributeError):  # the coefficients are not set
            cpp_rod.external_fourier_forces(0.05)
        coefficients = {"_ax": [0.01, 0.02, -0.01], "_ay": [-0.005, 0.01], "_bx": [9.0, 0.003], "_by": [0.0]}
        for rod in (py_rod, cpp_rod):
            for key, value in coefficients.items():
                setattr(rod, key, np.array(value))
        for s in (0.0, 0.033, 0.1):
            np.testing.assert_allclose(cpp_rod.external_fourier_forces(s), py_rod.external_fourier_forces(s), rtol=1e-14)
        assert_states_close(self, py_rod.apply_external_forces(), cpp_rod.apply_external_forces(), rtol=1e-5)

    def test_apply_external_forces(self):
        for name in ("StraightCosseratRod", "CurvedCosseratRodExt"):
            py_rod, cpp_rod = pair(name, CURVED)
            for rod in (py_rod, cpp_rod):
                rod._gaussians = [((0.1, -0.05), 1000, 0.05)]
            with self.subTest(name=name):
                py_states, cpp_states = py_rod.apply_external_forces(), cpp_rod.apply_external_forces()
                assert_states_close(self, py_states, cpp_states, rtol=1e-5)
                np.testing.assert_allclose(cpp_rod.shooting_function_external_force(cpp_states[2][0]), 0, atol=1e-10)


@unittest.skipIf(not HAVE_CPP, "pyctr_cpp with the rod classes is not built")
class TestCppRodShooting(unittest.TestCase):
    def test_push_end(self):
        for name, ci in (("CosseratRod", False), ("CosseratRod", True), ("CurvedCosseratRod", False)):
            for wrench in WRENCHES:
                py_rod, cpp_rod = pair(name, CURVED)
                with self.subTest(name=name, ci=ci, wrench=wrench):
                    py_states, cpp_states = py_rod.push_end(wrench, ci), cpp_rod.push_end(wrench, ci)
                    np.testing.assert_allclose(cpp_states[2][-1], wrench, atol=1e-10)
                    assert_states_close(self, py_states, cpp_states, rtol=1e-5)

    def test_move_end(self):
        for pL, RL in ((np.array([[0.02, -0.01, 0.17]]), R_TILT), (np.array([[0.03, 0.0, 0.16]]), Rz(0.2) @ R_TILT)):
            py_rod, cpp_rod = pair("CosseratRod")
            p, R, w = cpp_rod.move_end(pL, RL)
            self.assertEqual(p.shape, (20, 3))
            np.testing.assert_allclose(p[-1], pL[0], atol=1e-8)
            np.testing.assert_allclose(R[-1].reshape(3, 3), RL, atol=1e-7)
            # the C++ solution solves the Python boundary value problem (Python's own lm stops earlier)
            py_rod.set_bounding_values(["pL", "RL"], [pL, RL])
            np.testing.assert_allclose(py_rod.shooting_function(w[0]), 0, atol=1e-6)
        with self.assertRaises(AttributeError):  # Python: CurvedCosseratRod has no cur_kappa before apply_force
            pair("CurvedCosseratRod")[1].move_end(pL, RL)


if __name__ == "__main__":
    unittest.main()
