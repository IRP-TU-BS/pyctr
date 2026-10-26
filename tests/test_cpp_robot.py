"""
Compares the C++ port pyctr_cpp.ConcentricTubeContinuumRobot with pyctr.robots.ConcentricTubeContinuumRobot.

Tolerances: Python integrates with LSODA (rtol=1e-8, atol=1e-10), C++ with Dormand-Prince (rtol=1e-10,
atol=1e-12). Against a DOP853 reference with rtol=1e-13 the error of LSODA is up to 2e-9 m in the positions
and 1e-7 in the other quantities (curvatures in 1/m of order 10), the error of C++ is smaller, so 1e-7 m and
1e-6 are used. The shooting methods solve
with the same method (MINPACK hybr, xtol=1.49e-8 relative to the unknowns) on residuals that differ by
the integration error, so their results are compared 10 times less strict.
"""

import copy
import os
import unittest

import numpy as np

from pyctr.cosserat_rod_force_along import CurvedCosseratRodExt
from pyctr.robots import ConcentricTubeContinuumRobot, shooting_function_force
from pyctr.utils import Rz
from pyctr.yaml_to_model import setup_tubes

try:
    import pyctr_cpp

    CppRobot = pyctr_cpp.ConcentricTubeContinuumRobot
except (ImportError, AttributeError):
    CppRobot = None

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "example_robots", "ctr_robot.yaml")
NYLON = {"E": 339889091.1107947, "G": 97898219.42862949, "rho": 8000, "g": 0, "s": 1e-3}


def example_params():
    out = []
    for conf in setup_tubes(CONFIG):
        conf = copy.deepcopy(conf)
        conf["s"] = 1e-3
        conf["L"] = conf["L"] * 1e-3
        conf["curved_len"] = conf["curved_len"] * 1e-3
        conf["straight_length"] = conf["L"] - conf["curved_len"]
        out.append(conf)
    return out


def three_tubes():
    """Hand-made robot: three tubes, the outer one is straight up to 60 mm and strongly curved"""
    return [
        {**NYLON, "L": 0.2, "straight_length": 0.12, "kappa": 9.0, "r_inner": 0.4e-3, "r_outer": 0.8e-3},
        {**NYLON, "L": 0.15, "straight_length": 0.1, "kappa": 6.0, "r_inner": 1.0e-3, "r_outer": 1.5e-3},
        {**NYLON, "L": 0.08, "straight_length": 0.06, "kappa": 15.0, "r_inner": 1.8e-3, "r_outer": 2.2e-3},
    ]


def both(params, **kwargs):
    rods = []
    for p in params:
        rod = CurvedCosseratRodExt(copy.deepcopy(p))
        rod.set_initial_conditions(np.array([[0, 0, 0]]), np.eye(3))
        rods.append(rod)
    return ConcentricTubeContinuumRobot(rods, **kwargs), CppRobot(rods, **kwargs)


def configure(robots, alphas, betas):
    for r in robots:
        r.rotate(alphas)
        r.translate(betas)


@unittest.skipIf(CppRobot is None, "pyctr_cpp with ConcentricTubeContinuumRobot is not built")
class TestCppRobot(unittest.TestCase):
    def assert_results(self, expected, actual, scale=1.0):
        # positions, orientations, wrenches, us, uzs, thetas
        for name, e, a, atol in zip(
            ("positions", "orientations", "wrenches", "us", "uzs", "thetas"),
            expected,
            actual,
            (1e-7, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6),
        ):
            self.assertEqual(np.shape(e), np.shape(a), name)
            np.testing.assert_allclose(a, e, rtol=0, atol=atol * scale, err_msg=name)

    def test_fwd_static_matches_python(self):
        R_init = Rz(0.4) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        cases = [
            (example_params(), {}, [0.3, -0.2], [0.0, 0.0], np.zeros(6), None),
            (example_params(), {}, [1.0, 2.0], [0.4, 0.2], [0.01, -0.02, 0.0, 1e-3, -2e-3, 1e-4], [0.5, -0.3]),
            (example_params(), {"R_init": R_init, "p_init": np.array([[0.01], [0.02], [-0.03]])}, [0, np.pi], [0.1, 0.3],
             [0.0, 0.05, 0.02, 0.0, 0.0, 0.0], None),
            (three_tubes(), {}, [0.2, 1.5, -2.0], [0.1, 0.05, 0.0], [0.02, 0.01, 0.0, 0.0, 1e-3, 0.0], [0.1, 0.2, -0.4]),
            # beta < 0 of the inner tube is valid and moves the straight/curved transition out
            (three_tubes(), {}, [0.0, 0.0, 0.0], [-0.1, 0.0, 0.3], np.zeros(6), None),
        ]
        for params, kwargs, alphas, betas, wrench, uzs0 in cases:
            with self.subTest(alphas=alphas, betas=betas):
                py, cpp = both(params, **kwargs)
                configure((py, cpp), alphas, betas)
                expected = py.fwd_static(np.asarray(wrench), 0.002, uzs0=uzs0)
                self.assert_results(expected, cpp.fwd_static(wrench, 0.002, uzs0=uzs0))
                self.assertEqual(py.seg_indexes, cpp.seg_indexes)
                self.assertEqual(py.tube_tip_indices(), cpp.tube_tip_indices())
                for attr in ("positions", "orientations", "wrenches", "uzs", "thetas"):
                    np.testing.assert_allclose(getattr(cpp, attr), getattr(py, attr), rtol=0, atol=1e-6)

    def test_calc_forward_and_ode_match_python(self):
        py, cpp = both(three_tubes())
        configure((py, cpp), [0.2, 1.5, -2.0], [0.1, 0.05, 0.0])
        wrench = np.array([0.02, 0.01, 0.0, 0.0, 1e-3, 0.0])
        state_py, us_py, seg_py = py.calc_forward(wrench, 0.003)
        state_cpp, us_cpp, seg_cpp = cpp.calc_forward(wrench, 0.003)
        np.testing.assert_allclose(state_cpp, state_py, rtol=0, atol=1e-6)
        np.testing.assert_allclose(us_cpp, us_py, rtol=0, atol=1e-6)
        self.assertEqual(seg_py, seg_cpp)
        # after calc_forward both robots hold the tubes of the last segment
        s = 0.19
        state = state_py[-3]
        np.testing.assert_allclose(cpp.cosserate_rod_ode(s, state), py.cosserate_rod_ode(s, state), rtol=1e-12, atol=1e-14)
        n = len(py.tubes)
        args = (state[18 + n :], state[15:18], s, state[18])
        np.testing.assert_allclose(cpp.calc_uxz(*args), py.calc_uxz(*args), rtol=1e-12)
        self.assertEqual(cpp.calc_uxz(*args).shape, (1, 3))
        batch = (np.array([s, s]), state_py[-2:, 18 + n :], state_py[-2:, 15:18], state_py[-2:, 18 : 18 + n])
        np.testing.assert_allclose(cpp.calc_us(*batch), py.calc_us(*batch), rtol=1e-12)

    def test_segments(self):
        py, cpp = both(three_tubes())
        for betas in ([0.0, 0.0, 0.0], [0.1, 0.05, 0.0], [0.5, 0.4, 0.5], [-0.1, 0.0, 0.3]):
            configure((py, cpp), [0, 0, 0], betas)
            for only_positive in (True, False):
                np.testing.assert_allclose(cpp.get_ordered_segments(only_positive), py.get_ordered_segments(only_positive))

    def test_segment_shorter_than_step(self):
        """A segment shorter than step_size gets a single output point and the state does not advance there,
        as in Python (numpy.linspace with one point)"""
        params = three_tubes()
        params[1]["L"] = 0.081  # 1 mm segment between the ends of tube 2 and tube 3
        py, cpp = both(params)
        configure((py, cpp), [0.3, 0.0, 1.0], [0.0, 0.0, 0.0])
        expected = py.fwd_static(np.zeros(6), 0.004)
        self.assert_results(expected, cpp.fwd_static(np.zeros(6), 0.004))
        self.assertEqual(py.seg_indexes, cpp.seg_indexes)

    def test_rotate_translate(self):
        _, cpp = both(example_params())
        cpp.rotate([0.1, 0.2])
        cpp.translate([0.3, 0.1])
        np.testing.assert_array_equal(cpp.alphas, [0.1, 0.2])
        np.testing.assert_array_equal(cpp.betas, [0.3, 0.1])
        with self.assertRaises(Exception):
            cpp.translate([0.5, 0.0])  # outer tube would end beyond the inner tube
        np.testing.assert_array_equal(cpp.betas, [0.3, 0.1])
        self.assertEqual(cpp.get_tube_outer_radii(), [1.25e-3, 2.5e-3])
        self.assertIsNone(cpp.positions)
        with self.assertRaises(ValueError):
            cpp.alphas = [0.1]
        # parameter dicts work like rod objects
        from_dicts = CppRobot(example_params())
        from_dicts.rotate([0.1, 0.2])
        from_dicts.translate([0.3, 0.1])
        np.testing.assert_array_equal(from_dicts.fwd_kinematic(0.004)[0], cpp.fwd_kinematic(0.004)[0])

    def test_beta_validity_matches_python(self):
        # the inner tube must stick out; aligned tube ends pass with 1e-12 m tolerance (b1 = 0.05: the outer tube
        # ends 1.4e-17 m after the inner one in floating point)
        aligned = 1 - 0.124 * 0.95 / 0.185
        cases = [
            ([1.0, 1.0], False),
            ([1.001, 1.0], False),
            ([0.5, 0.0], False),
            ([aligned, 0.05], True),
            ([aligned + 1e-9, 0.05], False),
            ([-0.1, 0.0], True),
            ([0.0, 1.0], True),
        ]
        py, cpp = both(example_params())
        for betas, valid in cases:
            with self.subTest(betas=betas):
                results = []
                for r in (py, cpp):
                    r.translate([0.0, 0.0])
                    try:
                        r.translate(betas)
                        results.append(True)
                    except Exception:
                        results.append(False)
                        np.testing.assert_array_equal(r.betas, [0, 0])
                self.assertEqual(results, [valid, valid])

    def test_push_end_matches_python(self):
        cases = [
            (example_params(), [0.3, -0.2], [0.0, 0.0], [0.2, 0.05, 0.0, 0.002, 0.0, 0.001]),
            (example_params(), [1.0, 2.0], [0.4, 0.2], [0.0, -0.1, 0.05, 0.0, 0.003, 0.0]),
            (three_tubes(), [0.2, 1.5, -2.0], [0.1, 0.05, 0.0], [0.05, 0.0, 0.0, 0.0, 0.0, -0.002]),
        ]
        for params, alphas, betas, wrench in cases:
            with self.subTest(alphas=alphas, betas=betas):
                py, cpp = both(params)
                configure((py, cpp), alphas, betas)
                wrench = np.asarray(wrench)
                expected = py.push_end(wrench, 0.004)
                actual = cpp.push_end(wrench, 0.004)
                self.assert_results(expected, actual, scale=10)
                self.assertEqual(cpp.boundary_values, {})
                # the boundary conditions hold: world-frame tip wrench and free torsional ends
                p, R, w, us, uzs, th = actual
                np.testing.assert_allclose(w[-1, :3], wrench[:3], atol=1e-9)
                np.testing.assert_allclose(R[-1].reshape(3, 3) @ w[-1, 3:], wrench[3:], atol=1e-9)
                tips = cpp.tube_tip_indices()
                self.assertLess(max(abs(uzs[tips[i], i]) for i in range(len(params))), 1e-8)

    def test_push_end_to_position_matches_python(self):
        py, cpp = both(example_params())
        configure((py, cpp), [0.0, np.pi], [0.1, 0.3])
        target = py.fwd_kinematic(0.004)[0][-1] + np.array([0.003, -0.002, 0.0])
        expected = py.push_end_to_position(target, 0.004)
        actual = cpp.push_end_to_position(target, 0.004)
        self.assert_results(expected, actual, scale=10)
        np.testing.assert_allclose(actual[0][-1], target, atol=1e-9)
        np.testing.assert_allclose(actual[1][-1], np.eye(3).flatten(), atol=1e-8)

    def test_gaussian_forces_match_python(self):
        gaussians = [((0.1, 0.0), 2000.0, 0.1), ((0.0, -0.05), 5000.0, 0.15)]
        py, cpp = both(example_params())
        configure((py, cpp), [0.3, -0.2], [0.1, 0.0])
        py.set_gaussians(gaussians)
        cpp.set_gaussians(gaussians)
        for s in (0.0, 0.1, 0.13):
            np.testing.assert_allclose(cpp.external_gauss_forces(s), py.external_gauss_forces(s), rtol=1e-14)
        expected = py.fwd_external_gaussian_forces(0.004)
        actual = cpp.fwd_external_gaussian_forces(0.004)
        self.assert_results(expected, actual, scale=10)
        # free tip: the tip wrench vanishes, the base force carries the integrated load
        np.testing.assert_allclose(actual[2][-1], np.zeros(6), atol=1e-9)
        cpp.remove_gaussians()
        np.testing.assert_array_equal(cpp.external_gauss_forces(0.1), np.zeros((3, 1)))

    def test_shooting_function_selection(self):
        """fwd_static_with_boundarys takes the enum, the name, or the shooting functions of pyctr.robots"""
        _, cpp = both(example_params())
        configure((cpp,), [0.3, -0.2], [0.0, 0.0])
        wrench = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        cpp.set_boundary_condition(["tip_wrench"], [wrench])
        results = [
            cpp.fwd_static_with_boundarys(np.zeros(8), f, 0.004)
            for f in (pyctr_cpp.ShootingFunction.force, "force", shooting_function_force, pyctr_cpp.shooting_function_force)
        ]
        for r in results[1:]:
            self.assert_results(results[0], r, scale=1e-3)
        residual = pyctr_cpp.shooting_function_force(np.hstack([results[0][2][0], results[0][4][0]]), cpp, 0.004)
        self.assertLess(np.linalg.norm(residual), 1e-8)
        np.testing.assert_array_equal(cpp.boundary_values["tip_wrench"], wrench)
        cpp.remove_boundary_condition("tip_wrench")
        # KeyError as the boundary_values dict in Python
        with self.assertRaises(KeyError):
            cpp.remove_boundary_condition("tip_wrench")
        with self.assertRaises(KeyError):
            cpp.fwd_static_with_boundarys(np.zeros(8), "force", 0.004)
        with self.assertRaises(ValueError):
            cpp.fwd_static_with_boundarys(np.zeros(8), "unknown", 0.004)

        # tip position only: three of the six residuals are zero
        target = cpp.fwd_kinematic(0.004)[0][-1] + np.array([0.002, 0.0, 0.0])
        cpp.set_boundary_condition(["pL"], [target])
        p = cpp.fwd_static_with_boundarys(np.zeros(6), "tip_position", 0.004)[0]
        np.testing.assert_allclose(p[-1], target, atol=1e-8)
        cpp.remove_all_boundary_condition()
        self.assertEqual(cpp.boundary_values, {})

    def test_in_place_attribute_writes(self):
        """alphas, betas, p_init and R_init are numpy views as the attributes in Python: in-place writes change
        the configuration"""
        # own arrays: the Python defaults of R_init and p_init are shared by all Python robots
        py, cpp = both(example_params(), R_init=np.eye(3), p_init=np.zeros((3, 1)))
        self.assertEqual(cpp.p_init.shape, (3, 1))
        for r in (py, cpp):
            r.alphas[1] = 0.7
            r.p_init[2] = 0.05
            r.R_init[:] = Rz(0.3)
        np.testing.assert_array_equal(cpp.alphas, [0.0, 0.7])
        self.assert_results(py.fwd_kinematic(0.004), cpp.fwd_kinematic(0.004))
        # In Python an in-place write to betas changes only the segment ends, not the curved/straight
        # classification of the rods (that reads params["beta"], set by translate). C++ uses betas for both,
        # so it equals translate.
        reference = CppRobot(example_params())
        reference.translate([0.0, 0.2])
        cpp.R_init[:], cpp.p_init[:] = np.eye(3), 0.0
        cpp.alphas[1] = 0.0
        cpp.betas[1] = 0.2
        self.assert_results(reference.fwd_kinematic(0.004), cpp.fwd_kinematic(0.004), scale=1e-3)

    def test_invalid_input_raises(self):
        """Wrong input raises ValueError instead of reading out of bounds (it crashed the process before)"""
        with self.assertRaises(ValueError):
            CppRobot([])
        _, cpp = both(three_tubes())
        with self.assertRaises(ValueError):  # before calc_forward there is no current segment
            cpp.calc_uxz(np.zeros(3), np.zeros(3), 0.1, 0.0)
        for step in (0.0, -0.01, np.nan, np.inf, 1e-12):
            with self.assertRaises(ValueError, msg=str(step)):
                cpp.fwd_static(np.zeros(6), step)
        cpp.fwd_kinematic(0.01)
        with self.assertRaises(ValueError):
            cpp.cosserate_rod_ode(0.1, np.zeros(3))
        with self.assertRaises(ValueError):
            cpp.calc_uxz(np.zeros(0), np.zeros(3), 0.1, 0.0)
        with self.assertRaises(ValueError):
            cpp.calc_us(np.linspace(0, 0.2, 1000), np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            cpp.calc_us([0.1], np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 0)))

    # Ideas of tests/test_theory.py against the C++ class
    def test_stiffness_weighted_bending_average(self):
        base = {**NYLON, "L": 0.15, "straight_length": 0.05, "r_inner": 0.0}
        params = [{**base, "kappa": 6.0, "r_outer": 1.25e-3}, {**base, "kappa": 12.0, "r_outer": 2.0e-3}]
        _, cpp = both(params)
        configure((cpp,), [0.0, 0.0], [0.0, 0.0])
        us = cpp.fwd_static(np.zeros(6), 0.002)[3]
        EI = [t["Kbt"][0, 0] for t in cpp.tubes]
        expected = (EI[0] * 6.0 + EI[1] * 12.0) / (EI[0] + EI[1])
        self.assertAlmostEqual(us[-1, 1], expected, places=10)

    def test_opposed_identical_tubes_cancel(self):
        params = example_params()[0]
        _, cpp = both([params, params])
        configure((cpp,), [0.0, np.pi], [0.0, 0.0])
        us = cpp.fwd_static(np.zeros(6), 0.002)[3]
        self.assertLess(np.max(np.linalg.norm(us[:, :2], axis=1)), 1e-6)

    def test_rigid_rotation_of_base(self):
        _, cpp = both(example_params())
        configure((cpp,), [0.0, np.pi], [0.0, 0.0])
        p0 = cpp.fwd_static(np.zeros(6), 0.004)[0]
        cpp.rotate([0.7, 0.7 + np.pi])
        pa = cpp.fwd_static(np.zeros(6), 0.004)[0]
        self.assertLess(np.max(np.abs((Rz(0.7) @ p0.T).T - pa)), 1e-8)

    def test_unloaded_push_end_is_forward_kinematics(self):
        _, cpp = both(example_params())
        configure((cpp,), [0.0, 0.0], [0.0, 0.0])
        p_fwd = cpp.fwd_kinematic(0.004)[0]
        p_push = cpp.push_end(np.zeros(6), 0.004)[0]
        self.assertLess(np.max(np.abs(p_fwd - p_push)), 1e-9)


if __name__ == "__main__":
    unittest.main()
