"""
pyctr_cpp.DynamicConcentricTubeRobot against pyctr.cosserat_rod_dynamics.DynamicConcentricTubeRobot on the same inputs.

Tolerances: both implementations solve the same discretized equations with Newton's method until the max residual
is below 1e-9, but with differently computed Jacobians, so they stop at slightly different iterates. Newton converges
quadratically here (final residuals are about 1e-12), the remaining differences are about 1e-12 m in the positions.
POS = 1e-9 m (1e-8 of the robot length) leaves a margin for the differences to accumulate over many time steps.
"""

import copy
import os
import unittest
import numpy as np
from scipy.integrate import solve_bvp
from pyctr.cosserat_rod import CurvedCosseratRod
from pyctr.cosserat_rod_dynamics import DynamicConcentricTubeRobot as PyRobot
from pyctr.utils import Rz
from pyctr.yaml_to_model import load_continous_ctcr_model

try:
    from pyctr_cpp import DynamicConcentricTubeRobot as CppRobot
except ImportError:
    CppRobot = None

try:
    from pydrake.geometry import Box, GeometryInstance, ProximityProperties, SceneGraph, Sphere
    from pydrake.math import RigidTransform
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, CoulombFriction
    from pydrake.multibody.tree import SpatialInertia
    from pydrake.systems.analysis import Simulator
    from pydrake.systems.framework import DiagramBuilder
    from pydrake.systems.primitives import TrajectorySource
    from pydrake.trajectories import PiecewisePolynomial
    from pyctr.drake_system import ConcentricTubeRobotSystem
except ImportError:
    SceneGraph = None

EXAMPLE_ROBOT = os.path.join(os.path.dirname(__file__), "../example_robots/ctr_robot.yaml")
NYLON = {"E": 339889091.1107947, "G": 97898219.42862949, "rho": 8000, "g": 0}
POS = 1e-9


def example_robot(cls):
    return load_continous_ctcr_model(EXAMPLE_ROBOT, cls, dt=2e-3)


def two_tubes(cls, L, ds, kappa, dt=2e-3, **kwargs):
    tubes = [
        CurvedCosseratRod({**NYLON, "L": L, "s": ds, "kappa": kappa, "r_inner": ri, "r_outer": ro})
        for ri, ro in ((0.5e-3, 1.25e-3), (1.5e-3, 2.5e-3))
    ]
    return cls(tubes, dt=dt, **kwargs)


def both(make):
    return make(PyRobot), make(CppRobot)


def period(x, dt):
    x = np.asarray(x)
    i = np.flatnonzero(np.sign(x[:-1]) != np.sign(x[1:]))
    return 2 * np.mean(np.diff((i + x[i] / (x[i] - x[i + 1])) * dt))


@unittest.skipIf(CppRobot is None, "pyctr_cpp with DynamicConcentricTubeRobot is not built")
class TestCppDynamicConcentricTubeRobot(unittest.TestCase):
    def assert_same(self, out_py, out_cpp, atol=POS):
        for a, b in zip(out_py, out_cpp):
            self.assertEqual(a.shape, b.shape)
        np.testing.assert_allclose(out_cpp[0], out_py[0], rtol=0, atol=atol)  # positions
        np.testing.assert_allclose(out_cpp[1], out_py[1], rtol=0, atol=atol * 100)  # R, 1e-7 per m of arc
        np.testing.assert_allclose(out_cpp[2], out_py[2], rtol=1e-6, atol=1e-9)  # wrenches (N, N m)

    def compare_steps(self, make, betas, start=(0, 0)):
        """init_static at the betas start, then one step per betas, alphas [0, 1] and a tip force"""
        a, b = both(make)
        for r in (a, b):
            r.rotate([0, 1.0])
            r.translate(start)
        self.assert_same(a.init_static(), b.init_static())
        np.testing.assert_array_equal(b.s, a.s)
        for beta in betas:
            for r in (a, b):
                r.translate(beta)
            self.assert_same(a.step([0, 0.01, 0, 0, 0, 0]), b.step([0, 0.01, 0, 0, 0, 0]))
            np.testing.assert_array_equal(b.s, a.s)  # same largest remainder distribution
            np.testing.assert_array_equal(b.radius, a.radius)
        x_py, x = a.get_state(), b.get_state()
        self.assertEqual(x.shape, x_py.shape)
        np.testing.assert_array_equal(x[-2:], x_py[-2:])
        np.testing.assert_allclose(x, x_py, rtol=1e-6, atol=1e-6)
        return a, b

    def test_grid_and_static_shapes(self):
        # several alphas, betas and tip wrenches on the example robot (tubes ending on the same node: torsion test)
        cases = [
            ([0, 0], [0, 0], np.zeros(6)),
            ([0.3, 2.0], [0, 0], [0.01, -0.02, 0, 0, 0, 1e-4]),
            ([1.0, -1.0], [0.1, 0.25], [0, 0, 0.05, 1e-4, 0, 0]),
            ([0, np.pi], [0, 1 - 0.095 / 0.124], np.zeros(6)),  # outer tube ends about where the inner one curves
            ([0.5, 0.2], [0.6, 0.45], [0, 0, 0, 0, 1e-4, 0]),  # both tubes curved from the base
        ]
        for alphas, betas, wrench in cases:
            with self.subTest(alphas=alphas, betas=betas):
                a, b = both(lambda cls: load_continous_ctcr_model(EXAMPLE_ROBOT, cls))
                for r in (a, b):
                    r.rotate(alphas)
                    r.translate(betas)
                out_py, out_cpp = a.init_static(wrench), b.init_static(wrench)
                np.testing.assert_array_equal(b.s, a.s)
                np.testing.assert_array_equal(b.radius, a.radius)
                self.assertEqual(b.y.shape, a.y.shape)
                self.assert_same(out_py, out_cpp)
                np.testing.assert_allclose(b.y, a.y, rtol=1e-6, atol=1e-8)

    def test_static_torsion_against_analytic(self):
        L, kappa = 0.09, 11.636
        for alpha in (0.5, 2.5, 3.1):
            with self.subTest(alpha=alpha):
                a, b = both(lambda cls: two_tubes(cls, L, 1e-3, kappa))
                EI, GJ = a._EI, a._GJ
                c = EI[0] * EI[1] / EI.sum() * kappa**2 * (1 / GJ[0] + 1 / GJ[1])
                s = np.linspace(0, L, 50)
                ref = solve_bvp(
                    lambda s, y: np.vstack([y[1], c * np.sin(y[0])]),
                    lambda ya, yb: np.array([ya[0] - alpha, yb[1]]),
                    s,
                    np.vstack([np.full(50, alpha), np.zeros(50)]),
                )
                for r in (a, b):
                    r.rotate([0, alpha])
                self.assert_same(a.init_static(), b.init_static())
                np.testing.assert_allclose(b.y[-1, 23 + 2 + 1], ref.sol(L)[0], rtol=1e-3)

    def test_static_example_robot_against_arcs(self):
        a, b = both(lambda cls: load_continous_ctcr_model(EXAMPLE_ROBOT, cls))
        out_py, (p, _, _) = a.init_static(), b.init_static()
        self.assertEqual(len(b.s), 186)
        EI, kappa = a._EI, 11.636
        s = np.linspace(0, 0.185, 100001)
        u = np.select([s < 0.034, s < 0.095], [0, EI[1] * kappa / EI.sum()], kappa)
        phi = np.concatenate([[0], np.cumsum(0.5 * (u[1:] + u[:-1]) * np.diff(s))])
        tip = [np.trapezoid(np.sin(phi), s), 0, np.trapezoid(np.cos(phi), s)]
        np.testing.assert_allclose(p[-1], tip, atol=1e-5)
        np.testing.assert_allclose(p, out_py[0], atol=POS)

    def test_example_robot_rotating_steps(self):
        # 186 nodes, dt = 2e-3, both tubes rotate while a tip force acts
        a, b = both(lambda cls: load_continous_ctcr_model(EXAMPLE_ROBOT, cls, dt=2e-3))
        for r in (a, b):
            r.rotate([0, 1.0])
        self.assert_same(a.init_static([0.01, 0, 0, 0, 0, 0]), b.init_static([0.01, 0, 0, 0, 0, 0]))
        for i in range(15):
            for r in (a, b):
                r.rotate([0.5 * np.sin(0.2 * i), 1.0 + 0.1 * i])
            self.assert_same(a.step([0, 0.01, 0, 0, 0, 0]), b.step([0, 0.01, 0, 0, 0, 0]))
        np.testing.assert_allclose(b.get_state(), a.get_state(), rtol=1e-6, atol=1e-6)

    def test_bending_and_torsion_periods(self):
        L = 0.1
        a, b = both(lambda cls: two_tubes(cls, L, 5e-3, 0.0, dt=1e-3))
        EI, rhoA, F = a._EI.sum(), a._rhoA, 0.01
        self.assert_same(a.init_static([F, 0, 0, 0, 0, 0]), b.init_static([F, 0, 0, 0, 0, 0]))
        x = []
        for _ in range(100):
            out_py, out_cpp = a.step(), b.step()
            self.assert_same(out_py, out_cpp)
            x.append(out_cpp[0][-1, 0])
        np.testing.assert_allclose(period(x, 1e-3), 2 * np.pi / 1.8751**2 * np.sqrt(rhoA * L**4 / EI), rtol=1e-2)

        a, b = both(lambda cls: two_tubes(cls, L, 5e-3, 0.0, dt=5e-5))
        self.assert_same(a.init_static([0, 0, 0, 0, 0, 1e-4]), b.init_static([0, 0, 0, 0, 0, 1e-4]))
        twist = []
        for _ in range(80):
            out_py, out_cpp = a.step(), b.step()
            self.assert_same(out_py, out_cpp)
            twist.append(np.arctan2(out_cpp[1][-1, 3], out_cpp[1][-1, 0]))
        np.testing.assert_allclose(period(twist, 5e-5), 4 * L / np.sqrt(NYLON["G"] / NYLON["rho"]), rtol=1e-2)

    def test_slow_rotation_reaches_static_shape(self):
        a, b = both(lambda cls: load_continous_ctcr_model(EXAMPLE_ROBOT, cls, dt=1e-2))
        for r in (a, b):
            r.tubes[0][1].params["s"] = 5e-3  # read from the rod in init_static
        self.assert_same(a.init_static(), b.init_static())
        self.assertEqual(len(b.s), len(a.s))
        tips = []
        for i in range(150):
            for r in (a, b):
                r.rotate([0, np.pi / 2 * min(1, (i + 1) / 100)])
            out_py, out_cpp = a.step(), b.step()
            self.assert_same(out_py, out_cpp)
            tips.append(out_cpp[0][-1])
        p_static, _, _ = b.init_static()
        np.testing.assert_allclose(np.mean(tips[100:], axis=0), p_static[-1], atol=1e-3)

    def test_contact_with_wall(self):
        x_wall = -2.5e-3 - 0.8e-3
        for dissipation in (0.0, 2.0):
            with self.subTest(contact_dissipation=dissipation):
                a, b = both(lambda cls: two_tubes(cls, 0.1, 2.5e-3, 0.0, dt=1e-3, contact_dissipation=dissipation))
                a.init_static()
                b.init_static()
                intervals = len(b.s) - 1
                contacts = (np.tile([1.0, 0, 0], (intervals, 1)), np.tile([x_wall, 0, 0], (intervals, 1)))
                self.assert_same(a.init_static([0.05, 0, 0, 0, 0, 0], contacts), b.init_static([0.05, 0, 0, 0, 0, 0], contacts))
                depth, force = [], []
                for _ in range(30):
                    out_py, out_cpp = a.step(np.zeros(6), contacts), b.step(np.zeros(6), contacts)
                    self.assert_same(out_py, out_cpp)
                    # force = stiffness * depth * length: 1e6 N/m^2 * 1e-9 m * 2.5e-3 m
                    np.testing.assert_allclose(b.contact_force, a.contact_force, rtol=0, atol=1e-5)
                    force.append(np.abs(b.contact_force).max())
                    p = out_cpp[0]
                    depth.append((b.radius - (0.5 * (p[1:, 0] + p[:-1, 0]) - x_wall)).max())
                self.assertGreater(max(depth), 0)
                self.assertLess(max(depth), 1e-4)
                self.assertGreater(max(force), 0)

    def test_moving_base(self):
        W = 3.0
        a, b = both(lambda cls: two_tubes(cls, 0.1, 5e-3, 11.636, dt=1e-3))
        for r in (a, b):
            r.init_static()
        rotated = two_tubes(CppRobot, 0.1, 5e-3, 11.636, dt=1e-3)
        rotated.init_static()
        for i in range(1, 61):
            for r in (a, b):
                r.R_init, r.base_velocity = Rz(W * i * r.dt), (np.zeros(3), np.array([0, 0, W]))
            rotated.rotate([W * i * rotated.dt] * 2)
            out_py, out_cpp = a.step(), b.step()
            self.assert_same(out_py, out_cpp)
            np.testing.assert_allclose(out_cpp[0], rotated.step()[0], atol=1e-4)

        # base accelerating up with g along a horizontal robot, against gravity switched on with a fixed base (Python)
        R0 = np.array([[0, 0, 1.0], [0, 1, 0], [-1, 0, 0]])
        a, b = both(lambda cls: two_tubes(cls, 0.1, 5e-3, 0.0, dt=1e-3, R_init=R0))
        gravity = two_tubes(PyRobot, 0.1, 5e-3, 0.0, dt=1e-3, R_init=R0)
        for r in (a, b, gravity):
            r.init_static()
        gravity._f = gravity._rhoA * np.array([0, 0, -9.81])
        for i in range(1, 61):
            t = i * b.dt
            for r in (a, b):
                r.p_init, r.base_velocity = np.array([[0, 0, 0.5 * 9.81 * t**2]]).T, (np.array([0, 0, 9.81 * t]), np.zeros(3))
            out_py, out_cpp = a.step(), b.step()
            self.assert_same(out_py, out_cpp)
            np.testing.assert_allclose(out_cpp[0] - b.p_init.ravel(), gravity.step()[0], atol=1e-4)

    def test_interface(self):
        b = two_tubes(CppRobot, 0.1, 5e-3, 11.636)
        self.assertFalse(hasattr(b, "s"))  # ConcentricTubeRobotSystem relies on this
        self.assertEqual(len(b.tubes), 2)
        self.assertEqual(b.p_init.shape, (3, 1))
        np.testing.assert_array_equal(b.R_init, np.eye(3))
        b.alphas[1] = 0.5  # numpy view, as the attribute in Python
        np.testing.assert_array_equal(b.alphas.copy(), [0, 0.5])
        with self.assertRaises(Exception):
            b.translate([0.5, 0])  # inner tube would end inside the outer tube
        b.init_static()
        self.assertEqual(b.y.shape, (len(b.s), 23 + 3 * 2))
        self.assertEqual(b.radius.shape, (len(b.s) - 1,))

        # state round trip and deepcopy give the same steps
        c = copy.deepcopy(b)
        self.assertIsNot(c.tubes[0][1], b.tubes[0][1])
        x = b.get_state()
        p1 = b.step([0.01, 0, 0, 0, 0, 0])[0]
        b.set_state(x)
        np.testing.assert_array_equal(b.step([0.01, 0, 0, 0, 0, 0])[0], p1)
        np.testing.assert_array_equal(c.step([0.01, 0, 0, 0, 0, 0])[0], p1)
        a = two_tubes(PyRobot, 0.1, 5e-3, 11.636)
        a.rotate([0, 0.5])
        a.init_static()
        self.assertEqual(a.get_state().shape, x.shape)

        with self.assertRaises(ValueError):
            b.set_state(x[:-1])
        # an outer tube longer than the inner one (no translate checks it): its end is the nearest node, the last one
        a, b = both(lambda cls: cls([CurvedCosseratRod({**NYLON, "L": L, "s": 5e-3}) for L in (0.05, 0.1)]))
        self.assert_same(a.init_static([0.01, 0, 0, 0, 0, 0]), b.init_static([0.01, 0, 0, 0, 0, 0]))
        self.assert_same(a.step(), b.step())

    def test_insertion_sequences(self):
        # betas between steps: slow and fast, both tubes and single tubes, retraction and extension. Outer tube:
        # 0.2339 ends where the inner tube curves, above 0.274 its curved part starts below the base plate.
        # Inner tube: its curve passes the end of the outer tube.
        edge = 1 - 0.095 / 0.124
        sequences = {
            "slow, both tubes": [[0.002 * i, 0.003 * i] for i in range(1, 16)],
            "fast, outer tube": [[0, b] for b in (0.1, edge, 0.3, 0.4, 0.3, edge, 0.1, 0.0)],
            "inner tube": [[b, 0] for b in (0.05, 0.1, 0.2, 0.3, 0.2, 0.0)],
            "both tubes, mixed": [[0.1, 0.2], [0.2, 0.2], [0.2, 0.35], [0.0, 0.1], [0.0, 0.1], [0.3, 0.0]],
        }
        for name, betas in sequences.items():
            with self.subTest(name):
                self.compare_steps(example_robot, betas)

    def test_tiny_beta_change(self):
        # init_static builds the grid of remesh, so a tiny change of the betas moves the nodes only a little
        for beta in ([0, 1e-12], [1e-13, 0], [1e-12, 1e-12], [0, 1e-10], [0, -1e-12]):
            with self.subTest(beta=beta):
                b = self.compare_steps(example_robot, [beta, beta])[1]
                s = example_robot(CppRobot)
                s.init_static()
                np.testing.assert_allclose(b.s, s.s, rtol=0, atol=1e-9)

    def test_aligned_tube_tips(self):
        # b0 = 1 - L1 (1 - b1) / L0: in floating point the outer tube ends up to 3e-17 m after the inner one for
        # b1 = 0.05 and 0.3, the tolerance of translate accepts that
        L0, L1 = 0.185, 0.124
        for b1 in (0.0, 0.05, 0.3):
            aligned = [1 - L1 * (1 - b1) / L0, b1]
            with self.subTest(b1=b1):
                self.compare_steps(example_robot, [aligned, aligned, [0, b1]])
                self.compare_steps(example_robot, [aligned, [aligned[0] - 0.01, b1]], start=aligned)

    def test_outer_tube_retracted_and_extended(self):
        # no history of u_z and gamma of the outer tube while it is retracted: it starts with 0
        self.compare_steps(example_robot, [[0, 1.0], [0, 1.0], [0, 0.5], [0, 0.2]])
        self.compare_steps(example_robot, [[0, 0.5], [0, 1.0], [0.1, 0.3]], start=[0, 1.0])
        self.compare_steps(lambda cls: two_tubes(cls, 0.1, 5e-3, 11.636), [[0, 1.0], [0, 0.5]])

    def test_borders_close_together(self):
        # end of the outer tube at the start of the curve of the inner tube (0.095 m) +- eps: borders closer than
        # 1e-9 m are merged (with 2e-9 to 1e-8 m Newton does not reach the residual 1e-9, in Python too)
        for eps in (5e-13, -5e-13, 5e-10, -5e-10):
            b1 = 1 - (0.095 + eps) / 0.124
            with self.subTest(eps=eps):
                self.compare_steps(example_robot, [[0, b1], [0, b1], [0, 0.1]])
                self.compare_steps(example_robot, [[0, 0.1], [0, b1]], start=[0, b1])
        # tubes of equal length, the outer one ends 1e-12 m before the inner one
        self.compare_steps(lambda cls: two_tubes(cls, 0.1, 5e-3, 11.636), [[0, 1e-11], [0, 0.2]], start=[0, 1e-11])

    def test_set_state(self):
        a, b = both(example_robot)
        for r in (a, b):
            r.init_static([0.01, 0, 0, 0, 0, 0])
        n = len(b.get_state())
        for r in (a, b):
            with self.assertRaisesRegex(ValueError, f"state has {n + 1} values, this robot needs {n}"):
                r.set_state(np.zeros(n + 1))
        # after translate: the betas of the state are set again, the grid stays
        for r in (a, b):
            x = r.get_state()
            r.translate([0.1, 0.3])
            r.set_state(x)
            np.testing.assert_array_equal(r.betas, [0, 0])
            np.testing.assert_array_equal(r.get_state(), x)
            r.translate([0.1, 0.3])
        self.assert_same(a.step(), b.step())
        x_py, x = a.get_state(), b.get_state()
        for r in (a, b):
            r.translate([0.2, 0.2])
        a.set_state(x_py)
        b.set_state(x)
        np.testing.assert_array_equal(b.betas, [0.1, 0.3])
        np.testing.assert_array_equal(b.s, a.s)
        self.assert_same(a.step(), b.step())
        # invalid betas in the state: nothing changes
        bad = b.get_state()
        bad[-2:] = [1.0, 1.0]
        for r in (a, b):
            x = r.get_state()
            with self.assertRaises(Exception):
                r.set_state(bad)
            np.testing.assert_array_equal(r.get_state(), x)

    def test_invalid_betas(self):
        L0, L1 = 0.185, 0.124
        aligned = 1 - L1 * 0.95 / L0
        cases = [
            ([1.0, 1.0], False),  # the inner tube must stick out
            ([1.001, 1.0], False),
            ([0.5, 0.0], False),
            ([aligned, 0.05], True),
            ([aligned + 1e-9, 0.05], False),
            ([-0.1, 0.0], True),
            ([0.0, 1.0], True),
        ]
        a, b = both(example_robot)
        for betas, valid in cases:
            with self.subTest(betas=betas):
                results = []
                for r in (a, b):
                    r.translate([0, 0])
                    try:
                        r.translate(betas)
                        results.append(True)
                    except Exception:
                        results.append(False)
                        np.testing.assert_array_equal(r.betas, [0, 0])
                self.assertEqual(results, [valid, valid])

    def test_state_with_betas(self):
        a, b = both(lambda cls: load_continous_ctcr_model(EXAMPLE_ROBOT, cls, dt=2e-3))
        for r in (a, b):
            r.init_static([0.01, 0, 0, 0, 0, 0])
        x = b.get_state()
        np.testing.assert_array_equal(x[-2:], [0, 0])
        for r in (a, b):
            r.translate([0.1, 0.3])
        self.assert_same(a.step(), b.step())
        y = b.get_state()
        np.testing.assert_array_equal(y[-2:], [0.1, 0.3])
        p = b.step()[0]
        b.set_state(x)  # back to the betas and the grid of x
        np.testing.assert_array_equal(b.betas, [0, 0])
        np.testing.assert_array_equal(b.get_state(), x)
        b.set_state(y)
        np.testing.assert_array_equal(b.step()[0], p)
        # the state of the Python robot gives the same step in the C++ robot
        a.translate([0.2, 0.2])
        a.step()
        b.set_state(a.get_state())
        np.testing.assert_array_equal(b.betas, [0.2, 0.2])
        np.testing.assert_array_equal(b.s, a.s)
        self.assert_same(a.step(), b.step())

    def test_remesh_errors(self):
        # one interval (s of tube 1 = L) but two segments after the translation; an outer tube without history
        a, b = both(lambda cls: two_tubes(cls, 0.1, 0.1, 11.636))
        for r in (a, b):
            r.init_static()
            self.assertEqual(len(r.s), 2)
            r.translate([0, 0.5])
            with self.assertRaisesRegex(ValueError, "1 intervals are not enough for 2 segments"):
                r.step()
        with self.assertRaises(RuntimeError):  # Python: AttributeError
            two_tubes(CppRobot, 0.1, 5e-3, 11.636).remesh()

    def test_failed_init_static(self):
        # no history after a failed init_static: step and get_state raise (Python: AttributeError on the first call,
        # on a later call it would continue with the history of the old configuration)
        for later in (False, True):
            b = two_tubes(CppRobot, 0.1, 5e-3, 11.636)
            if later:
                b.init_static()
                b.translate([0.0, 0.5])
            with self.assertRaisesRegex(RuntimeError, "max residual nan"):  # as Python
                b.init_static([np.nan, 0, 0, 0, 0, 0])
            with self.assertRaises(RuntimeError):
                b.step()
            with self.assertRaises(RuntimeError):
                b.get_state()
            b.init_static()  # recovers
            b.step()
        for ds in (0.0, -1e-3, np.nan):
            b = two_tubes(CppRobot, 0.1, ds, 11.636)
            with self.assertRaises(ValueError):  # Python: OverflowError or ValueError
                b.init_static()

    def test_in_place_attributes(self):
        # in-place writes into the numpy attributes change the model, as in Python. Own R_init and p_init, because the
        # Python defaults are shared default arguments.
        a, b = both(lambda cls: two_tubes(cls, 0.1, 5e-3, 11.636, R_init=np.eye(3), p_init=np.zeros((3, 1))))
        for r in (a, b):
            r.rotate([0.3, 1.0, 7.0])  # longer sequences are cut, as in Python
            r.R_init[:] = Rz(0.7)
            r.p_init[2, 0] = 0.01
            r.base_velocity[1][:] = [0, 0, 1.0]
        np.testing.assert_array_equal(b.alphas, [0.3, 1.0])
        np.testing.assert_allclose(b.init_static()[0], a.init_static()[0], rtol=0, atol=POS)
        np.testing.assert_allclose(b.step()[0], a.step()[0], rtol=0, atol=POS)


def simulate_free(cls):
    """test_drake_system.test_simulator_matches_direct_steps: positions of the system and of direct steps"""
    nylon = {**NYLON, "rho": 1140, "kappa": 11.636, "s": 5e-3}
    tubes = [
        CurvedCosseratRod({**nylon, "L": L, "straight_length": 0.03, "r_inner": ri, "r_outer": ro})
        for L, ri, ro in ((0.1, 0.5e-3, 1.25e-3), (0.06, 1.5e-3, 2.5e-3))
    ]
    robot = cls(tubes, dt=2e-3)
    robot.init_static()
    reference = copy.deepcopy(robot)
    builder = DiagramBuilder()
    scene_graph = builder.AddSystem(SceneGraph())
    ctr = builder.AddSystem(ConcentricTubeRobotSystem(robot, scene_graph))
    builder.Connect(ctr.GetOutputPort("geometry_pose"), scene_graph.get_source_pose_port(ctr.source_id))
    simulator = Simulator(builder.Build())
    context = ctr.GetMyMutableContextFromRoot(simulator.get_mutable_context())
    alphas, wrench = np.array([0.0, 1.0]), np.array([0, 0.01, 0, 0, 0, 0])
    ctr.GetInputPort("alphas").FixValue(context, alphas)
    ctr.GetInputPort("tip_wrench").FixValue(context, wrench)
    simulator.AdvanceTo(4.5 * robot.dt)
    reference.rotate(alphas)
    for _ in range(5):
        p_ref = reference.step(wrench)[0]
    return ctr.GetOutputPort("positions").Eval(context).reshape(-1, 3), p_ref


def straight_robot(cls):
    nylon = {**NYLON, "rho": 1140, "kappa": 0.0, "s": 5e-3}
    tubes = [
        CurvedCosseratRod({**nylon, "L": 0.1, "r_inner": ri, "r_outer": ro})
        for ri, ro in ((0.5e-3, 1.25e-3), (1.5e-3, 2.5e-3))
    ]
    return cls(tubes, dt=1e-3)


def connect_to_plant(builder, plant, scene_graph, ctr):
    builder.Connect(ctr.GetOutputPort("geometry_pose"), scene_graph.get_source_pose_port(ctr.source_id))
    builder.Connect(scene_graph.get_query_output_port(), ctr.GetInputPort("geometry_query"))
    builder.Connect(plant.get_body_poses_output_port(), ctr.GetInputPort("body_poses"))
    builder.Connect(plant.get_body_spatial_velocities_output_port(), ctr.GetInputPort("body_spatial_velocities"))
    builder.Connect(ctr.GetOutputPort("applied_spatial_force"), plant.get_applied_spatial_force_input_port())


def simulate_box_in_scene_graph(cls):
    """Positions per step of a straight robot pushed against a fixed box 1 mm away"""
    robot = straight_robot(cls)
    robot.init_static()
    face = 2.5e-3 + 1e-3
    builder = DiagramBuilder()
    scene_graph = builder.AddSystem(SceneGraph())
    source = scene_graph.RegisterSource("obstacles")
    box = scene_graph.RegisterAnchoredGeometry(
        source, GeometryInstance(RigidTransform([face + 0.01, 0, 0.05]), Box(0.02, 0.05, 0.2), "box")
    )
    scene_graph.AssignRole(source, box, ProximityProperties())
    ctr = builder.AddSystem(ConcentricTubeRobotSystem(robot, scene_graph))
    builder.Connect(ctr.GetOutputPort("geometry_pose"), scene_graph.get_source_pose_port(ctr.source_id))
    builder.Connect(scene_graph.get_query_output_port(), ctr.GetInputPort("geometry_query"))
    simulator = Simulator(builder.Build())
    context = ctr.GetMyMutableContextFromRoot(simulator.get_mutable_context())
    ctr.GetInputPort("tip_wrench").FixValue(context, [0.05, 0, 0, 0, 0, 0])
    positions = []
    for n in range(20):
        simulator.AdvanceTo((n + 0.5) * robot.dt)
        positions.append(ctr.GetOutputPort("positions").Eval(context).reshape(-1, 3))
    return np.array(positions), face


def simulate_mounted(cls):
    """Reaction on the base body and positions after 3 steps, base welded at p_WB, static tip force"""
    p_WB, F = np.array([0.1, 0.2, 0.3]), np.array([0.05, 0, 0])
    robot = straight_robot(cls)
    robot.p_init = p_WB.reshape(3, 1)
    p, _, _ = robot.init_static(np.r_[F, 0, 0, 0])
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=robot.dt)
    base = plant.AddRigidBody("base", SpatialInertia.SolidBoxWithMass(0.1, 0.02, 0.02, 0.02))
    plant.WeldFrames(plant.world_frame(), base.body_frame(), RigidTransform(p_WB))
    plant.Finalize()
    ctr = builder.AddSystem(ConcentricTubeRobotSystem(robot, scene_graph, plant, base))
    connect_to_plant(builder, plant, scene_graph, ctr)
    simulator = Simulator(builder.Build())
    context = ctr.GetMyMutableContextFromRoot(simulator.get_mutable_context())
    reaction = ctr.GetOutputPort("applied_spatial_force").Eval(context)[0]
    ctr.GetInputPort("tip_wrench").FixValue(context, np.r_[F, 0, 0, 0])
    simulator.AdvanceTo(2.5 * robot.dt)
    after = ctr.GetOutputPort("positions").Eval(context).reshape(-1, 3)
    return p, after, reaction.F_Bq_W.translational(), reaction.F_Bq_W.rotational(), np.cross(p[-1] - p_WB, F), F


def simulate_free_box(cls):
    """Positions per step, gaps to and final velocity of a free box pushed by the robot"""
    robot = straight_robot(cls)
    robot.init_static()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=robot.dt)
    plant.mutable_gravity_field().set_gravity_vector([0, 0, 0])
    base = plant.AddRigidBody("base", SpatialInertia.SolidBoxWithMass(0.1, 0.02, 0.02, 0.02))
    plant.WeldFrames(plant.world_frame(), base.body_frame())
    box = plant.AddRigidBody("box", SpatialInertia.SolidBoxWithMass(0.005, 0.02, 0.02, 0.02))
    geometry = plant.RegisterCollisionGeometry(box, RigidTransform(), Box(0.02, 0.02, 0.02), "box", CoulombFriction())
    plant.Finalize()
    ctr = builder.AddSystem(ConcentricTubeRobotSystem(robot, scene_graph, plant, base))
    connect_to_plant(builder, plant, scene_graph, ctr)
    simulator = Simulator(builder.Build())
    root = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(root)
    plant.SetFreeBodyPose(plant_context, box, RigidTransform([2.5e-3 + 1e-3 + 0.01, 0, 0.09]))
    context = ctr.GetMyMutableContextFromRoot(root)
    ctr.GetInputPort("tip_wrench").FixValue(context, [0.05, 0, 0, 0, 0, 0])
    positions, gaps = [], []
    for n in range(20):
        simulator.AdvanceTo((n + 0.5) * robot.dt)
        positions.append(ctr.GetOutputPort("positions").Eval(context).reshape(-1, 3))
        query = scene_graph.get_query_output_port().Eval(scene_graph.GetMyContextFromRoot(root))
        gaps.append(min(query.ComputeSignedDistancePairClosestPoints(g, geometry).distance for g in ctr._piece))
    velocity = plant.EvalBodySpatialVelocityInWorld(plant_context, box).translational()
    return np.array(positions), np.array(gaps), velocity


def simulate_insertion(cls):
    """test_drake_system.test_insertion_with_contact: positions and contact forces per step while the outer tube is
    retracted next to a fixed sphere (the system calls remesh in its update)"""
    robot = straight_robot(cls)
    robot.init_static()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=robot.dt)
    base = plant.AddRigidBody("base", SpatialInertia.SolidBoxWithMass(0.1, 0.02, 0.02, 0.02))
    plant.WeldFrames(plant.world_frame(), base.body_frame())
    plant.RegisterCollisionGeometry(
        plant.world_body(), RigidTransform([2.5e-3 + 5e-3 + 0.3e-3, 0, 0.04]), Sphere(5e-3), "sphere", CoulombFriction()
    )
    plant.Finalize()
    ctr = builder.AddSystem(ConcentricTubeRobotSystem(robot, scene_graph, plant, base))
    connect_to_plant(builder, plant, scene_graph, ctr)
    betas = builder.AddSystem(TrajectorySource(PiecewisePolynomial.FirstOrderHold([0, 0.03], [[0, 0], [0, 0.5]])))
    builder.Connect(betas.get_output_port(), ctr.GetInputPort("betas"))
    simulator = Simulator(builder.Build())
    context = ctr.GetMyMutableContextFromRoot(simulator.get_mutable_context())
    ctr.GetInputPort("tip_wrench").FixValue(context, [0.05, 0, 0, 0, 0, 0])
    positions, forces = [], []
    for n in range(40):
        simulator.AdvanceTo((n + 0.5) * robot.dt)
        positions.append(ctr.GetOutputPort("positions").Eval(context).reshape(-1, 3))
        forces.append(robot.contact_force)
    return np.array(positions), np.array(forces), robot.betas.copy()


@unittest.skipIf(CppRobot is None or SceneGraph is None, "pyctr_cpp or drake is not available")
class TestCppRobotInDrakeSystem(unittest.TestCase):
    """The diagrams of test_drake_system.py with the C++ robot, positions compared with the Python robot"""

    def test_simulator_matches_direct_steps(self):
        p, p_ref = simulate_free(CppRobot)
        np.testing.assert_allclose(p, p_ref, atol=1e-12)  # same C++ robot, the same computations
        np.testing.assert_allclose(p, simulate_free(PyRobot)[0], atol=POS)

    def test_contact_with_box_in_scene_graph(self):
        positions, face = simulate_box_in_scene_graph(CppRobot)
        depth = (positions[:, :, 0] + 2.5e-3 - face).max(axis=1)
        self.assertGreater(depth.max(), 0)
        self.assertLess(depth.max(), 1e-4)
        np.testing.assert_allclose(positions, simulate_box_in_scene_graph(PyRobot)[0], atol=POS)

    def test_mounted_on_plant_body(self):
        p, after, force, moment, expected_moment, F = simulate_mounted(CppRobot)
        np.testing.assert_allclose(force, F, atol=1e-9)
        np.testing.assert_allclose(moment, expected_moment, atol=1e-9)
        np.testing.assert_allclose(after, p, atol=1e-9)
        np.testing.assert_allclose(after, simulate_mounted(PyRobot)[1], atol=POS)

    def test_pushes_free_body_of_plant(self):
        positions, gaps, velocity = simulate_free_box(CppRobot)
        self.assertLess(gaps.min(), 0)
        self.assertGreater(gaps.min(), -1e-4)
        self.assertGreater(velocity[0], 0)
        positions_py, gaps_py, velocity_py = simulate_free_box(PyRobot)
        np.testing.assert_allclose(positions, positions_py, atol=POS)
        np.testing.assert_allclose(gaps, gaps_py, atol=POS)
        np.testing.assert_allclose(velocity, velocity_py, rtol=1e-6, atol=1e-9)

    def test_insertion_with_contact(self):
        positions, forces, betas = simulate_insertion(CppRobot)
        np.testing.assert_allclose(betas, [0, 0.5])
        self.assertGreater(np.abs(forces).max(), 0)
        positions_py, forces_py, _ = simulate_insertion(PyRobot)
        np.testing.assert_allclose(positions, positions_py, atol=POS)
        np.testing.assert_allclose(forces, forces_py, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
