import os
import unittest
import numpy as np
from scipy.integrate import solve_bvp
from pyctr.cosserat_rod import CurvedCosseratRod
from pyctr.cosserat_rod_dynamics import DynamicCosseratRod, DynamicConcentricTubeRobot
from pyctr.utils import Rz
from pyctr.yaml_to_model import load_continous_ctcr_model

EXAMPLE_ROBOT = os.path.join(os.path.dirname(__file__), "../example_robots/ctr_robot.yaml")
NYLON = {"E": 339889091.1107947, "G": 97898219.42862949, "rho": 8000, "g": 0}


def two_tubes(L, ds, kappa, dt=2e-3):
    tubes = [
        CurvedCosseratRod({**NYLON, "L": L, "s": ds, "kappa": kappa, "r_inner": ri, "r_outer": ro})
        for ri, ro in ((0.5e-3, 1.25e-3), (1.5e-3, 2.5e-3))
    ]
    return DynamicConcentricTubeRobot(tubes, dt=dt)


def period(x, dt):
    """Twice the mean time between zero crossings (linear interpolation)"""
    x = np.asarray(x)
    i = np.flatnonzero(np.sign(x[:-1]) != np.sign(x[1:]))
    return 2 * np.mean(np.diff((i + x[i] / (x[i] - x[i + 1])) * dt))


class TestDynamicCosseratRod(unittest.TestCase):
    def test_cantilever_against_beam_theory(self):
        # steel cantilever from Till's 02_Dynamics/00_Cosserat_Rod_PDE example, along z, no gravity
        params = {"E": 200e9, "G": 80e9, "r_outer": 1e-3, "rho": 8000, "g": 0, "L": 0.5, "s": 0.0125}
        rod = DynamicCosseratRod(params, dt=2e-3, bdf_alpha=-0.2)
        rod.set_initial_conditions(np.array([[0, 0, 0]]), np.eye(3))
        EI, rhoA, L, F = rod.params["Kbt"][0, 0], rod.params["rho"] * rod.params["A"], 0.5, 0.02

        # static tip deflection: F L^3 / (3 EI)
        p, _, _ = rod.init_static([F, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(p[-1, 0], F * L**3 / (3 * EI), rtol=0.03)

        # release the load: first bending mode f1 = 1.8751^2 / (2 pi) * sqrt(EI / (rhoA L^4))
        x = [rod.step()[0][-1, 0] for _ in range(200)]
        t = np.flatnonzero(np.diff(np.sign(x)) > 0) * rod.dt  # upward zero crossings
        np.testing.assert_allclose(
            np.mean(np.diff(t)), 2 * np.pi / 1.8751**2 * np.sqrt(rhoA * L**4 / EI), rtol=0.03
        )


class TestDynamicConcentricTubeRobot(unittest.TestCase):
    def test_static_torsion_against_analytic(self):
        # two fully curved tubes of equal length: theta'' = c sin(theta), theta(0) = alpha, theta'(L) = 0
        L, kappa, alpha = 0.09, 11.636, 2.5
        ctr = two_tubes(L, 1e-3, kappa)
        EI, GJ = ctr._EI, ctr._GJ
        c = EI[0] * EI[1] / EI.sum() * kappa**2 * (1 / GJ[0] + 1 / GJ[1])
        s = np.linspace(0, L, 50)
        ref = solve_bvp(
            lambda s, y: np.vstack([y[1], c * np.sin(y[0])]),
            lambda ya, yb: np.array([ya[0] - alpha, yb[1]]),
            s,
            np.vstack([np.full(50, alpha), np.zeros(50)]),
        )
        ctr.rotate([0, alpha])
        ctr.init_static()
        np.testing.assert_allclose(ctr.y[-1, 23 + 2 + 1], ref.sol(L)[0], rtol=1e-3)

    def test_static_example_robot_against_arcs(self):
        # alphas = 0: planar circular arcs with the combined curvature of each segment
        ctr = load_continous_ctcr_model(EXAMPLE_ROBOT, DynamicConcentricTubeRobot)
        p, _, _ = ctr.init_static()
        EI, kappa = ctr._EI, 11.636
        s = np.linspace(0, 0.185, 100001)
        u = np.select([s < 0.034, s < 0.095], [0, EI[1] * kappa / EI.sum()], kappa)
        phi = np.concatenate([[0], np.cumsum(0.5 * (u[1:] + u[:-1]) * np.diff(s))])
        tip = [np.trapezoid(np.sin(phi), s), 0, np.trapezoid(np.cos(phi), s)]
        np.testing.assert_allclose(p[-1], tip, atol=1e-5)

    def test_bending_and_torsion_periods(self):
        L = 0.1
        # bending: release a tip force, first mode with summed EI and rhoA
        ctr = two_tubes(L, 5e-3, 0.0, dt=1e-3)
        EI, rhoA, F = ctr._EI.sum(), ctr._rhoA, 0.01
        p, _, _ = ctr.init_static([F, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(p[-1, 0], F * L**3 / (3 * EI), rtol=1e-2)
        x = [ctr.step()[0][-1, 0] for _ in range(100)]
        np.testing.assert_allclose(period(x, 1e-3), 2 * np.pi / 1.8751**2 * np.sqrt(rhoA * L**4 / EI), rtol=1e-2)

        # torsion: release a tip moment on the inner tube, clamped-free period 4 L / sqrt(G / rho)
        ctr = two_tubes(L, 5e-3, 0.0, dt=5e-5)
        ctr.init_static([0, 0, 0, 0, 0, 1e-4])
        twist = []
        for _ in range(80):
            _, R, _ = ctr.step()
            twist.append(np.arctan2(R[-1, 3], R[-1, 0]))
        np.testing.assert_allclose(period(twist, 5e-5), 4 * L / np.sqrt(NYLON["G"] / NYLON["rho"]), rtol=1e-2)

    def test_slow_rotation_reaches_static_shape(self):
        ctr = load_continous_ctcr_model(EXAMPLE_ROBOT, DynamicConcentricTubeRobot, dt=1e-2)
        ctr.tubes[0][1].params["s"] = 5e-3
        ctr.init_static()
        tips = []
        for i in range(150):  # outer tube to 90 deg within 1 s, hold 0.5 s
            ctr.rotate([0, np.pi / 2 * min(1, (i + 1) / 100)])
            tips.append(ctr.step()[0][-1])
        p_static, _, _ = ctr.init_static()
        np.testing.assert_allclose(np.mean(tips[100:], axis=0), p_static[-1], atol=1e-3)

    def test_contact_stops_robot_at_wall(self):
        # released from a tip force, the free robot swings to x = -1.6 mm; a wall touches its surface at x = -0.8 mm
        ctr = two_tubes(0.1, 2.5e-3, 0.0, dt=1e-3)
        ctr.init_static()
        x_wall = -2.5e-3 - 0.8e-3
        intervals = len(ctr.s) - 1
        contacts = (np.tile([1.0, 0, 0], (intervals, 1)), np.tile([x_wall, 0, 0], (intervals, 1)))
        ctr.init_static([0.05, 0, 0, 0, 0, 0], contacts)
        depth = []
        for _ in range(30):  # contact after about 20 ms (period 66 ms with rho = 8000)
            p = ctr.step(np.zeros(6), contacts)[0]
            depth.append((ctr.radius - (0.5 * (p[1:, 0] + p[:-1, 0]) - x_wall)).max())
        self.assertGreater(max(depth), 0)  # it touches the wall
        self.assertLess(max(depth), 1e-4)  # and does not go through

    def test_moving_base(self):
        # rotating the base about z at W equals rotating all tubes (tip moves about 5 mm)
        a, b = two_tubes(0.1, 5e-3, 11.636, dt=1e-3), two_tubes(0.1, 5e-3, 11.636, dt=1e-3)
        a.init_static()
        b.init_static()
        W = 3.0
        for i in range(1, 61):
            a.rotate([W * i * a.dt] * 2)
            b.R_init, b.base_velocity = Rz(W * i * b.dt), (np.zeros(3), np.array([0, 0, W]))
            np.testing.assert_allclose(a.step()[0], b.step()[0], atol=1e-4)

        # robot along x: the base accelerating up with g deforms it like gravity on a fixed base (about 0.7 mm)
        a, b = two_tubes(0.1, 5e-3, 0.0, dt=1e-3), two_tubes(0.1, 5e-3, 0.0, dt=1e-3)
        a.R_init = b.R_init = np.array([[0, 0, 1.0], [0, 1, 0], [-1, 0, 0]])
        a.init_static()
        b.init_static()
        b._f = b._rhoA * np.array([0, 0, -9.81])  # gravity switched on at t = 0, like the acceleration of a
        for i in range(1, 61):
            t = i * a.dt
            a.p_init, a.base_velocity = np.array([[0, 0, 0.5 * 9.81 * t**2]]).T, (np.array([0, 0, 9.81 * t]), np.zeros(3))
            np.testing.assert_allclose(a.step()[0] - a.p_init.ravel(), b.step()[0], atol=1e-4)


    def test_insertion(self):
        def robot():
            ctr = load_continous_ctcr_model(EXAMPLE_ROBOT, DynamicConcentricTubeRobot, dt=1e-2)
            ctr.tubes[0][1].params["s"] = 5e-3
            ctr.rotate([0, 0.8])
            return ctr

        # a tiny change of the betas does not move the robot (init_static and remesh give the same grid)
        a = robot()
        p0 = a.init_static()[0]
        a.translate([1e-9, 0])
        np.testing.assert_allclose(a.step()[0], p0, atol=1e-8)

        # slow insertion ends at the static shape of the new betas
        a = robot()
        a.init_static()
        for i in range(100):
            a.translate(np.array([0.3, 0.25]) * min(1, (i + 1) / 80))
            p = a.step()[0]
        b = robot()
        b.translate([0.3, 0.25])
        np.testing.assert_allclose(p[-1], b.init_static()[0][-1], atol=5e-4)

        # outer tube fully retracted and out again, tube tips aligned (borders that coincide in exact arithmetic)
        size = len(a.get_state())
        for betas in ([0.3, 0.6], [0.3, 1.0], [0.3, 0.5], [1 - 0.124 * 0.63 / 0.185, 0.37], [0.1, 0.3]):
            a.translate(betas)
            a.step()
            self.assertEqual(len(a.get_state()), size)

        # the state holds the betas: set_state undoes a later translate
        x = a.get_state()
        a.translate([0.2, 0.4])
        a.set_state(x)
        np.testing.assert_array_equal(a.betas, [0.1, 0.3])
        with self.assertRaises(ValueError):
            a.set_state(x[:-1])


if __name__ == "__main__":
    unittest.main()
