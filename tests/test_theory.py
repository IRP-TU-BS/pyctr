"""
Quantitative checks of the CTCR/Cosserat model against analytic results from the theory
papers (Rucker/Jones/Webster 2010 and the CR lecture notes). These are the acceptance
tests for the physics fixes described in PLAN_OPUS.md.
"""

import copy
import os
import unittest

import matplotlib

matplotlib.use("Agg")
import numpy as np

from pyctr.cosserat_rod_force_along import CurvedCosseratRodExt
from pyctr.robots import ConcentricTubeContinuumRobot, shooting_function_force
from pyctr.utils import Rz
from pyctr.yaml_to_model import setup_tubes

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "..", "example_robots", "ctr_robot.yaml")


def make_rod(**overrides):
    params = {
        "E": 339889091.1107947,
        "G": 97898219.42862949,
        "r_inner": 0.0,
        "r_outer": 1.25e-3,
        "rho": 8000,
        "g": 0,
        "L": 0.1,
        "straight_length": 0.0,
        "curved_len": 0.1,
        "s": 1e-3,
        "kappa": 8.0,
    }
    params.update(overrides)
    rod = CurvedCosseratRodExt(params)
    rod.set_initial_conditions(np.array([[0, 0, 0]]), np.eye(3))
    return rod


def build_robot(tube_params_list):
    """Build a CTCR from explicit per-tube parameter dicts (lengths already in metres)."""
    rods = []
    for params in tube_params_list:
        rod = CurvedCosseratRodExt(copy.deepcopy(params))
        rod.set_initial_conditions(np.array([[0, 0, 0]]), np.eye(3))
        rods.append(rod)
    return ConcentricTubeContinuumRobot(rods)


def load_example_tube_params():
    tube_conf = setup_tubes(CONFIG)
    out = []
    for rod_conf in tube_conf:
        rod_conf = copy.deepcopy(rod_conf)
        rod_conf["s"] = 1e-3
        rod_len = rod_conf["L"] * 1e-3
        rod_conf["L"] = rod_len
        rod_conf["straight_length"] = rod_len - rod_conf["curved_len"] * 1e-3
        rod_conf["curved_len"] = rod_conf["curved_len"] * 1e-3
        out.append(rod_conf)
    return out


class TheoryTestCase(unittest.TestCase):
    def test_precurved_rod_is_circular_arc(self):
        """A single precurved rod with no load bends into a circular arc of radius 1/kappa."""
        kappa = 8.0
        L = 0.1
        rod = make_rod(kappa=kappa, L=L, straight_length=0.0, g=0)
        p, R, w = rod.apply_force(np.zeros(6), steps=400, curvature_integration=False)

        # planar arc: curvature about +y keeps the curve in the x-z plane (y == 0)
        self.assertLess(np.max(np.abs(p[:, 1])), 1e-9)

        # chord length between the endpoints of a circular arc of radius r=1/kappa
        # spanning arc length L equals 2 r sin(kappa L / 2)
        r = 1.0 / kappa
        expected_chord = 2 * r * np.sin(kappa * L / 2)
        chord = np.linalg.norm(p[-1] - p[0])
        self.assertAlmostEqual(chord, expected_chord, places=4)

        # every point lies on a circle of radius r about the analytic centre
        # centre = p0 + r * (principal normal). The initial tangent is +z and the curve
        # bends toward +x, so the centre is at p0 + r * x_hat.
        centre = p[0] + r * np.array([1.0, 0.0, 0.0])
        radii = np.linalg.norm(p - centre, axis=1)
        self.assertLess(np.max(np.abs(radii - r)), 1e-3)

    def test_stiffness_weighted_bending_average(self):
        """In the overlap region the bending curvature is the EI-weighted mean of the
        rotated pre-curvatures (Rucker 2010, unloaded case). Two coextensive tubes with
        different curvature and different stiffness make the average distinct from both."""
        base = {
            "E": 339889091.1107947,
            "G": 97898219.42862949,
            "rho": 8000,
            "g": 0,
            "L": 0.15,
            "straight_length": 0.05,
            "curved_len": 0.10,
            "s": 1e-3,
            "r_inner": 0.0,
        }
        tube_a = {**base, "kappa": 6.0, "r_outer": 1.25e-3}
        tube_b = {**base, "kappa": 12.0, "r_outer": 2.0e-3}
        robot = build_robot([tube_a, tube_b])
        robot.rotate([0.0, 0.0])
        robot.translate([0.0, 0.0])
        _, _, _, us, _, _ = robot.fwd_static(np.zeros(6), 0.002)

        EI = [t[1].params["Kbt"][0, 0] for t in robot.tubes]
        kappa = [t[1].params["kappa"] for t in robot.tubes]
        expected = (EI[0] * kappa[0] + EI[1] * kappa[1]) / (EI[0] + EI[1])

        # both tubes are coextensive and curved at the tip; sample there
        overlap_curvature = us[-1, 1]
        self.assertNotAlmostEqual(kappa[0], expected, places=2)
        self.assertNotAlmostEqual(kappa[1], expected, places=2)
        self.assertAlmostEqual(overlap_curvature, expected, places=2)

    def test_opposed_identical_tubes_cancel(self):
        """Two identical tubes rotated by pi cancel: the overlap region is straight."""
        params = load_example_tube_params()[0]  # inner tube
        robot = build_robot([params, params])
        robot.rotate([0.0, np.pi])
        robot.translate([0.0, 0.0])
        _, _, _, us, _, _ = robot.fwd_static(np.zeros(6), 0.002)

        # both tubes identical length -> the whole curved section overlaps and cancels
        self.assertLess(np.max(np.linalg.norm(us[:, :2], axis=1)), 1e-6)

    def test_rigid_rotation_of_base(self):
        """Rotating every tube by the same delta rotates the whole shape rigidly."""
        robot = build_robot(load_example_tube_params())
        robot.translate([0.0, 0.0])

        robot.rotate([0.0, np.pi])
        p0, _, _, _, _, _ = robot.fwd_static(np.zeros(6), 0.004)

        delta = 0.7
        robot.rotate([delta, delta + np.pi])
        pa, _, _, _, _, _ = robot.fwd_static(np.zeros(6), 0.004)

        rotated = (Rz(delta) @ p0.T).T
        self.assertLess(np.max(np.abs(rotated - pa)), 1e-4)

    def test_tip_wrench_boundary_conditions(self):
        """push_end solves the BVP: the world-frame tip force matches the applied force,
        and every tube's torsional free-end condition u_iz(L_i)=0 holds."""
        robot = build_robot(load_example_tube_params())
        robot.rotate([0.3, -0.2])
        robot.translate([0.0, 0.0])
        step = 0.004

        wrench = np.array([0.2, 0.05, 0.0, 0.0, 0.0, 0.0])
        p, R, w, us, uzs, th = robot.push_end(wrench, step)

        # residual of the shooting system at the returned solution
        robot.set_boundary_condition(["tip_wrench"], [wrench])
        guess = np.hstack([w[0, :3], w[0, 3:], uzs[0]])
        residual = shooting_function_force(guess, robot, step)
        robot.remove_boundary_condition("tip_wrench")
        self.assertLess(np.linalg.norm(residual), 1e-6)

        # world-frame internal tip force equals the applied force
        self.assertLess(np.linalg.norm(w[-1, :3] - wrench[:3]), 1e-6)

        # torsional free end at each tube's own tip
        tips = robot.tube_tip_indices()
        for i in range(len(robot.tubes)):
            self.assertLess(abs(uzs[tips[i], i]), 1e-6)

    def test_unloaded_matches_push_zero(self):
        """Pushing with a zero tip wrench reproduces the plain forward kinematics."""
        robot = build_robot(load_example_tube_params())
        robot.rotate([0.0, 0.0])
        robot.translate([0.0, 0.0])
        p_fwd, _, _, _, _, _ = robot.fwd_static(np.zeros(6), 0.004)
        p_push, _, _, _, _, _ = robot.push_end(np.zeros(6), 0.004)
        self.assertLess(np.max(np.abs(p_fwd - p_push)), 1e-6)


if __name__ == "__main__":
    unittest.main()
