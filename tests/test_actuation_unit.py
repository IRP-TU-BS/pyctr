import os
import tempfile
import unittest
import numpy as np

# STL files of https://github.com/IRP-TU-BS/irp_concentric_tube_continuum_robot, next to this repository by default
STL_DIR = os.environ.get(
    "PYCTR_ACTUATION_UNIT_STL",
    os.path.join(os.path.dirname(__file__), "../../irp_concentric_tube_continuum_robot/Models/stl_files"),
)

try:
    from pydrake.geometry import Role
    from pydrake.math import RigidTransform, RotationMatrix
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
    from pydrake.systems.framework import DiagramBuilder
    from pydrake.geometry import SceneGraph
    from pyctr.actuation_unit import SLEDGES_Z, X_CE, ActuationUnitCarriages, _carriage_parts, _fixed_parts
    from pyctr.actuation_unit import add_actuation_unit, read_stl
except ImportError:
    RigidTransform = None


@unittest.skipIf(RigidTransform is None or not os.path.isdir(STL_DIR), "drake or the STL files are missing")
class TestActuationUnit(unittest.TestCase):
    def test_tube_axis_through_tube_hole_and_carriage_bearings(self):
        # vertices of the placed parts in the exit frame E (z along the tubes): around the z axis the tube hole
        # (Ø5) of AdapterEnd directly behind the exit and the tube bearing seats (Ø18) of the three carriages
        X_EC = X_CE.inverse()
        parts = _fixed_parts() + [(n, R, np.asarray(p) + (0, 0, z), c) for z in SLEDGES_Z for n, R, p, c in _carriage_parts(0)[0]]
        for name, R, p, _ in parts:
            if name not in ("AdapterEnd.stl", "Sledge_V2.1.7.2.stl"):
                continue
            vertices, _ = read_stl(os.path.join(STL_DIR, name))
            X = X_EC.multiply(RigidTransform(R, np.asarray(p, dtype=float) * 1e-3))
            v = (X.rotation().matrix() @ (vertices * 1e-3).T).T + X.translation()
            radius = np.hypot(v[:, 0], v[:, 1])
            if name == "AdapterEnd.stl":
                self.assertAlmostEqual(v[:, 2].max(), 0, delta=1e-9)  # front face at the exit
                np.testing.assert_allclose(radius.min(), 2.5e-3, atol=1e-4)  # axis in the middle of the hole
            else:
                plate = v[:, 2] > v[:, 2].max() - 10e-3  # the 10 mm plate of the carriage
                np.testing.assert_allclose(radius[plate].min(), 9e-3, atol=1e-4)

    def test_exit_pose_in_plant(self):
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
        X_WUnit = RigidTransform([0.1, 0.2, 0.3])
        with tempfile.TemporaryDirectory() as cache:
            X_WExit = add_actuation_unit(plant, STL_DIR, X_BodyUnit=X_WUnit, tube_radii=[1.25e-3, 2.5e-3], cache_dir=cache)
            plant.Finalize()
            self.assertEqual(len(os.listdir(cache)), 12)  # OBJ copies of the 12 STL files that are used
        np.testing.assert_allclose(X_WExit.translation(), [0.1, 0.2, 0.3 + 0.1152], atol=1e-12)
        np.testing.assert_allclose(X_WExit.rotation().matrix(), [[0, 0, 1], [0, -1, 0], [1, 0, 0]], atol=1e-12)
        inspector = scene_graph.model_inspector()
        self.assertEqual(inspector.NumGeometriesWithRole(Role.kProximity), 0)  # visual only


    def test_carriages_follow_betas_and_alphas(self):
        class Robot:  # the attributes ActuationUnitCarriages reads
            class Tube:
                def __init__(self, L, r):
                    self.params = {"L": L, "r_outer": r}

            tubes = [(0, Tube(0.185, 1.25e-3)), (1, Tube(0.124, 2.5e-3))]
            alphas, betas = np.zeros(2), np.zeros(2)

        with tempfile.TemporaryDirectory() as cache:
            unit = ActuationUnitCarriages(SceneGraph(), STL_DIR, Robot(), cache_dir=cache)
        context = unit.CreateDefaultContext()
        betas, alphas = np.array([0.3, 0.2]), np.array([0.0, 1.0])
        unit.GetInputPort("betas").FixValue(context, betas)
        unit.GetInputPort("alphas").FixValue(context, alphas)
        poses = unit.GetOutputPort("geometry_pose").Eval(context)
        z = {}  # front face of each carriage from the pose of its sledge in the unit frame (x out of the tube hole)
        for frame, kind, c, X in unit._geometry:
            if kind == "sliding" and np.allclose(X.rotation().matrix(), np.eye(3)):  # the sledge (the nut is turned)
                z[c] = -poses.value(frame).multiply(X.inverse()).translation()[0] * 1e3
        # default transmission lengths: outer tube 31 mm, inner tube 81 mm; carriage 0 has no tube and is parked
        np.testing.assert_allclose([z[0], z[1], z[2]], [280, 81 + 185 * 0.3, 31 + 124 * 0.2], atol=1e-9)
        # turning parts of carriage 2 (outer tube) are turned by alpha = 1 about the tube axis (x of the world)
        frame = next(f for f, kind, c, X in unit._geometry if kind == "turning" and c == 2)
        X1 = poses.value(frame)
        unit.GetInputPort("alphas").FixValue(context, [0.0, 0.0])
        X0 = unit.GetOutputPort("geometry_pose").Eval(context).value(frame)
        X = X1.multiply(X0.inverse())  # rotation about a line parallel to x through the tube axis
        np.testing.assert_allclose(X.rotation().matrix(), RotationMatrix.MakeXRotation(1.0).matrix(), atol=1e-12)
        np.testing.assert_allclose(X.multiply([0, 0, 0.1152]), [0, 0, 0.1152], atol=1e-12)  # a point on the axis
        for wrong in ([0.0, -0.1], [0.3, 0.5], [0.0, 0.4]):  # in front of the travel, behind it, too close
            unit.GetInputPort("betas").FixValue(context, wrong)
            with self.assertRaises(ValueError):
                unit.GetOutputPort("geometry_pose").Eval(context)


if __name__ == "__main__":
    unittest.main()
