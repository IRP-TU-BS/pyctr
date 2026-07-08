"""
Headless verification of the plotter fixes in pyctr/visualization/scenes.py, per
PLAN_SONNET.md. Run with: python -m pytest tests/test_plotter.py -x -q
"""

import os
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np

from pyctr.yaml_to_model import load_continous_ctcr_model
from pyctr.visualization.scenes import MatplotLib, Robot, Scene, tube_end_indices

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "..", "example_robots", "ctr_robot.yaml")


def build_frame(alphas=(0.5, 0.0), betas=(0.0, 0.1)):
    robot = Robot(load_continous_ctcr_model, CONFIG)
    scene = Scene(MatplotLib())
    scene.add_robot("ctcr", robot)
    robot.set_config(alphas=list(alphas), betas=list(betas))
    scene.update()
    frame = scene.robot_frames["ctcr"][-1]
    return scene, frame


class PlotterTestCase(unittest.TestCase):
    def test_import_without_vispy(self):
        import pyctr.visualization.scenes as s

        self.assertTrue(hasattr(s, "MatplotLib"))

    def test_axis_limits_contain_robot(self):
        scene, frame = build_frame()
        engine = scene.engine
        engine.displ_fwd(frame, "ctcr", show_forces=True)

        pos = frame[0]
        xlim = engine.ax.get_xlim3d()
        ylim = engine.ax.get_ylim3d()
        zlim = engine.ax.get_zlim3d()

        self.assertLessEqual(xlim[0], pos[:, 0].min())
        self.assertGreaterEqual(xlim[1], pos[:, 0].max())
        self.assertLessEqual(ylim[0], pos[:, 1].min())
        self.assertGreaterEqual(ylim[1], pos[:, 1].max())
        self.assertLessEqual(zlim[0], pos[:, 2].min())
        self.assertGreaterEqual(zlim[1], pos[:, 2].max())

    def test_shadow_tracks_tube(self):
        scene, frame = build_frame()
        engine = scene.engine
        engine.displ_fwd(frame, "ctcr", show_forces=False)

        segments = frame[1]
        number_of_tubes = frame[2]
        ends = tube_end_indices(segments, number_of_tubes)

        for tube_idx, end in enumerate(ends, start=1):
            self.assertIsNotNone(end)
            tube_line = engine.plotted_robots["ctcr"]["ctcrtube" + str(tube_idx)]
            shade_line = engine.plotted_robots["ctcr"][
                "ctcrtube" + str(tube_idx) + "_shade"
            ]

            tube_x, tube_y, tube_z = tube_line._verts3d
            shade_x, shade_y, shade_z = shade_line._verts3d

            np.testing.assert_allclose(shade_x, tube_x)
            np.testing.assert_allclose(shade_y, tube_y)
            self.assertEqual(len(shade_z), len(shade_x))
            np.testing.assert_allclose(shade_z, np.zeros_like(np.asarray(shade_z)))

    def test_tube_lines_nondegenerate_and_end_at_expected_point(self):
        scene, frame = build_frame()
        engine = scene.engine
        engine.displ_fwd(frame, "ctcr", show_forces=False)

        pos = frame[0]
        segments = frame[1]
        number_of_tubes = frame[2]
        ends = tube_end_indices(segments, number_of_tubes)

        for tube_idx, end in enumerate(ends, start=1):
            self.assertIsNotNone(end)
            tube_line = engine.plotted_robots["ctcr"]["ctcrtube" + str(tube_idx)]
            x, y, z = tube_line._verts3d
            pts = np.column_stack([x, y, z])
            self.assertGreater(len(np.unique(pts, axis=0)), 1)

            expected_last = pos[end - 1]
            np.testing.assert_allclose(pts[-1], expected_last)

    def test_no_arrows_for_zero_wrench(self):
        scene, frame = build_frame()
        engine = scene.engine
        engine.displ_fwd(frame, "ctcr", show_forces=True)
        self.assertEqual(len(engine.arrows), 0)

    def test_arrows_appear_for_nonzero_wrench(self):
        scene, frame = build_frame()
        engine = scene.engine
        frame = list(frame)
        wrenches = np.array(frame[4], dtype=float)
        wrenches[:, :3] = np.array([0.05, 0.02, 0.0])
        frame[4] = wrenches

        engine.displ_fwd(frame, "ctcr", show_forces=True)
        self.assertGreater(len(engine.arrows), 0)
        for arrow in engine.arrows.values():
            for coord in list(arrow._xyz) + list(arrow._dxdydz):
                self.assertTrue(np.isfinite(coord))

    def test_arrows_do_not_persist_across_frames(self):
        scene, frame1 = build_frame(alphas=(0.2, 0.0), betas=(0.0, 0.0))
        engine = scene.engine

        frame1 = list(frame1)
        w1 = np.array(frame1[4], dtype=float)
        w1[:, :3] = np.array([0.05, 0.0, 0.0])
        frame1[4] = w1
        engine.displ_fwd(frame1, "ctcr", show_forces=True)
        frame1_arrow_count = len(engine.arrows)
        self.assertGreater(frame1_arrow_count, 0)

        # second frame with zero wrench must clear frame 1's arrows, not accumulate
        _, frame2 = build_frame(alphas=(0.8, 0.0), betas=(0.0, 0.2))
        engine.displ_fwd(frame2, "ctcr", show_forces=True)
        self.assertEqual(len(engine.arrows), 0)

    def test_save_png(self):
        scene, frame = build_frame()
        engine = scene.engine
        engine.displ_fwd(frame, "ctcr", show_forces=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "robot.png")
            engine.fig.savefig(path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_vispy_engine_raises_clear_import_error(self):
        from pyctr.visualization.scenes import VisPy

        with self.assertRaises(ImportError):
            VisPy()


if __name__ == "__main__":
    unittest.main()
