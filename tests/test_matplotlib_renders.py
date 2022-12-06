import unittest
import matplotlib
matplotlib.use('qtagg')
import numpy as np
from pyctcr.yaml_to_model import *
from pyctcr.visualization.scenes import *


from pyctcr import cosserat_rod_force_along
from pyctcr.robots import ConcentricTubeContinuumRobot


class RobotTestCase(unittest.TestCase):
    def test_something(self):
        test_scene = Scene(MatplotLib(), None)
        robot = Robot(load_continous_ctcr_model, "../example_robots/ctr_robot.yaml")
        test_scene.add_robot('ctcr', robot)
        robot.set_config(alphas=[np.pi/2, 0], betas=[0,0.5])
        test_scene.update()
        robot.set_config(alphas=[np.pi / 2, -np.pi/2], betas=[0, 0.])
        test_scene.update()
        test_scene.show()

if __name__ == '__main__':
    unittest.main()