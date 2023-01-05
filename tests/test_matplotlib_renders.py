import unittest
import matplotlib
matplotlib.use('Qt4Agg')
import numpy as np
from pyctcr.yaml_to_model import *
from pyctcr.visualization.scenes import *


from pyctcr import cosserat_rod_force_along
from pyctcr.robots import ConcentricTubeContinuumRobot


class RobotTestCase(unittest.TestCase):
    def test_something(self):
        test_scene = Scene(MatplotLib())
        robot = Robot(load_continous_ctcr_model, "../example_robots/ctr_robot.yaml")
        test_scene.add_robot('ctcr', robot)
        beta = np.linspace(0,0.5,100)
        for i,alpha in enumerate(np.linspace(0,np.pi/2,100)):
            robot.set_config(alphas=[-alpha, alpha], betas=[0,beta[i]])
            test_scene.update()
        test_scene.show()

if __name__ == '__main__':
    unittest.main()