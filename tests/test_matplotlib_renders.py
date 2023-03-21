import unittest
import matplotlib
matplotlib.use('Qt4Agg')
import numpy as np
from pyctcr.yaml_to_model import *
from pyctcr.visualization.scenes import *


from pyctcr import cosserat_rod_force_along
from pyctcr.robots import ConcentricTubeContinuumRobot


class RobotTestCase(unittest.TestCase):
    def test_matplotlib(self):
        test_scene = Scene(MatplotLib())
        robot = Robot(load_continous_ctcr_model, "../example_robots/ctr_robot.yaml")
        test_scene.add_robot('ctcr', robot)
        beta = np.linspace(0,0.5,100)
        #test_scene.add_arrows(0,0,0,1,1,1, 'red')
        test_scene.show_forces(True)
        for i,alpha in enumerate(np.linspace(0,np.pi,100)):
            robot.set_config(alphas=[alpha, 0], betas=[0,beta[i]])
            test_scene.update()
        test_scene.show()

    def test_vispy(self):
        test_scene = Scene(VisPy())
        robot = Robot(load_continous_ctcr_model, "../example_robots/ctr_robot.yaml")
        test_scene.add_robot('ctcr', robot)
        beta = np.linspace(0,0.5,100)
        for i,alpha in enumerate(np.linspace(0,np.pi/2,100)):
            robot.set_config(alphas=[-alpha, alpha], betas=[0,beta[i]])
        test_scene.update()
        test_scene.show()

if __name__ == '__main__':
    unittest.main()