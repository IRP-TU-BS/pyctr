import unittest

import matplotlib.pyplot as plt
import numpy as np
from pyctcr.yaml_to_model import *


from pyctcr import cosserat_rod_force_along
from pyctcr.robots import ConcentricTubeContinuumRobot


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

class RobotTestCase(unittest.TestCase):
    def test_something(self):
        tube_conf = setup_tubes("../example_robots/ctr_robot.yaml")

        rods = []
        tubes_lenghts = []

        for rod_conf in tube_conf:
            rod_conf['s'] = 1 * 1e-3
            rod_len = 0.5 #rod_conf['L'] * 1e-3
            rod_conf['L'] = rod_len
            rod_conf['straight_length'] = rod_len - rod_conf['curved_len'] * 1e-3
            rod_conf['curved_len'] = rod_conf['curved_len'] * 1e-3
            tubes_lenghts.append(rod_len)
            rod = cosserat_rod_force_along.CurvedCosseratRodExt(rod_conf)
            p0 = np.array([[0, 0, 0]])
            R0 = np.eye(3)
            rod.set_initial_conditions(p0, R0)
            rods.append(rod)
        ctr = ConcentricTubeContinuumRobot(rods)
        # test fwd kin
        ctr.fwd_kinematic()
        #ctr.fwd_static([0,0.1,0,0,0,0])
        #p0, _, _, _, _ = ctr.push_end([0, 0, 0, 0, 0, 0])
        L = 0.5
        p1, _, _, _, _ = ctr.push_end_to_position(np.array([0, -0.1*L, 0.8*L]))
        #print(p0)
        print(p1)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(p1[:,0],p1[:,1],p1[:,2])
        set_axes_equal(ax)
        plt.show()


if __name__ == '__main__':
    unittest.main()
