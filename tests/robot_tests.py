import unittest

import numpy as np
from pyctcr.yaml_to_model import *


from pyctcr import cosserat_rod_force_along
from pyctcr.robots import ConcentricTubeContinuumRobot


class RobotTestCase(unittest.TestCase):
    def test_something(self):
        tube_conf = setup_tubes("../example_robots/ctr_robot.yaml")

        rods = []
        tubes_lenghts = []

        for rod_conf in tube_conf:
            rod_conf['s'] = 1 * 1e-3
            rod_len = rod_conf['L'] * 1e-3
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
        ctr.fwd_static([0,0.1,0,0,0,0])
        ctr.push_end([0,0,0,0,0,0])


if __name__ == '__main__':
    unittest.main()
