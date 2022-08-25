import unittest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyctcr.robots import ConcentricTubeContinuumRobot
from pyctcr.cosserat_rod import CurvedCosseratRod



class TestCTCRRobot(unittest.TestCase):

    def test_upper(self):
        # define base
        p0 = np.array([[0, 0, 0]])
        R0 = np.eye(3)

        # setup a single rod
        L_inner = 152  # mm

        L_inner_curved = 150  # mm
        kappa = 10.47  # curvature 1/R

        params = {}
        params['kappa'] = kappa
        params['alpha'] = 0
        params['beta'] = 0.
        params['straight_length'] = (L_inner - L_inner_curved) * 1e-3
        params['E'] = 1459985588.7614686  # not correct!
        params['G'] = 410231439.6207139  # not correct!
        params['L'] = L_inner * 1e-3  # conversion to m
        inner_rod = CurvedCosseratRod(params)
        inner_rod.set_initial_conditions(p0, R0)  # Arbitrary base frame assignment
        inner_rod.inital_conditions['kappa_0'] = np.array([0, 0, 0])  # no curvature at base frame

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(xs, ys, *args, **kwargs)


if __name__ == '__main__':
    unittest.main()
