import numpy as np

from .cosserat_rod import *
from enum import Enum


class Force_Model(Enum):
    GAUSSIAN = 1
    FOURIER = 2


import pdb


class StraightCosseratRod(CosseratRod):
    """
    A class describing a straight rod with the capability to be pushed on several sides along the rods body
    """

    def __init__(self, params=None, force_model=Force_Model.GAUSSIAN):
        super(StraightCosseratRod, self).__init__(params)
        self._force_model = force_model
        self._ax = np.array([0, 1])
        self._ay = np.array([0, 1])
        self._bx = np.array([0, 1])
        self._by = np.array([0, 1])

        self._gaussiansx = [(0, 0, 0)]  # just to have a zero for the summation
        self._gaussiansy = [(0, 0, 0)]

    def cosserate_rod_ode(self, state, s):
        #print(s)
        R = np.reshape(state[3:12], (3, 3))
        n = state[12:15]
        m = state[15:]
        # u = state[18:21]
        #v = np.dot(np.linalg.inv(self.params['Kse']).dot(R.T), n) + np.array([[0, 0, 1]])
        u = np.dot(np.linalg.inv(self.params['Kbt']).dot(R.T), m)   # TODO research
        # ode
        ps = R.dot(np.array([[0,0,1]]).T)

        Rs = R.dot(hat(u))
        external_forces = self.external_gauss_forces(
            s) if self._force_model == Force_Model.GAUSSIAN else self.external_forces(s)
        #ns = -self.params['rho'] * self.params['A'] * self.params['g'].T + external_forces
        ns = -external_forces
        ms = -np.cross(ps.T[0], n)
        # pdb.set_trace()
        return np.hstack([ps.T[0], np.reshape(Rs, (1, 9))[0], ns.T[0], ms])

    def external_forces(self, s):
        fx = self._ax[0] + np.sum(self._ax[1:] * np.sin((s * np.pi *
                                                         np.linspace(1, self._ax[1:].shape[0], self._ax[1:].shape[0])) \
                                                        / self.params['L'])) \
             + np.sum(self._bx[1:] * np.sin((s * np.pi *
                                             np.linspace(1, self._bx[1:].shape[0], self._bx[1:].shape[0])) \
                                            / self.params['L']))
        fy = self._ay[0] + np.sum(self._ay[1:] * np.sin((s * np.pi *
                                                         np.linspace(1, self._ay[1:].shape[0], self._ay[1:].shape[0])) \
                                                        / self.params['L'])) \
             + np.sum(self._by[1:] * np.sin((s * np.pi *
                                             np.linspace(1, self._by[1:].shape[0], self._by[1:].shape[0])) \
                                            / self.params['L']))
        return np.array([[fx, fy, 0]]).T

    def external_gauss_forces(self, s):
        fx = np.sum(
            [self._gaussiansx[i][0] * np.exp(-self._gaussiansx[i][1] * (s - self._gaussiansx[i][2]) ** 2) for i in
             range(len(self._gaussiansx))])
        fy = np.sum(
            [self._gaussiansy[i][0] * np.exp(-self._gaussiansy[i][1] * (s - self._gaussiansy[i][2]) ** 2) for i in
             range(len(self._gaussiansy))])
        return np.array([[fx, fy, 0]]).T

    def shooting_function_force(self, guess):
        s = np.linspace(0, self.params['L'], 100)

        n0 = guess[:3]
        m0 = guess[3:6]
        tip_wrench = np.zeros(6)
        states = self.apply_force(np.hstack([n0, m0]))
        tip_wrench_shooting = states[-1][12:18]

        distal_force_error = tip_wrench[:3] - tip_wrench_shooting[:3]
        distal_moment_error = invhat(hat(tip_wrench[3:]).T.dot(hat(tip_wrench_shooting[3:])) - hat(tip_wrench[3:]).dot(
            hat(tip_wrench_shooting[3:]).T))
        return np.hstack([distal_force_error, distal_moment_error])

    def push_end(self, wrench):
        self.set_bounding_values(['tip_wrench'], [wrench])
        state = np.zeros((1, 6))
        solution_bvp = least_squares(self.shooting_function_force, state[0], method='lm', loss='linear', ftol=1e-6)
        states = self.apply_force(solution_bvp.x)
        return states


if __name__ == "__main__":
    pass
