import numpy as np

from .cosserat_rod import CosseratRod, CurvedCosseratRod, hat, least_squares
from enum import Enum


class Force_Model(Enum):
    GAUSSIAN = 1
    FOURIER = 2


class ExtForceRod(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # This class is a mixin
        self._gaussians = [((0, 0), 0, 0)]  # just to have a zero for the summation

    def external_gauss_forces(self, s):
        fx = np.sum(
            [
                self._gaussians[i][0][0]
                * np.exp(-self._gaussians[i][1] * (s - self._gaussians[i][2]) ** 2)
                for i in range(len(self._gaussians))
            ]
        )
        fy = np.sum(
            [
                self._gaussians[i][0][1]
                * np.exp(-self._gaussians[i][1] * (s - self._gaussians[i][2]) ** 2)
                for i in range(len(self._gaussians))
            ]
        )
        return np.array([[fx, fy, 0]]).T

    def external_fourier_forces(self, s):
        fx = (
            self._ax[0]
            + np.sum(
                self._ax[1:]
                * np.sin(
                    (
                        s
                        * np.pi
                        * np.linspace(1, self._ax[1:].shape[0], self._ax[1:].shape[0])
                    )
                    / self.params["L"]
                )
            )
            + np.sum(
                self._bx[1:]
                * np.sin(
                    (
                        s
                        * np.pi
                        * np.linspace(1, self._bx[1:].shape[0], self._bx[1:].shape[0])
                    )
                    / self.params["L"]
                )
            )
        )
        fy = (
            self._ay[0]
            + np.sum(
                self._ay[1:]
                * np.sin(
                    (
                        s
                        * np.pi
                        * np.linspace(1, self._ay[1:].shape[0], self._ay[1:].shape[0])
                    )
                    / self.params["L"]
                )
            )
            + np.sum(
                self._by[1:]
                * np.sin(
                    (
                        s
                        * np.pi
                        * np.linspace(1, self._by[1:].shape[0], self._by[1:].shape[0])
                    )
                    / self.params["L"]
                )
            )
        )
        return np.array([[fx, fy, 0]]).T

    def apply_external_forces(self):
        state = np.zeros((1, 6))
        solution_bvp = least_squares(
            self.shooting_function_external_force,
            state[0],
            method="lm",
            loss="linear",
            ftol=1e-6,
        )
        states = self.apply_force(solution_bvp.x)
        return states

    def shooting_function_external_force(self, guess, s_l=100):
        # gaussian_indexes = []
        # for gauss in self._gaussians:
        #    gaussian_indexes.append(int((gauss[2] / self.params['L']) * (s_l - 1)))

        n0 = guess[:3]
        m0 = guess[3:6]
        tip_wrench = np.zeros(6)
        states = self.apply_force(np.hstack([n0, m0]), s_l)
        tip_wrench_shooting = states[-1, 12:18]
        # diff_sums = (np.linalg.norm(states[:,12:18]) - np.sqrt(np.pi/(self._gaussians[0][1]+0.000000001))*np.linalg.norm(np.asarray(self._gaussians[0][0])))**2

        return np.hstack([(tip_wrench - tip_wrench_shooting) ** 2])


class StraightCosseratRod(ExtForceRod, CosseratRod):
    """
    A class describing a straight rod with the capability to be pushed on several sides along the rods body
    """

    def __init__(self, params=None, force_model=Force_Model.GAUSSIAN):
        super().__init__(params)
        self._force_model = force_model

    def cosserate_rod_ode(self, state, s):
        R = np.reshape(state[3:12], (3, 3))
        n = state[12:15]
        m = state[15:]
        u = np.dot(np.linalg.inv(self.params["Kbt"]).dot(R.T), m)  # + self.get_kappa()
        # ode
        ps = R.dot(np.array([[0, 0, 1]]).T)

        Rs = R.dot(hat(u))
        external_forces = (
            self.external_gauss_forces(s)
            if self._force_model == Force_Model.GAUSSIAN
            else self.external_fourier_forces(s)
        )
        # ns = -self.params['rho'] * self.params['A'] * self.params['g'].T + external_forces
        ns = -R @ external_forces
        ms = -np.cross(ps.T[0], n)

        return np.hstack([ps.T[0], np.reshape(Rs, (1, 9))[0], ns.T[0], ms])


class CurvedCosseratRodExt(ExtForceRod, CurvedCosseratRod):
    """
    A class describing a straight rod with the capability to be pushed on several sides along the rods body
    """

    def __init__(self, params=None, force_model=Force_Model.GAUSSIAN):
        super().__init__(params)
        self._force_model = force_model

    # def cosserate_rod_ode(self, state, s):
    #     R = np.reshape(state[3:12], (3, 3))
    #     # R_k = hat(np.array([0,0,self.params['k']]))
    #     # R = R @ R_k
    #     n = state[12:15]
    #     m = state[15:18]
    #     u = state[18:21]
    #
    #     u_i_star_ds = invhat(R @ hat(u))
    #     #u_div = u_i_star_ds + #self.get_u_div(R, n, m, u)
    #
    #
    #     ps = R.dot(self._e3.T)  # simplification -> Kirchoff rod
    #     Rs = R.dot(hat(u))
    #     external_forces = self.external_gauss_forces(
    #         s) if self._force_model == Force_Model.GAUSSIAN else self.external_forces(s)
    #     #ns = -self.params['rho'] * self.params['A'] * self.params['g'].T - R@external_forces
    #     ns = -R@external_forces
    #     ms = -np.cross(ps.T[0], n)
    #     return np.hstack([ps.T[0], np.reshape(Rs, (1, 9))[0], ns.T[0], ms])

    def cosserate_rod_ode(self, state, s):
        R = np.reshape(state[3:12], (3, 3))
        n = state[12:15]
        m = state[15:]
        u = np.dot(np.linalg.inv(self.params["Kbt"]).dot(R.T), m) + self.get_kappa()
        external_forces = (
            self.external_gauss_forces(s)
            if self._force_model == Force_Model.GAUSSIAN
            else self.external_fourier_forces(s)
        )

        # ode
        ps = R.dot(self._e3.T)
        Rs = R.dot(hat(u))
        ns = (
            -self.params["rho"] * self.params["A"] * self.params["g"].T
            - R @ external_forces
        )
        ms = -np.cross(ps.T[0], n)  # -l = 0

        return np.hstack([ps.T[0], np.reshape(Rs, (1, 9))[0], ns.T[0], ms])
