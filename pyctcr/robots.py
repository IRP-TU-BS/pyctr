import pdb
import timeit
import numpy as np
import scipy as sc
from scipy import integrate
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from .utils import *

from cosserat_rod import  *


class ConcentricTubeContinuumRobot:
    """

    """
    def __init__(self, tubes):
        """

        :param tubes: list of tubes in the robot sorted by inner to outer, longest -> shortest
        """
        self.tubes = tubes
        self.alphas = np.zeros(len(self.tubes)) # rotation variables
        self.betas = np.zeros(len(self.tubes)) # elongation variables

        self._integrate_index = 0  #TODO ?
        self.tubes_end_indexes = [] # TODO ?
        self.bounding_values = None # TODO ?

    def get_ordered_segments(self):
        """
        Caclulates the start and end of each segment (a part of the robot that has a constant curvature)
        :return: list of sorted segment ends (starting 0 as base)
        """
        ends = [(rod.params['straight_length'] - self.betas[i],
                 rod.params['L'] - rod.params['straight_length'] - self.betas[i]) for i, rod in enumerate(self.tubes)]
        return list(np.sort([0] + [item for t in ends for item in t]))

    @property
    def betas(self):
        """betas getter"""
        return self.betas

    @betas.setter
    def betas(self, betas):
        # TODO check beta contraints
        self.betas = betas

    def calc_forward(self, R_init, p_init, wrench, step_len):
        R = R_init
        p = p_init[0]
        self.step_len = step_len
        self._integrate_index = 0 # TODO ?
        self.tubes_end_indexes = [] # TODO ?
        state = np.hstack([p, R.reshape((1, 9))[0], wrench, np.zeros(3)]) # guess
        return integrate.solve_ivp(self.cosserate_rod_ode, (
        0, self.tubes[0].params['L'] - self.tubes[0].params['L'] * self.tubes[0].params['beta']), state,
                                   dense_output=True, max_step=step_len) # beta is defined 0-1 -> L-L*beta  if beta 0 -> fully elongated tube

    def cosserate_rod_ode(self, s, state):
        R = np.reshape(state[3:12], (3, 3))
        n = state[12:15]
        m = state[15:18]
        u = state[18:21]

        new_u = np.zeros(u.shape) # TODO ?

        avail_tubes = 0 # how many tubes overlay each other

        K = np.zeros((3, 3)) # ??? TODO  we add K together?

        self._integrate_index += 1 # TODO ?

        for t in range(0, len(self.tubes)):
            rod = self.tubes[t]
            curved_part = rod.is_curved_or_at_end(s) # returns 1 for curved, 0 for not curved and -1 if s > L_t
            if -1 == int(curved_part):
                self.tubes_end_indexes.append(self._integrate_index) # ?
                continue
            avail_tubes += 1 # if s is along
            K = rod.params['Kbt']
            if t > 0:
                theta = rod.params['alpha'] - self.tubes[0].params['alpha']
                R_theta = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1],
                                    ])
            else:
                R_theta = np.identity(3)

            rod._step_size = self.step_len # TODO ?
            u_rod_skala = rod.params["kappa"] * curved_part # also weired TODO
            rod.set_kappa(u_rod_skala) # TODO more weired

            u_i_star_div = invhat((R @ R_theta) @ hat(rod.get_kappa()))
            u_i_star = invhat((R @ R_theta).T @ hat(u_i_star_div))

            u_div = rod.params['Kbt'] @ (-u_i_star_div) + (hat(u) @ rod.params['Kbt']) @ (u - u_i_star) - (
                        np.dot(hat(rod._e3[0]) @ R.T, rod._step_size * np.asarray([n]).T).T[0] + R.T @ m)

            u_div_z = u_i_star_div[2] + (rod.params['E'] * rod.params['I']) / (rod.params['G'] * rod.params['J']) * (
                        u[0] * u_i_star[1] - u[1] * u_i_star[0]) - 1 / (rod.params['G'] * rod.params['J']) * (
                                  rod._e3 @ R_theta @ m)

            new_u[:2] += u_div[:2]
            new_u[2:] += u_div_z

        new_u = np.linalg.inv(K) @ new_u
        ns = np.sum([-self.tubes[i].params['rho'] * self.tubes[i].params['A'] * self.tubes[i].params['g'].T for i in
                     range(0, avail_tubes)], axis=0)
        ps = R.dot(np.array([[0, 0, 1]]).T)
        Rs = R.dot(hat(new_u))
        ms = -np.cross(ps.T[0], n)
        return np.hstack([ps.T[0], np.reshape(Rs, (1, 9))[0], ns.T[0], ms, new_u])

    def set_bounding_values(self, names, values):
        if self.bounding_values is None:
            self.bounding_values = {}
        for i, name in enumerate(names):
            self.bounding_values[name] = values[i]

    def rotate(self, alphas):
        for i in range(len(self.tubes)):
            self.tubes[i].params["alpha"] = alphas[i]

    def translate(self, betas):
        if self._check_beta_validity(betas):
            for i in range(len(self.tubes)):
                self.tubes[i].params["beta"] = betas[i]
        else:
            raise Exception('Parameter Error', 'The beta values do not correspond to the specifications!')

    def _check_beta_validity(self, betas):
        valid = True
        for i in range(1, np.asarray(betas).shape[0]):
            valid = valid and betas[i - 1] <= betas[i]
            valid = valid and betas[i] * self.tubes[i].params['L'] <= betas[i - 1] * self.tubes[i - 1].params['L']
        return valid

    def _apply_fwd_static(self, wrench, step_size=0.01):
        state = self.calc_forward(self.tubes[0].inital_conditions['R0'], self.tubes[0].inital_conditions['p0'],
                                  np.asarray(wrench), step_size)
        return state

    def fwd_static(self, wrench, step_size=0.01):
        state = self._apply_fwd_static(wrench, step_size)
        positions = state.y.T[:, :3]
        orientations = state.y.T[:, 3:12]
        return positions, orientations

    def shooting_function_force(self, guess):
        n0 = guess[:3]
        m0 = guess[3:6]
        tip_wrench = self.bounding_values['tip_wrench']
        states = self._apply_fwd_static(np.hstack([n0, m0]))
        tip_wrench_shooting = states.y.T[-1, 12:18]

        distal_force_error = tip_wrench[:3] - tip_wrench_shooting[:3]
        distal_moment_error = invhat(hat(tip_wrench[3:]).T.dot(hat(tip_wrench_shooting[3:])) - hat(tip_wrench[3:]).dot(
            hat(tip_wrench_shooting[3:]).T))
        return np.hstack([distal_force_error, distal_moment_error])

    def push_end(self, wrench):
        self.set_bounding_values(['tip_wrench'], [wrench])
        state = np.zeros((1, 6))
        solution_bvp = least_squares(self.shooting_function_force, state[0], method='lm', loss='linear', ftol=1e-6)
        states = self._apply_fwd_static(solution_bvp.x)
        positions = states.y.T[:, :3]
        orientations = states.y.T[:, 3:12]
        return positions, orientations