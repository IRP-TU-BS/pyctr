import pdb
import timeit
import numpy as np
import scipy as sc
from scipy import integrate
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from .utils import *


class ConcentricTubeContinuumRobot:
    """

    """
    def __init__(self, tubes):
        """

        :param tubes: list of tubes in the robot sorted by inner to outer, longest -> shortest
        """
        self.tubes = list(zip(range(len(tubes)), tubes)) # zip with number to have an order
        self.alphas = np.zeros(len(tubes)) # rotation variables
        self.betas = np.zeros(len(tubes)) # elongation variables

        self._integrate_index = 0  #TODO ?
        self.tubes_end_indexes = [] # TODO ?
        self.bounding_values = None # TODO ?

    def get_ordered_segments(self):
        """
        Caclulates the start and end of each segment (a part of the robot that has a constant curvature)
        :return: list of sorted segment ends (starting 0 as base)
        """
        ends = [(rod[1].params['straight_length'] - self.betas[i],
                 rod[1].params['L'] - rod[1].params['straight_length'] - self.betas[i]) for i, rod in enumerate(self.tubes)]
        return list(np.sort([0] + [item for t in ends for item in t]))

    def calc_forward(self, R_init, p_init, wrench, step_len):
        R = R_init
        p = p_init[0]
        self.step_len = step_len

        segment_list = self.get_ordered_segments()
        ode_returns = []
        for i in range(1,len(segment_list)):
            self._curr_calc_tubes = [] # gather tubes that determine this segment. Attribute because it is need in ode
            for t in self.tubes:
                if t[1].is_curved_or_at_end(segment_list[i]) >= 0:
                    self._curr_calc_tubes.append(t)

            """
             __guess state__
             p - positions
             R - orientations
             wrench - wrench
             uxy - curvature along x and y
             n*uz - torsion of each tube
            """

            state = np.hstack([p, R.reshape((1, 9))[0], wrench, np.zeros(3), np.zeros(len(self.tubes))]) # guess

            ode_states = integrate.solve_ivp(self.cosserate_rod_ode, (
                segment_list[i-1], segment_list[i]), state, dense_output=True, max_step=step_len) # beta is defined 0-1 -> L-L*beta  if beta 0 -> fully elongated tube
            p = ode_states.y.T[-1:,:3][0]
            R = np.reshape(ode_states.y.T[-1:,3:12], (3, 3))
            ode_returns.append(ode_states.y.T)
        return np.vstack(ode_returns)

    def get_curvature_vetor(self, kappa):
            return np.array([0, kappa, 0])
    def cosserate_rod_ode(self, s, state):
        R = np.reshape(state[3:12], (3, 3))
        n = state[12:15]
        m = state[15:18]
        u = state[18:21]

        new_u_s = np.zeros(3)
        summed_K = np.zeros((3, 3)) # see Rucker and Webster

        tube_z_torsions = []
        for tube in self._curr_calc_tubes:
            curved_part = tube[1].is_curved_or_at_end(s) # returns 1 for curved, 0 for not curved and -1 if s > L_t. Last one should not occure at this point
            if tube[0] > 0:
                theta = tube[1].params['alpha'] - self.tubes[1][0].params['alpha']
                R_theta = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1],
                                    ])
            else:
                R_theta = np.identity(3)

            summed_K += tube[1].params['Kbt'] # adding the Ks

            EI = 0.0197 # tube[1].params['E'] * tube[1].params['I']
            JG = 0.0123 # tube[1].params['G'] * tube[1].params['J']

            tube_curvature = tube[1].params["kappa"] * curved_part # also weired TODO
            u_tube = self.get_curvature_vetor(tube_curvature)

            u_i_star_div = invhat((R @ R_theta) @ hat(u_tube))
            u_i_star = invhat((R @ R_theta).T @ hat(u_i_star_div))

            u_div = tube[1].params['Kbt'] @ (-u_i_star_div) + (hat(u) @ tube[1].params['Kbt']) @ (u - u_i_star)
            #- ( np.dot(hat(tube[1]._e3[0]) @ R.T, self.step_len * np.asarray([n]).T).T[0] + R.T @ m) # external

            #tube_z_torsions.append((tube[0],u_div[2]))

            tube_z_torsions.append((tube[0],
             u_i_star_div[2]
            + (EI / JG) * (u[0] * u_i_star[1] - u[1] * u_i_star[0])
            + (1 / JG) * (u_i_star[2]-u[2])))
            #- (1 / JG) * (tube[1]._e3 @ (R @ R_theta) @ m))) # external

            new_u_s[:2] += u_div[:2] # TODO what does 3 mean?

        tube_z_torsions.sort(key=lambda y: y[0])
        new_u_s[2:3] = tube_z_torsions[0][1]

        new_u_s = np.linalg.inv(summed_K) @ new_u_s # TODO dimension missmatch
        ns = np.sum([-self.tubes[i][1].params['rho'] * self.tubes[i][1].params['A'] * self.tubes[i][1].params['g'].T for i in
                     range(len(self._curr_calc_tubes))], axis=0)
        ps = R.dot(np.array([[0, 0, 1]]).T)
        Rs = R.dot(hat(new_u_s))
        ms = -np.cross(ps.T[0], n)
        return np.hstack([ps.T[0],
                          np.reshape(Rs, (1, 9))[0],
                          ns.T[0],
                          ms,
                          new_u_s,
                          np.asarray([u[1] for u in tube_z_torsions])[0]])

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
        state = self.calc_forward(self.tubes[0][1].inital_conditions['R0'], self.tubes[0][1].inital_conditions['p0'],
                                  np.asarray(wrench), step_size)
        return state

    def fwd_static(self, wrench, step_size=0.01):
        state = self._apply_fwd_static(wrench, step_size)
        positions = state[:, :3]
        orientations = state[:, 3:12]
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
