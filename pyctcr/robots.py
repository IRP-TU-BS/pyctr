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
        ends = [(rod[1].params['straight_length'] - rod[1].params['L']*(self.betas[i]),
                 rod[1].params['L']  - rod[1].params['L']*(self.betas[i])) for i, rod in enumerate(self.tubes)]
        sorted_ends = np.sort([0] + [item for t in ends for item in t])
        sorted_ends = sorted_ends[sorted_ends >= 0]
        return list(sorted_ends)

    def calc_forward(self, R_init, p_init, wrench, step_len):
        R = R_init
        p = p_init[0]
        self.step_len = step_len
        w = wrench

        segment_list = self.get_ordered_segments()
        ode_returns = []
        u = np.zeros(3)
        uzs = np.zeros(len(self.tubes))
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

            state = np.hstack([p, R.reshape((1, 9))[0], w, u, uzs]) # guess

            ode_states = integrate.solve_ivp(self.cosserate_rod_ode, (
                segment_list[i-1], segment_list[i]), state, dense_output=True, max_step=step_len) # beta is defined 0-1 -> L-L*beta  if beta 0 -> fully elongated tube
            p = ode_states.y.T[-1:,:3][0]
            R = np.reshape(ode_states.y.T[-1:,3:12], (3, 3))
            w = ode_states.y.T[-1,12:18]
            u = ode_states.y.T[-1,18:21]
            uzs = ode_states.y.T[-1,21:]
            ode_returns.append(ode_states.y.T)
        return np.vstack(ode_returns)

    def get_curvature_vector(self, kappa):
            return np.array([0, kappa, 0]) # we consider that the curvature is defined over y
    def cosserate_rod_ode(self, s, state):
        R = np.reshape(state[3:12], (3, 3))
        n = state[12:15]
        m = state[15:18]
        u = state[18:21]

        new_u_s = np.zeros(3)
        summed_K = np.zeros((3, 3)) # see Rucker and Webster

        tube_z_torsions = []
        tube_uzs = np.zeros(len(self.tubes)).tolist()
        for tube in self._curr_calc_tubes:
            curved_part = tube[1].is_curved_or_at_end(s) # returns 1 for curved, 0 for not curved and -1 if s > L_t. Last one should not occure at this point

            theta = self.alphas[tube[0]] - self.alphas[self.tubes[0][0]]
            R_theta = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1],
                                    ])
            R_theta_dtheta = np.array([[-np.sin(theta), -np.cos(theta), 0],
                                    [np.cos(theta), -np.sin(theta), 0],
                                    [0, 0, 0],
                                    ])

            theta_ds = state[21+tube[0]] - state[21]

            summed_K += tube[1].params['Kbt'] # adding the Ks

            EI = tube[1].params['E'] * tube[1].params['I']
            JG = tube[1].params['G'] * tube[1].params['J']

            tube_curvature = tube[1].params["kappa"] * curved_part # also weired TODO
            u_tube = self.get_curvature_vector(tube_curvature)

            u_i_star_ds = invhat((R @ R_theta) @ hat(u_tube))
            u_i_star = invhat((R @ R_theta).T @ hat(u_i_star_ds))

            u_i = u.copy()
            u_i[2] = state[21+tube[0]]

            u_div = R_theta @ (tube[1].params['Kbt'] @ (theta_ds*R_theta_dtheta@u-u_i_star_ds) + (hat(u) @ tube[1].params['Kbt']) @ (u_i - u_i_star)) \
                    - ( np.dot(hat(tube[1]._e3[0]) @ R.T, self.step_len * np.asarray([n]).T).T[0] + R.T @ m) # external

            tube_uzs[tube[0]]= u_div[2]

            #tube_z_torsions.append((tube[0],
            # u_div[2]))
            #+ (EI / JG) * (u[0] * u_i_star[1] - u[1] * u_i_star[0])))
            #+ (1 / JG) * (u_i_star[2]-u[2]) is zero because we assume G and J are constant
            #- (1 / JG) * (tube[1]._e3 @ (R @ R_theta) @ m))) # external

            #print(EI/JG)
            #print(1/JG)

            new_u_s[:2] += u_div[:2] # TODO what does 3 mean?

        new_u_s[2:3] = tube_uzs[0]

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
                          tube_uzs])

    def set_bounding_values(self, names, values):
        if self.bounding_values is None:
            self.bounding_values = {}
        for i, name in enumerate(names):
            self.bounding_values[name] = values[i]

    def rotate(self, alphas):
        for i in range(len(self.tubes)):
            self.tubes[i][1].params["alpha"] = alphas[i]
            self.alphas[i] = alphas[i]
        self.tubes[0][1].inital_conditions['R0'] = np.array([[np.cos(alphas[0]), -np.sin(alphas[0]), 0],
                  [np.sin(alphas[0]), np.cos(alphas[0]), 0],
                  [0, 0, 1],
                  ])

    def translate(self, betas):
        if self._check_beta_validity(betas):
            for i in range(len(self.tubes)):
                self.tubes[i][1].params["beta"] = betas[i]
                self.betas[i] = betas[i]
        else:
            raise Exception('Parameter Error', 'The beta values do not correspond to the specifications!')

    def _check_beta_validity(self, betas):
        valid = True
        for i in range(1, np.asarray(betas).shape[0]):
            #valid = valid and betas[i - 1] <= betas[i]
            #print(valid)
            valid = valid and self.tubes[i][1].params['L'] -betas[i] * self.tubes[i][1].params['L'] <= self.tubes[i - 1][1].params['L'] -  betas[i - 1] * self.tubes[i - 1][1].params['L']
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


class CTCRExternalForces(ConcentricTubeContinuumRobot):

    def __init__(self,tubes):
        super().__init__(tubes)


    def cosserate_rod_ode(self, s, state):
        R = np.reshape(state[3:12], (3, 3))
        n = state[12:15]
        m = state[15:18]
        uzs = []
        thetas = []
        for i in range(len(self._curr_calc_tubes)):
            uzs.append(state[18+i:18+(i+1)].item())

        for i in range(len(self._curr_calc_tubes)):
            thetas.append(state[18+len(self.tubes)+i:18+len(self.tubes)+(i+1)].item())

        EIk = 0
        for i in range(len(self._curr_calc_tubes)):
            EIk += self.tubes[i][1].params['Kbt'][0,0]

        RthetaEkIkuj_star = np.zeros((3,1)) # PhD Rucker page 91 eq 3.56 last part

        for i in range(len(self._curr_calc_tubes)):
            Rtheta =  np.array([[np.cos(thetas[i]), -np.sin(thetas[i]), 0],
                                     [np.sin(thetas[i]), np.cos(thetas[i]), 0],
                                     [0, 0, 1],
                                     ])
            EIj = self.tubes[i][1].params['Kbt'][0, 0]


            curved_part = self.tubes[i][1].is_curved_or_at_end(s)
            tube_curvature = self.tubes[i][1].params["kappa"] * curved_part  # also weired TODO
            uj_star = self.get_curvature_vector(tube_curvature)
            RthetaEkIkuj_star += EIj*Rtheta@uj_star.reshape(-1,1)


        u1 = (1/EIk * (m + RthetaEkIkuj_star.T))

        uixy = []
        for i in range(1,len(self._curr_calc_tubes)):
            Rtheta =  np.array([[np.cos(thetas[i]), -np.sin(thetas[i]), 0],
                                     [np.sin(thetas[i]), np.cos(thetas[i]), 0],
                                     [0, 0, 1],
                                     ])
            uixy.append((Rtheta@u1.T).T[0,:2])
        uixy = [u1[0,:2]] + uixy

        external_forces = self.tubes[0][1].external_gauss_forces(s)

        #ode
        ps = R@self.tubes[0][1]._e3.T
        Rs = R@hat(u1.T)
        thetas_s = []
        for i in range(len(self._curr_calc_tubes)):
            thetas_s.append(uzs[i]-uzs[0])

        uiz_s = []
        for i in range(1, len(self._curr_calc_tubes)):
            curved_part = self.tubes[i][1].is_curved_or_at_end(s)
            tube_curvature = self.tubes[i][1].params["kappa"] * curved_part  # also weired TODO
            ui_star = self.get_curvature_vector(tube_curvature)
            EIi = self.tubes[i][1].params['Kbt'][0, 0]
            GJi = self.tubes[i][1].params['Kbt'][2, 2]
            uiz_s.append(ui_star[2] + EIi/GJi * (uixy[i][0]*ui_star[1] - uixy[i][1]*ui_star[0])) # - 1/GJi)
        uiz_s = [u1[0,2]] + uiz_s

        uzs = np.zeros(len(self.tubes))
        thetas = np.zeros(len(self.tubes))
        for i in range(len(self._curr_calc_tubes)):
            uzs[i] = uiz_s[i]
            thetas[i] = thetas_s[i]

        ns = -R@external_forces
        msbxy = -hat(u1.T)@m.T-hat(self.tubes[0][1]._e3.T)@R.T@n # - R.T@l

        return np.hstack([ps.T.flatten(), Rs.reshape((1,9)).flatten(), ns.flatten(), msbxy.flatten(), np.hstack(uzs), np.hstack(thetas)])


    # def cosserate_rod_ode(self, s, state):
    #     R = np.reshape(state[3:12], (3, 3))
    #     n = state[12:15]
    #     m = state[15:18]
    #     u = state[18:21]
    #
    #     new_u_s = np.zeros(3)
    #     summed_K = np.zeros((3, 3)) # see Rucker and Webster
    #
    #     tube_z_torsions = []
    #     tube_uzs = np.zeros(len(self.tubes)).tolist()
    #     for tube in self._curr_calc_tubes:
    #         curved_part = tube[1].is_curved_or_at_end(s) # returns 1 for curved, 0 for not curved and -1 if s > L_t. Last one should not occure at this point
    #
    #         theta = self.alphas[tube[0]] - self.alphas[self.tubes[0][0]]
    #         R_theta = np.array([[np.cos(theta), -np.sin(theta), 0],
    #                                 [np.sin(theta), np.cos(theta), 0],
    #                                 [0, 0, 1],
    #                                 ])
    #         R_theta_dtheta = np.array([[-np.sin(theta), -np.cos(theta), 0],
    #                                 [np.cos(theta), -np.sin(theta), 0],
    #                                 [0, 0, 0],
    #                                 ])
    #
    #         theta_ds = state[21+tube[0]] - state[21]
    #
    #         summed_K += tube[1].params['Kbt'] # adding the Ks
    #
    #         EI = tube[1].params['E'] * tube[1].params['I']
    #         JG = tube[1].params['G'] * tube[1].params['J']
    #
    #         tube_curvature = tube[1].params["kappa"] * curved_part # also weired TODO
    #         u_tube = self.get_curvature_vector(tube_curvature)
    #
    #         u_i_star_ds = invhat((R @ R_theta) @ hat(u_tube))
    #         u_i_star = invhat((R @ R_theta).T @ hat(u_i_star_ds))
    #
    #         u_i = u.copy()
    #         u_i[2] = state[21+tube[0]]
    #
    #         u_div = R_theta @ (tube[1].params['Kbt'] @ (theta_ds*R_theta_dtheta@u-u_i_star_ds) + (hat(u) @ tube[1].params['Kbt']) @ (u_i - u_i_star)) \
    #                 - (np.dot(hat(tube[1]._e3[0]) @ R.T, self.step_len * np.asarray([n]).T).T[0] + R.T @ m)  # external
    #
    #         tube_uzs[tube[0]]= u_div[2]
    #
    #         new_u_s[:2] += u_div[:2] # TODO what does 3 mean?
    #
    #     new_u_s[2:3] = tube_uzs[0]
    #
    #     new_u_s = np.linalg.inv(summed_K) @ new_u_s # TODO dimension missmatch
    #     ns = np.sum([-self.tubes[i][1].params['rho'] * self.tubes[i][1].params['A'] * self.tubes[i][1].params['g'].T for i in
    #                  range(len(self._curr_calc_tubes))], axis=0) - self.external_gauss_forces(s)
    #     ps = R.dot(np.array([[0, 0, 1]]).T)
    #     Rs = R.dot(hat(new_u_s))
    #     ms = -np.cross(ps.T[0], n)
    #
    #     return np.hstack([ps.T[0],
    #                       np.reshape(Rs, (1, 9))[0],
    #                       ns.T[0],
    #                       ms,
    #                       new_u_s,
    #                       tube_uzs])

    def external_gauss_forces(self, s):

        fx = np.sum([self._gaussians[i][0][0]*np.exp(-self._gaussians[i][1]*(s-self._gaussians[i][2])**2) for i in range(len(self._gaussians))])
        fy = np.sum([self._gaussians[i][0][1] * np.exp(-self._gaussians[i][1] * (s - self._gaussians[i][2]) ** 2) for i in range(len(self._gaussians))])
        return np.array([[fx, fy, 0]]).T

    def fwd_external_gaussian_forces(self, gaussians, step_size=0.01, pos_indexes=False):
        self._gaussians = gaussians
        self.tubes[0][1]._gaussians = gaussians

        wrench = np.zeros(6)
        solution_bvp = least_squares(self.shooting_function_external_force, wrench, method='lm', loss='linear',
                                     ftol=1e-6)
        state, pos_indexes = self._apply_fwd_static(solution_bvp.x,step_size)
        positions = state[:, :3]
        orientations = state[:, 3:12]
        wrenches = state[:,12:18]
        if pos_indexes:
            return positions, orientations, wrenches, pos_indexes
        else:
            return positions, orientations, wrenches


    def _apply_fwd_static(self, wrench, step_size=0.01):
        state, pos_indexes = self.calc_forward(self.tubes[0][1].inital_conditions['R0'], self.tubes[0][1].inital_conditions['p0'],
                                  np.asarray(wrench), step_size)
        return state, pos_indexes

    def calc_forward(self, R_init, p_init, wrench, step_len):
        R = R_init
        p = p_init[0]
        self.step_len = step_len
        w = wrench

        segment_list = self.get_ordered_segments()
        ode_returns = []
        thetas = []
        for i in range(len(self.tubes)):
            thetas.append(self.alphas[self.tubes[i][0]] - self.alphas[self.tubes[0][0]])
        uzs = np.zeros(len(self.tubes))
        position_indexes = []
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
             n*thetas - difference in rotation
            """

            state = np.hstack([p, R.reshape((1, 9))[0], w, uzs, thetas]) # guess

            ode_states = integrate.solve_ivp(self.cosserate_rod_ode, (
                segment_list[i-1], segment_list[i]), state, dense_output=True, max_step=step_len) # beta is defined 0-1 -> L-L*beta  if beta 0 -> fully elongated tube
            p = ode_states.y.T[-1:,:3][0]
            R = np.reshape(ode_states.y.T[-1:,3:12], (3, 3))
            w = ode_states.y.T[-1,12:18]
            uzs = ode_states.y.T[-1,18:18+len(self.tubes)]
            thetas = ode_states.y.T[-1,18+len(self.tubes):]
            ode_returns.append(ode_states.y.T)
            position_indexes.append(ode_states.y.T[:,:3].shape[0])
        return np.vstack(ode_returns), position_indexes


    def shooting_function_external_force(self, guess, s_l = 0.01):
        n0 = guess[:3]
        m0 = guess[3:6]
        tip_wrench = np.zeros(6)
        states, _ = self._apply_fwd_static(np.hstack([n0, m0]), s_l)
        tip_wrench_shooting = states[-1,12:18]

        return np.hstack([(tip_wrench - tip_wrench_shooting)**2])