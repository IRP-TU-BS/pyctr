import pdb
import timeit
import numpy as np
import scipy as sc
from scipy import integrate
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from .utils import *


class CosseratRod:
    """
    parent class for all rods
    """

    def __init__(self, params=None):
        self.params = {
            'E': 339889091.1107947  # Young's modulus - Nylon Taulmann3d 618 Nylon
            , 'G': 97898219.42862949  # Shear modulus - Nylon Taulmann3d 618 Nylon
            , 'r_inner': 0.  # Cross-sectional radius
            , 'r_outer': 1.4 * 1e-3 # m
            , 'rho': 8000  # Density - wrong
            , 'g': self._make_garvitational_vec(9.81, 'z')  # Gravitational acceleration
            , 'L': 150 * 1e-3  # Length (before s')'
            ,'straight_length': 0.
            , 's': 1e-3  # arc lenght steps
            , 'kappa': 0
        }
        if not (params is None):
            for key in params.keys():
                if key == 'g':
                    self.params[key] = self._make_garvitational_vec( params[key], 'z')
                else:
                    self.params[key] = params[key]

        self.params['beta'] = 0.
        self.params['alpha'] = 0.

        self.params['A'] = np.pi * (self.params['r_outer'] ** 2 -self.params['r_inner'] ** 2)  # Cross-sectional area
        self.params['I'] = np.pi/4 * (self.params['r_outer'] ** 4 - self.params['r_inner'] ** 4)  # Area moment of inertia
        self.params['J'] = (np.pi * ((2*self.params['r_outer']) ** 4 - (2*self.params['r_inner']) ** 4))/32.  # Polar moment of inertia

        self.params['Kse'] = np.diag([self.params['G'] * self.params['A'], self.params['G'] * self.params['A'],
                                      self.params['E'] * self.params['A']])  # Stiffness matrices
        self.params['Kbt'] = np.diag([self.params['E'] * self.params['I'], self.params['E'] * self.params['I'],
                                      self.params['G'] * self.params['J']])
        #self.params['Kbt'] = np.diag([0.008929444846481, 0.009185881509851,
        #                              0.000644390038492])

        self._e3 = np.array([[0, 0, 1]])  # define z-axis as coinciding with curve
        self.bounding_values = None

    def _make_garvitational_vec(self, g, dir):
        switch = {
            'x': np.array([[g, 0, 0]]),
            'y': np.array([[0, g, 0]]),
            'z': np.array([[0, 0, g]]),
        }
        return switch.get(dir, "Invalid input")

    def set_initial_conditions(self, p0, R0):
        self.inital_conditions = {
            'p0': p0,
            'R0': R0
        }
        # always sets thirs basis vector as up
        self._e3 = R0[:, 2:3].T

    def set_bounding_values(self, names, values):
        if self.bounding_values is None:
            self.bounding_values = {}
        for i, name in enumerate(names):
            self.bounding_values[name] = values[i]

    def get_u_div(self, R, n, m, u):
        return np.dot(np.linalg.inv(self.params['Kbt']).dot(R.T), m)

    def cosserate_rod_ode(self, state, s):
        R = np.reshape(state[3:12], (3, 3))
        n = state[12:15].reshape(3,-1)
        m = state[15:].reshape(3,-1)

        #v = np.dot(np.linalg.inv(self.params['Kse']).dot(R.T), n) + np.array([[0, 0, 1]])  # Not used here !
        u = np.dot(np.linalg.inv(self.params['Kbt']).dot(R.T), m) #+
        # ode

        ps = R.dot(self._e3.T)
        Rs = R.dot(hat(u))
        ns = -self.params['rho'] * self.params['A'] * self.params['g'].T
        ms = -np.cross(ps.T, n.T)
        return np.hstack([ps.T[0], np.reshape(Rs, (1, 9))[0], ns.T[0], ms.flatten()])

    def cosserat_rod_ode_curvature(self, state, s):
        R = np.reshape(state[3:12], (3, 3))
        n = state[12:15].reshape(3,-1)
        u = state[15:].reshape(3,-1)
        ps = R.dot(self._e3.T)
        Rs = R.dot(hat(u))
        ns = -self.params['rho'] * self.params['A'] * self.params['g'].T
        us = -np.linalg.inv(self.params['Kbt']) @ (hat(u) @ (self.params['Kbt'] @ u) + hat(self._e3.T) @ R.T @ n)
        return np.hstack([ps.T[0], np.reshape(Rs, (1, 9))[0], ns.T[0], us.flatten()])

    def external_gauss_forces(self, s):
        return np.array([[0, 0, 0]]).T # TODO workaround - should be called external forces and should be modular

    def shooting_function(self, guess):
        s = np.linspace(0, self.params['L'], 100)

        n0 = guess[:3]
        m0 = guess[3:6]
        if self.params['kappa_0'] is None:
            u = np.zeros(3)
        else:
            u = self.params['kappa_0']

        y0 = np.hstack(
            [self.inital_conditions['p0'][0], np.reshape(self.inital_conditions['R0'], (1, 9))[0], n0, m0, u])
        pL = self.bounding_values['pL']
        RL = self.bounding_values['RL']

        states = integrate.odeint(self.cosserate_rod_ode, y0, s)
        pL_shooting = states[-1][:3]
        RL_shooting = np.reshape(states[-1][3:12], (3, 3))

        distal_pos_error = pL - pL_shooting
        distal_rot_error = invhat(RL.T.dot(RL_shooting) - RL.dot(RL_shooting.T))

        return np.hstack([distal_pos_error[0], distal_rot_error])

    def shooting_function_force(self, guess, curvature_integration):

        n0 = guess[:3]
        m0 = guess[3:6]
        tip_wrench = self.bounding_values['tip_wrench']

        steps = int(np.ceil(self.params['L'] / self.params['s']))

        p, R, w = self.apply_force(np.hstack([n0, m0]), steps, curvature_integration)
        tip_wrench_shooting = w[-1]

        distal_force_error = tip_wrench[:3] - tip_wrench_shooting[:3]
        distal_moment_error = tip_wrench[3:] - tip_wrench_shooting[3:]
        print(distal_force_error)
        #distal_moment_error = invhat(hat(tip_wrench[3:]).T.dot(hat(tip_wrench_shooting[3:])) - hat(tip_wrench[3:]).dot(
        #    hat(tip_wrench_shooting[3:]).T))
        return np.hstack([distal_force_error, distal_moment_error])

    def push_end(self, wrench, curvature_integration=False):
        self.set_bounding_values(['tip_wrench'], [wrench])
        state = np.zeros((1, 6))
        solution_bvp = least_squares(self.shooting_function_force, state[0], method='lm', loss='linear', ftol=1e-6, args=(curvature_integration, ))
        steps = int(np.ceil(self.params['L'] / self.params['s']))
        states = self.apply_force(solution_bvp.x, steps, curvature_integration)
        return states

    def move_end(self, p, R):
        self.set_bounding_values(p, R)
        p0, R0 = self.inital_conditions['p0'], self.inital_conditions['R0']
        state = np.zeros((1, 6))

        start = timeit.default_timer()

        solution_bvp = least_squares(self.shooting_function, state[0], method='lm', loss='linear')

        stop = timeit.default_timer()
        # print('Time: ', stop - start)

        p, R, w = self.apply_force(solution_bvp.x)

        return p, R, w

    def calc_m_from_curvature(self, R, u):
        m = np.zeros((R.shape[0],3))
        for i in range(R.shape[0]):
            m[i,:] = (R[i].reshape(3,3)@self.params['Kbt']@u[i].reshape(3,-1)).flatten() # hookes law
        return m

    def calc_u_from_m(self, R, m, u_star=None):
        u = np.zeros((R.shape[0],3))
        for i in range(R.shape[0]):
            u[i,:] = (np.linalg.inv(self.params['Kbt']) @ R[i].reshape(3,3).T @ m[i].reshape(3,-1)).flatten() # hookes law
        return u

    def apply_force(self, wrench, s_l=20, curvature_integration = True):
        p0, R0 = self.inital_conditions['p0'], self.inital_conditions['R0']
        s = np.linspace(0, self.params['L'], s_l)
        start = timeit.default_timer()
        if curvature_integration:
            u = self.calc_u_from_m(np.asarray([R0]), np.asarray([wrench[3:]]))
            state = np.hstack([p0[0], R0.reshape((1, 9))[0], wrench[0:3], u.flatten()])
            states = integrate.odeint(self.cosserat_rod_ode_curvature, state, s)
            p = states[:,:3]
            R = states[:,3:3+9]
            n = states[:,3+9:3+9+3]
            u = states[:,3+9+3:]
            m = self.calc_m_from_curvature(R, u)
            wrench = np.hstack([n,m])
        else:
            state = np.hstack([p0[0], R0.reshape((1, 9))[0], wrench])
            states = integrate.odeint(self.cosserate_rod_ode, state, s)
            p = states[:,:3]
            R = states[:,3:3+9]
            wrench = states[:,3+9:]
        stop = timeit.default_timer()
        return p, R, wrench

    def is_curved_or_at_end(self, s):
        if s <= self.params['straight_length'] - self.params['straight_length'] * self.params['beta']:
            return 0
        elif s > self.params['L'] - self.params['L'] * self.params['beta']:
            return -1

    def get_kappa(self):
        return np.array([0, self.cur_kappa, 0])

    def set_kappa(self, kappa):
        self.cur_kappa = kappa


class CurvedCosseratRod(CosseratRod):
    """
    Kirchoff Rod with curvature
    """

    def __init__(self, params=None):
        super().__init__(params)

        if params is None:
            self.params['kappa'] = np.array([0, 0.16, 0])

        self._e3 = np.array([[0, 0, 1]])  # define z-axis as coinciding with curve
        self._step_size = None

    # def curvature_ode(self, state, s):
    #     n = state[0:3]
    #     m = state[3:6]
    #     u = state[6:9]
    #     R = np.reshape(state[9:18], (3, 3))
    #     step_size = state[18:19]
    #     u = -np.linalg.inv(self.params['Kbt']) @ (
    #             (hat(u) @ self.params['Kbt']) @ (u - self.params['kappa']) + R.T @ m)
    #
    #     state[6:9] = u
    #     return state

    # def get_u_div(self, R, n, m, u):
    #     return -np.linalg.inv(self.params['Kbt']) @ (
    #             (hat(u) @ self.params['Kbt']) @ (u - self.get_kappa()) - (
    #                 np.dot(hat(self._e3[0]) @ R.T, self._step_size * np.asarray([n]).T).T[0] + R.T @ m))


    def cosserate_rod_ode(self, state, s):
        R = np.reshape(state[3:12], (3, 3))
        n = state[12:15]
        m = state[15:]
        u = np.dot(np.linalg.inv(self.params['Kbt']).dot(R.T), m) + self.get_kappa()  # TODO research

        # ode
        ps = R.dot(self._e3.T)
        Rs = R.dot(hat(u))
        ns = -self.params['rho'] * self.params['A'] * self.params['g'].T
        ms = -np.cross(ps.T[0], n)  # -l = 0

        return np.hstack([ps.T[0], np.reshape(Rs, (1, 9))[0], ns.T[0], ms])

    def get_kappa(self):
        return np.array([0, self.cur_kappa, 0])

    def set_kappa(self, kappa):
        self.cur_kappa = kappa

    def is_curved_or_at_end(self, s):
        if s <= self.params['straight_length'] - self.params['straight_length'] * self.params['beta']:
             return 0
        elif s > self.params['L'] - self.params['L'] * self.params['beta']:
             return -1
        else:
             return 1


    def apply_force(self, wrench, steps=100):
        self._step_size = self.params['L'] / steps # ?
        step_size_straight = int(steps * self.params['straight_length'] / self.params['L']) # get how many steps we need to integrate the singel tube
        step_size_curved = int(steps * (self.params['L'] - self.params['straight_length']) / self.params['L']) # get curved steps as we use ODE
        # straight part
        p0, R0 = self.inital_conditions['p0'], self.inital_conditions['R0'] # get start from initial condition
        curved_kappa = self.params['kappa']
        states = None
        start_integration_length = 0
        if step_size_straight > 0:
            self.set_kappa(0) # set kappa to 0 as we are in the straight part of the tube
            state = np.hstack([p0.flatten(), R0.reshape((1, 9)).flatten(), wrench]) # initial state
            s = np.linspace(0, self.params['straight_length'], step_size_straight)
            straight_states = integrate.odeint(self.cosserate_rod_ode, state, s) # integrate over s steps
            p0 = straight_states[-1][:3]
            R0 = straight_states[-1][3:12]
            wrench = straight_states[-1][12:18]
            states = straight_states
            start_integration_length = self.params['straight_length']
        self.set_kappa(curved_kappa) # set kappa to curved parts curvature
        state = np.hstack([p0.flatten(), R0.reshape((1, 9)).flatten(), wrench]) # take state from straight parts last step
        s = np.linspace(start_integration_length, self.params['L'], step_size_curved)
        curved_states = integrate.odeint(self.cosserate_rod_ode, state, s)
        if not (states is None):
            states = np.vstack([states, curved_states])
        else:
            states = curved_states
        return states




