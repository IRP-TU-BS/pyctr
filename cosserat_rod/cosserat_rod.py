import timeit
import numpy as np
import scipy as sc
from scipy import integrate
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from utils import *

class CosseratRod:
    """
    parent class for all rods
    """
    def __init__(self, params=None):
        if params is None:
            self.params = {
            'E' :  200e9  # Young's modulus
            ,'G' : 80e9  # Shear modulus
            ,'r' : 0.001  # Cross-sectional radius
            ,'rho':  8000  # Density
            ,'g' : self._make_garvitational_vec(9.81, 'z')  # Gravitational acceleration
            ,'L' : 5.0  # Length (before s')'
            ,'s' : 0.2  # arc lenght steps
            ,'k': 0.16
            }

        self.params['A'] = np.pi * self.params['r'] ** 2  # Cross-sectional area
        self.params['I'] = np.pi * self.params['r'] ** 4 / 4  # Area moment of inertia
        self.params['J'] = 2 * self.params['I']  # Polar moment of inertia

        self.params['Kse'] = np.diag([self.params['G'] * self.params['A'], self.params['G'] * self.params['A'], self.params['E'] * self.params['A']])  # Stiffness matrices
        self.params['Kbt'] = np.diag([self.params['E'] * self.params['I'], self.params['E'] * self.params['I'], self.params['G'] * self.params['J']])

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

    def set_bounding_values(self, names, values):
        self.bounding_values = {}
        for i,name in enumerate(names):
            self.bounding_values[name] = values[i]

    def cosserate_rod_ode(self, state, s):
        R = np.reshape(state[3:12], (3, 3))
        #R_k = hat(np.array([0,0,self.params['k']]))
        #R = R @ R_k
        n = state[12:15]
        m = state[15:]

        v = np.dot(np.linalg.inv(self.params['Kse']).dot(R.T), n) + np.array([[0, 0, 1]])  # TODO research
        u = np.dot(np.linalg.inv(self.params['Kbt']).dot(R.T), m)  # TODO research

        # ode
        ps = R.dot(v.T)

        Rs = R.dot(hat(u))
        ns = -self.params['rho'] * self.params['A'] * self.params['g'].T
        ms = -np.cross(ps.T[0], n)
        return np.hstack([ps.T[0], np.reshape(Rs, (1, 9))[0], ns.T[0], ms])

    def shooting_function(self, guess):
        s = np.linspace(0, self.params['L'], 100)

        n0 = guess[:3]
        m0 = guess[3:6]
        y0 = np.hstack([self.inital_conditions['p0'][0], np.reshape(self.inital_conditions['R0'],(1,9))[0], n0, m0])
        pL = self.bounding_values['pL']
        RL = self.bounding_values['RL']

        states = integrate.odeint(self.cosserate_rod_ode, y0, s)
        pL_shooting = states[-1][:3]
        RL_shooting = np.reshape(states[-1][3:12], (3, 3))

        distal_pos_error = pL - pL_shooting
        distal_rot_error = invhat(RL.T.dot(RL_shooting) - RL.dot(RL_shooting.T))

        return np.hstack([distal_pos_error[0], distal_rot_error])

    def shooting_function_force(self, guess):
        s = np.linspace(0, self.params['L'], 100)

        n0 = guess[:3]
        m0 = guess[3:6]
        y0 = np.hstack([self.inital_conditions['p0'][0], np.reshape(self.inital_conditions['R0'],(1,9))[0], n0, m0])
        tip_wrench = self.bounding_values['tip_wrench']

        states = integrate.odeint(self.cosserate_rod_ode, y0, s)
        tip_wrench_shooting = states[-1][-6:]

        distal_force_error = tip_wrench - tip_wrench_shooting

        return np.hstack([distal_force_error])

    def push_end(self, wrench):
        self.set_bounding_values(['tip_wrench'], [wrench])
        state = np.zeros((1, 6))
        solution_bvp = least_squares(self.shooting_function_force, state[0])
        states = self.apply_force(solution_bvp.x)
        return states

    def move_end(self, p, R):
        self.set_bounding_values(p, R)
        p0, R0 = self.inital_conditions['p0'], self.inital_conditions['R0']
        state = np.zeros((1, 6))

        start = timeit.default_timer()

        solution_bvp = least_squares(self.shooting_function, state[0])

        stop = timeit.default_timer()
        #print('Time: ', stop - start)

        states = self.apply_force(solution_bvp.x)

        return states

    def apply_force(self, wrench, s_l = 20):
        p0, R0 = self.inital_conditions['p0'], self.inital_conditions['R0']
        s = np.linspace(0, self.params['L'], s_l)
        start = timeit.default_timer()
        state = np.hstack([p0[0], R0.reshape((1, 9))[0], wrench])
        #print(state)
        states = integrate.odeint(self.cosserate_rod_ode, state, s)
        stop = timeit.default_timer()
        #print('Time: ', stop - start)
        return states

if __name__ == '__main__':
    rod = CosseratRod()
    # Arbitrary base frame assignment
    #L = rod.params['L']
    p0 = np.array([[0,0,0]])
    R0 = np.eye(3)
    #n0 = np.array([[0, 0, 0]])
    #m0 = np.array([[0, 0, 0]])
    #pL = np.array([[0,0.3,0.9*L]])
    #RL = np.eye(3)

    rod.set_initial_conditions(p0, R0)

    states = rod.push_end(np.array([0,0.0,-0.1,0,0,0]))
    #states = rod.apply_force(np.array([0,0.2,0,0,0,0]))
    plt.axis('scaled')
    for f in np.linspace(0, -2, 20):
        wrench = np.array([0, 0, f, 0, 0, 0])
        states = rod.push_end(wrench)
        x_vals, y_vals = states[:, 0], states[:, 2]
        plt.plot(x_vals, y_vals)

    #plt.plot(states[:,1],states[:,2])
    #plt.show()
    #plt.plot(states[:,0],states[:,1])
    plt.show()