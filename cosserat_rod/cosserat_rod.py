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
        u = state[18:21]
        v = np.dot(np.linalg.inv(self.params['Kse']).dot(R.T), n) + np.array([[0, 0, 1]])  # TODO research
        u = np.dot(np.linalg.inv(self.params['Kbt']).dot(R.T), m)  # TODO research
        # ode
        ps = R.dot(v.T)

        Rs = R.dot(hat(u))
        ns = -self.params['rho'] * self.params['A'] * self.params['g'].T
        ms = -np.cross(ps.T[0], n)
        return np.hstack([ps.T[0], np.reshape(Rs, (1, 9))[0], ns.T[0], ms, u])

    def shooting_function(self, guess):
        s = np.linspace(0, self.params['L'], 100)

        n0 = guess[:3]
        m0 = guess[3:6]
        if self.params['kappa_0'] is None:
            u = np.zeros(3)
        else:
            u = self.params['kappa_0']

        y0 = np.hstack([self.inital_conditions['p0'][0], np.reshape(self.inital_conditions['R0'],(1,9))[0], n0, m0, u])
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
        if not ('kappa_0' in self.inital_conditions.keys()):
            u = np.zeros(3)
        else:
            u = self.inital_conditions['kappa_0']
        y0 = np.hstack([self.inital_conditions['p0'][0], np.reshape(self.inital_conditions['R0'],(1,9))[0], n0, m0, u])
        tip_wrench = self.bounding_values['tip_wrench']

        states = integrate.odeint(self.cosserate_rod_ode, y0, s)
        tip_wrench_shooting = states[-1][-6:]

        distal_force_error = tip_wrench[:3] - tip_wrench_shooting[:3]
        distal_moment_error = invhat(hat(tip_wrench[3:]).T.dot(hat(tip_wrench_shooting[3:])) - hat(tip_wrench[3:]).dot(hat(tip_wrench_shooting[3:]).T))
        return np.hstack([distal_force_error, distal_moment_error])

    def push_end(self, wrench):
        self.set_bounding_values(['tip_wrench'], [wrench])
        state = np.zeros((1, 6))
        solution_bvp = least_squares(self.shooting_function_force, state[0], method='lm', loss = 'linear', ftol=1e-6)
        states = self.apply_force(solution_bvp.x)
        return states

    def move_end(self, p, R):
        self.set_bounding_values(p, R)
        p0, R0 = self.inital_conditions['p0'], self.inital_conditions['R0']
        state = np.zeros((1, 6))

        start = timeit.default_timer()

        solution_bvp = least_squares(self.shooting_function, state[0], method='lm', loss = 'linear')

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


class CurvedCosseratRod(CosseratRod):
    """
    parent class for all rods
    """

    def __init__(self, params=None):
        super().__init__(params)

        if params is None:
            self.params['kappa'] = np.array([0,0.16,0])

        self._e3 = np.array([[0,0,1]])

    def curvature_ode(self, state, s):
        n = state[0:3]
        m = state[3:6]
        u = state[6:9]
        R = np.reshape(state[9:18], (3, 3))
        step_size = state[18:19]
        u = -np.linalg.inv(self.params['Kbt']) @ (
                (hat(u) @ self.params['Kbt']) @ (u - self.params['kappa']) + R.T @ m)

        state[6:9] = u
        return state

    def cosserate_rod_ode(self, state, s):
        R = np.reshape(state[3:12], (3, 3))
        # R_k = hat(np.array([0,0,self.params['k']]))
        # R = R @ R_k
        n = state[12:15]
        m = state[15:18]
        u = state[18:21]

        u_div = -np.linalg.inv(self.params['Kbt']) @ (
                (hat(u) @ self.params['Kbt']) @ (u - self.kappa(s)) + R.T @ m)
        #u_s = np.linspace(0, step_size[0], 10)
        #u_state = np.hstack([n,m,u,state[3:12],step_size])
        #u_states = integrate.odeint(self.curvature_ode, u_state, u_s)
        u = u + u_div #u_states[-1,6:9]
        #print(u_states[:,6:9])

        ps = R.dot(self._e3.T)
        Rs = R.dot(hat(u))
        ns = -self.params['rho'] * self.params['A'] * self.params['g'].T
        ms = -np.cross(ps.T[0], n)
        return np.hstack([ps.T[0], np.reshape(Rs, (1, 9))[0], ns.T[0], ms, u])


    def kappa(self, s):
        #print(s)
        #if s < self.params['straight_length']:
        return np.array([0, self.params['kappa'], 0])
        #else:
        #    return np.zeros(3)


    def apply_force(self, wrench, step_size= 100):
        #s_l = self.params['L']/step_size
        p0, R0 = self.inital_conditions['p0'], self.inital_conditions['R0']
        kappa_0 = self.inital_conditions['kappa_0']
        s = np.linspace(0, self.params['L'], step_size)
        start = timeit.default_timer()
        state = np.hstack([p0[0], R0.reshape((1, 9))[0], wrench, kappa_0])
        states = integrate.odeint(self.cosserate_rod_ode, state, s)
        stop = timeit.default_timer()
        #print('Time: ', stop - start)
        return states


def plot_frame(ax, R, p):
    ex = R @ np.array([[0.1, 0, 0]]).T
    ey = R @ np.array([[0, 0.1, 0]]).T
    ez = R @ np.array([[0, 0, 0.1]]).T
    x = np.hstack([p.T, p.T + ex])
    y = np.hstack([p.T, p.T + ey])
    z = np.hstack([p.T, p.T + ez])
    ax.plot(x[0, :], x[1, :], x[2, :], color='b')
    ax.plot(y[0, :], y[1, :], y[2, :], color='g')
    ax.plot(z[0, :], z[1, :], z[2, :], color='r')
    return ax

if __name__ == '__main__':
    rod = CurvedCosseratRod()
    # Arbitrary base frame assignment
    p0 = np.array([[0,0,0]])
    R0 = np.eye(3)

    rod.set_initial_conditions(p0, R0)
    rod.params['L'] = 1.

    kappa = np.deg2rad(90) / rod.params['L']
    print(kappa)
    rod.inital_conditions['kappa_0'] = np.array([0, kappa, 0])

    rod.params['kappa'] = 0 #kappa
    rod.params['straight_length'] = 0.5


    states = rod.push_end(np.array([0,0, 0,0,0,0]))
    Rn0 = np.reshape(states[-1][3:12], (3, 3))
    pn0 = np.asarray([states[-1][:3]])

    ax = plt.figure().add_subplot(projection='3d')
    x_vals, y_vals, z_vals = states[:, 0], states[:, 1], states[:, 2]
    ax.plot(x_vals, y_vals, z_vals, label='parametric curve')
    plot_frame(ax, Rn0, pn0)

    for p in np.linspace(0.2, 1, 1):
        f = 0 #0.2*p
        r = 0 #np.pi/2 * p
        R0 = np.array([[np.cos(r), -np.sin(r), 0],
                       [np.sin(r), np.cos(r), 0],
                       [0, 0, 1]])
        rod.set_initial_conditions(p0, R0)
        rod.inital_conditions['kappa_0'] = np.array([0, kappa, 0])
        states = rod.push_end(np.array([0, 0, 0, 0, 0, 0]))
        Rn0 = np.reshape(states[-1][3:12], (3, 3))
        wrench = np.array([0, -f, 0, 0, 0, 0])
        wrench[:3] = Rn0 @ wrench[:3]
        wrench[3:] = Rn0 @ wrench[3:]
        states = rod.push_end(wrench)
        p = np.asarray([states[-1][:3]])
        R = np.reshape(states[-1][3:12], (3, 3))
        plot_frame(ax, R, p)

        x_vals, y_vals, z_vals = states[:, 0], states[:, 1], states[:, 2]
        ax.plot(x_vals, y_vals, z_vals, label='parametric curve')
    plot_frame(ax, R0, p0)
    ax.set_zlabel('Z')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set(xlim=[-0.2,0.2],ylim=[-0.2,0.2],zlim=[-0.2,0.2])
    plt.show()
