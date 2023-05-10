"""
Robot modules contains all necessary classes and functions to simulate a continous version of a CTCR
"""

import pdb
import timeit
import numpy as np
import scipy as sc
from scipy import integrate
from scipy.optimize import least_squares

from rich.console import Console
from rich.table import Table

from .utils import *

from typing import Union, List, Callable
import numpy.typing as npt


class ConcentricTubeContinuumRobot:
    """
    The main robot class, which combines mutliple tubes to a single robot with the same configuration space
    """

    def __init__(self, tubes: list, R_init: npt.NDArray = np.identity(3), p_init: npt.NDArray = np.zeros((3, 1))):
        """

        :param tubes: list of tubes in the robot sorted by inner to outer, longest -> shortest
        :param R_init: orientation of base frame as SO(3) matrix
        :param p_init: position of base frame in R^3
        """
        self.tubes = list(zip(range(len(tubes)), tubes))  # zip with number to have an order
        self.R_init = R_init  # public variable. Can change during runtime
        self.p_init = p_init  # public variable. Can change during runtime
        self.alphas = np.zeros(len(tubes))  # rotation variables
        self.betas = np.zeros(len(tubes))  # elongation variables

        self.boundary_values = None

        # current state of robot
        self.positions = None
        self.orientations = None
        self.wrenches = None
        self.uzs = None
        self.thetas = None
        self.seg_indexes = None

    def display_status(self):
        """
        Using rich to make a nice print out
        """
        console = Console()
        table_const = Table(title="Concentric Tube Continuum Robot - Constants")
        table_const.add_column("Constant", justify="right", style="cyan", no_wrap=True)
        table_const.add_column("Value", style="magenta")
        table_const.add_row("Number of Tubes", str(len(self.tubes)))

        table_vars = Table(title="Concentric Tube Continuum Robot - Variables")

        table_vars.add_column("Variable", justify="right", style="cyan", no_wrap=True)
        table_vars.add_column("Value", style="magenta")

        table_vars.add_row("Alpha (Rotation)", str(self.alphas))
        table_vars.add_row("Betas (Elongation)", str(self.betas))
        table_vars.add_row("Alpha (Rot)", str(self.alphas))

        console.print(table_const)
        console.print(table_vars)

    def get_curvature_vector(self, kappa: float):
        """
        Helper function to determine that the pre-curvature is always assumed to be around the y axis
        :param kappa: curvature 1/R
        :return:
        """
        return np.array([0, kappa, 0])  # we consider that the curvature is defined over y

    def get_ordered_segments(self):
        """
        Caclulates the start and end of each segment (a part of the robot that has a constant curvature)
        :return: list of sorted segment ends (starting 0 as base)
        """
        ends = [(rod[1].params['straight_length'] - rod[1].params['L'] * (self.betas[i]),
                 rod[1].params['L'] - rod[1].params['L'] * (self.betas[i])) for i, rod in enumerate(self.tubes)]
        sorted_ends = np.sort([0] + [item for t in ends for item in t])
        sorted_ends = np.unique(sorted_ends)
        # sorted_ends = np.sort([item for t in ends for item in t])
        sorted_ends = sorted_ends[sorted_ends >= 0]
        return list(sorted_ends)

    def rotate(self, alphas: Union[list, npt.ArrayLike]):
        """
        Change alphas to new rotation.
        ATTENTION: You need to call calc_forward again to get new shape
        :param alphas: list or array of alpha values
        :return: None
        """
        for i in range(len(self.tubes)):
            self.tubes[i][1].params["alpha"] = alphas[i]
            self.alphas[i] = alphas[i]

    def translate(self, betas: Union[list, npt.ArrayLike]):
        """
        Change beta values. NOTE: a validity check is performed to reject configurations with inner tubes disappearing
        into outer tubes
        ATTENTION: You need to call calc_forward again to get new shape
        :param betas:
        :return:
        """
        if self._check_beta_validity(betas):
            for i in range(len(self.tubes)):
                self.tubes[i][1].params["beta"] = betas[i]
                self.betas[i] = betas[i]
        else:
            raise Exception('Parameter Error', 'The beta values do not correspond to the specifications!')

    def calc_forward(self, wrench: Union[list, npt.ArrayLike], step_len: float):
        """
        ode integration for each element to reduce jumps at the segment boarders
        :param wrench: inital wrench at init frame
        :param step_len: integration step len
        :return:
        """
        R = self.R_init
        p = self.p_init
        self.step_len = step_len
        w = wrench

        segment_list = self.get_ordered_segments()
        ode_returns = []
        thetas = []
        for i in range(len(self.tubes)):
            thetas.append(self.alphas[self.tubes[i][0]])
        uzs = np.zeros(len(self.tubes))
        us = []
        seg_indexes = []
        current_seg_point = 0
        seg_indexes.append(current_seg_point)

        for i in range(1, len(segment_list)):
            self._curr_calc_tubes = []  # gather tubes that determine this segment. Attribute because it is need in ode
            for t in self.tubes:
                if t[1].is_curved_or_at_end(segment_list[i]) >= 0:
                    self._curr_calc_tubes.append(t)

            """
             __guess state__
             p - positions
             R - orientations
             wrench - wrench
             n*uz - torsion of each tube
             n*thetas - difference in rotation
            """

            state = np.hstack([p.flatten(), R.reshape((1, 9))[0], w, uzs, thetas])  # guess

            steps = int(np.ceil((segment_list[i] - segment_list[i - 1]) / self.step_len))
            s = np.linspace(segment_list[i - 1], segment_list[i], steps)  # TODO make it optional to set this?

            ode_states = integrate.solve_ivp(self.cosserate_rod_ode, (
                segment_list[i - 1], segment_list[i]), state, dense_output=True,
                                             t_eval=s,
                                             method='LSODA')  # beta is defined 0-1 -> L-L*beta  if beta 0 -> fully elongated tube

            p = ode_states.y.T[-1:, :3][0]
            R = np.reshape(ode_states.y.T[-1:, 3:12], (3, 3))
            w = ode_states.y.T[-1, 12:18]
            uzs = ode_states.y.T[-1, 18:18 + len(self.tubes)]
            thetas = ode_states.y.T[-1, 18 + len(self.tubes):]
            us.append(self.calc_us(ode_states.y.T[:, 18 + len(self.tubes):], ode_states.y.T[:, 15:18], step_len))
            ode_returns.append(ode_states.y.T)
            current_seg_point += ode_states.y.T[:, :3].shape[0]
            seg_indexes.append((len(self._curr_calc_tubes), current_seg_point))
        return np.vstack(ode_returns), np.vstack(us), seg_indexes

    def calc_us(self, thetas_batch, m_batch, step_len):
        s = 0
        u = np.zeros((thetas_batch.shape[0], 3))
        for i, thetas in enumerate(thetas_batch):
            u[i, :] = self.calc_uxz(thetas, m_batch[i], s)
            s += step_len
        return u

    def calc_uxz(self, thetas, m, s):
        EIk = 0
        for i in range(len(self._curr_calc_tubes)):
            EIk += self.tubes[i][1].params['Kbt'][0, 0]

        RthetaEkIkuj_star = np.zeros((3, 1))  # PhD Rucker page 91 eq 3.56 last part

        for i in range(len(self._curr_calc_tubes)):
            Rtheta = Rz(thetas[i])
            EIj = self.tubes[i][1].params['Kbt'][0, 0]

            curved_part = self.tubes[i][1].is_curved_or_at_end(s)
            tube_curvature = self.tubes[i][1].params["kappa"] * curved_part
            uj_star = self.get_curvature_vector(tube_curvature)
            RthetaEkIkuj_star += EIj * Rtheta @ uj_star.reshape(-1, 1)

        return 1 / EIk * (m + RthetaEkIkuj_star.T)

    def cosserate_rod_ode(self, s, state):
        """
        Implementation of ODE by doi: 10.1109/TRO.2010.2062570
        :param s: current arc length
        :param state: current state of ode
        :return:
        """
        R = np.reshape(state[3:12], (3, 3))
        n = state[12:15]
        m = state[15:18]
        uzs = []
        thetas = []
        for i in range(len(self._curr_calc_tubes)):
            uzs.append(state[18 + i:18 + (i + 1)].item())

        for i in range(len(self._curr_calc_tubes)):
            thetas.append(state[18 + len(self.tubes) + i:18 + len(self.tubes) + (i + 1)].item())

        u1 = self.calc_uxz(thetas, m, s)

        uixy = []
        for i in range(1, len(self._curr_calc_tubes)):
            Rtheta = Rz(thetas[i])
            uixy.append((Rtheta @ u1.T).T[0, :2])
        uixy = [u1[0, :2]] + uixy

        external_forces = self.tubes[0][1].external_gauss_forces(s)

        # ode
        ps = R @ self.tubes[0][1]._e3.T
        Rs = R @ hat(u1.T)
        thetas_s = []
        for i in range(len(self._curr_calc_tubes)):
            thetas_s.append(uzs[i] - uzs[0])

        uiz_s = []
        for i in range(1, len(self._curr_calc_tubes)):
            curved_part = self.tubes[i][1].is_curved_or_at_end(s)
            tube_curvature = self.tubes[i][1].params["kappa"] * curved_part
            ui_star = self.get_curvature_vector(tube_curvature)
            EIi = self.tubes[i][1].params['Kbt'][0, 0]
            GJi = self.tubes[i][1].params['Kbt'][2, 2]
            uiz_s.append(ui_star[2] + EIi / GJi * (uixy[i][0] * ui_star[1] - uixy[i][1] * ui_star[0]))  # - 1/GJi)
        uiz_s = [u1[0, 2]] + uiz_s

        uzs = np.zeros(len(self.tubes))
        thetas = np.zeros(len(self.tubes))
        for i in range(len(self._curr_calc_tubes)):
            uzs[i] = uiz_s[i]
            thetas[i] = thetas_s[i]

        ns = -R @ external_forces
        msbxy = -hat(u1.T) @ m.T - hat(self.tubes[0][1]._e3.T) @ R.T @ n  # - R.T@l

        return np.hstack([ps.T.flatten(),
                          Rs.reshape((1, 9)).flatten(),
                          ns.flatten(),
                          msbxy.flatten(),
                          np.hstack(uzs),
                          np.hstack(thetas)])

    # def cosserat_rod_ode_curvature(self, s, state):

    def external_gauss_forces(self, s):

        fx = np.sum(
            [self._gaussians[i][0][0] * np.exp(-self._gaussians[i][1] * (s - self._gaussians[i][2]) ** 2) for i in
             range(len(self._gaussians))])
        fy = np.sum(
            [self._gaussians[i][0][1] * np.exp(-self._gaussians[i][1] * (s - self._gaussians[i][2]) ** 2) for i in
             range(len(self._gaussians))])
        return np.array([[fx, fy, 0]]).T

    def set_boundary_condition(self, names, values):
        """
        Accounting of boundary conditions for the differential equations of the CTCR
        :param names: name of the bounda
        :param values:
        :return:
        """
        if self.boundary_values is None:
            self.boundary_values = {}
        for i, name in enumerate(names):
            self.boundary_values[name] = values[i]

    def remove_boundary_condition(self, name):
        """
        Remove a set boundary conditon
        :param name: name of the condition
        """
        del self.boundary_values[name]

    def remove_all_boundary_condition(self):
        """
        Removes all boundary conditions/values
        """
        self.boundary_values = {}

    def _check_beta_validity(self, betas):
        """
        Check for validity of beta
        :param betas: elongation values
        :return:
        """
        valid = True
        for i in range(1, np.asarray(betas).shape[0]):
            valid = valid and self.tubes[i][1].params['L'] - betas[i] * self.tubes[i][1].params['L'] <= \
                    self.tubes[i - 1][1].params['L'] - betas[i - 1] * self.tubes[i - 1][1].params['L']
        return valid

    def fwd_static(self, wrench, step_size: float = 0.01):
        """
        Apply the current configuration and the inital wrench
        :param wrench:
        :param step_size:
        :return: [positions as R^3],[orientations as matrix in SO(3)],[wrench as spear in R^6],[curvature in R^3],[curvatures in R]
        """
        state, us, seg_indexes = self.calc_forward(np.asarray(wrench), step_size)
        self.seg_indexes = seg_indexes
        self.positions = state[:, :3]
        self.orientations = state[:, 3:12]
        self.wrenches = state[:, 12:12 + 6]
        self.uzs = state[:, 18:18 + len(self.tubes)]  # uxy (and also z) of inner tube
        self.thetas = state[:, 18 + len(self.tubes):]  # all uz of all tubes
        return self.positions, self.orientations, self.wrenches, us, self.uzs, self.thetas

    def fwd_kinematic(self, step_size: float = 0.01):
        """
        Short hand function to get only the forward kinematics
        :param step_size:
        :return:
        """
        return self.fwd_static(np.zeros(6), step_size)

    def fwd_static_with_boundarys(self, init_wrench: Union[list, npt.ArrayLike], shooting_function: Callable,
                                  step_size: float = 0.01):
        """
        calculate the forward model with an externally defined shooting method
        :init_wrench: guess od the initial wrench
        :shooting_function: a function getting receiving a state, a robot object and the integration step size
        """
        state = init_wrench
        # solution_bvp = least_squares(shooting_function, state, method='lm', loss='linear', ftol=1e-6,
        #                             args=(self, step_size))
        solution_bvp = sc.optimize.root(shooting_function, state, method="lm", args=(self, step_size))
        return self.fwd_static(solution_bvp.x, step_size)

    def push_end(self, wrench, step_size=0.01):
        """
        Uses fwd_static_with_boundarys to calculate the shape given an external wrench at the tip
        """
        if np.count_nonzero(wrench) and wrench[2] != 0 and False:  # bad .. TODO
            # TODO simplification - only uses A of inner rod
            A = self.tubes[0][1].params['A']
            L = self.tubes[0][1].params['L']
            E = self.tubes[0][1].params['E']
            delta = (-wrench[2] * L) / (E * A)
            self.set_boundary_condition(['pL', 'RL'], [np.array([0, 0, L + delta]), np.identity(3)])
            shooting_foo = shooting_function_tip_position
        else:
            shooting_foo = shooting_function_force
            self.set_boundary_condition(['tip_wrench'], [-np.asarray(wrench)])
        state = np.zeros(6)
        positions, orientations, wrenches, us, uzs, thetas = self.fwd_static_with_boundarys(state, shooting_foo,
                                                                                            step_size)
        if np.count_nonzero(wrench) and wrench[2] != 0 and False:
            self.remove_boundary_condition("pL")
            self.remove_boundary_condition("RL")
        else:
            self.remove_boundary_condition("tip_wrench")
        return positions, orientations, wrenches, us, uzs, thetas

    def push_end_to_position(self, pos, step_size=0.01):
        """
        Uses fwd_static_with_boundarys to calculate the shape given an external wrench at the tip
        """

        self.set_boundary_condition(['pL', 'RL'], [pos, np.identity(3)])
        shooting_foo = shooting_function_tip_pose
        state = np.zeros(6)
        positions, orientations, wrenches, us, uzs, thetas = self.fwd_static_with_boundarys(state, shooting_foo,
                                                                                            step_size)
        self.remove_boundary_condition("pL")
        self.remove_boundary_condition("RL")

        return positions, orientations, wrenches, us, uzs, thetas

    def fwd_external_gaussian_forces(self, step_size=0.001):
        wrench = np.zeros(6)
        positions, orientations, wrenches, us, uzs, thetas = self.fwd_static_with_boundarys(wrench,
                                                                                            shooting_function_gaussian_forces,
                                                                                            step_size)
        return positions, orientations, wrenches, us, uzs, thetas

    def set_gaussians(self, gaussians):
        self._gaussians = gaussians
        self.tubes[0][1]._gaussians = gaussians

    def remove_gaussians(self):
        self._gaussians = []
        self.tubes[0][1]._gaussians = []

    def get_tube_outer_radii(self):
        """
        Helper function to get fast information about tube size
        :return:
        """
        outer_radii = []
        for t in self.tubes:
            outer_radii.append(t[1].params['r_outer'])
        return outer_radii


"""
Shooting Methods
"""


def shooting_function_force(guess, ctr, step_size):
    """

    :param guess:
    :return:
    """
    n0 = guess[:3]
    m0 = guess[3:6]
    tip_wrench = ctr.boundary_values['tip_wrench']
    _, _, tip_wrench_shooting, _, _, _ = ctr.fwd_static(np.hstack([n0, m0]), step_size)

    distal_force_error = tip_wrench[:3] - tip_wrench_shooting[-1, :3]
    distal_moment_error = tip_wrench[3:] - tip_wrench_shooting[-1, 3:]
    # invhat(hat(tip_wrench[3:]).T.dot(hat(tip_wrench_shooting[-1,3:])) - hat(tip_wrench[3:]).dot(
    # hat(tip_wrench_shooting[-1,3:]).T))
    return np.hstack([distal_force_error, distal_moment_error])


def shooting_function_gaussian_forces(guess, ctr, step_size):
    """

    :param guess:
    :return:
    """
    n0 = guess[:3]
    m0 = guess[3:6]
    tip_wrench = np.zeros(6)
    _, _, tip_wrench_shooting, _, _ = ctr.fwd_static(np.hstack([n0, m0]), step_size)
    distal_force_error = tip_wrench[:3] - tip_wrench_shooting[-1, :3]
    distal_moment_error = tip_wrench[3:] - tip_wrench_shooting[-1, 3:]
    return np.hstack([distal_force_error, distal_moment_error])


def shooting_function_tip_pose(guess, ctr, step_size):
    """

    :param guess:
    :return:
    """
    n0 = guess[:3]
    m0 = guess[3:6]
    pL = ctr.boundary_values['pL']
    RL = ctr.boundary_values['RL']

    p, R, _, _, _, _ = ctr.fwd_static(np.hstack([n0, m0]), step_size)

    R = R[-1].reshape((3, 3))
    position_error = p[-1] - pL
    rotation_error = invhat(R.T @ RL - R @ RL.T)
    return np.concatenate([position_error, rotation_error])


def shooting_function_tip_position(guess, ctr, step_size):
    """

    :param guess:
    :return:
    """
    n0 = guess[:3]
    m0 = guess[3:6]
    pL = ctr.boundary_values['pL']

    p, _, _, _, _, _ = ctr.fwd_static(np.hstack([n0, m0]), step_size)
    position_error = p[-1] - pL

    return np.hstack([position_error, np.zeros((3))])
