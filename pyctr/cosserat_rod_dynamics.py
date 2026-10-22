"""
Dynamic Cosserat rod models, BDF-alpha in time.
DynamicCosseratRod: port of 02_Dynamics/00_Cosserat_Rod_PDE from https://github.com/JohnDTill/ContinuumRobotExamples (MIT),
    Euler in space, shooting method at every time step.
DynamicConcentricTubeRobot: Till et al., "A Dynamic Model for Concentric Tube Robots", IEEE T-RO 36(6), 2020,
    midpoint rule in space, Newton's method on all nodes at every time step.
"""

import numpy as np
from scipy.optimize import root
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .cosserat_rod import CurvedCosseratRod
from .robots import ConcentricTubeContinuumRobot
from .utils import hat, Rz


def _bdf_alpha(dt, alpha):
    """Coefficients c0, c1, c2, d1 of x_t = c0 * x + x_h"""
    return (1.5 + alpha) / (dt * (1 + alpha)), -2 / dt, (0.5 + alpha) / (dt * (1 + alpha)), alpha / (1 + alpha)


def _advance(c, x, x_old, x_h):
    """History term x_h for the next time step"""
    c0, c1, c2, d1 = c
    return c1 * x + c2 * x_old + d1 * (c0 * x + x_h)


class DynamicCosseratRod(CurvedCosseratRod):
    """
    Rod with clamped base (initial conditions p0, R0) and a tip wrench [n, m] (world frame) as boundary condition.
    Extra params: Bse, Bbt (material damping, 3x3), C (square law drag, 3x3). All default to zero.
    """

    def __init__(self, params=None, dt=2e-3, bdf_alpha=-0.2):
        super().__init__(params)
        for key in ("Bse", "Bbt", "C"):
            self.params.setdefault(key, np.zeros((3, 3)))
        self.N = int(np.ceil(self.params["L"] / self.params["s"])) + 1
        self.rhoJ = self.params["rho"] * np.diag(
            [self.params["I"], self.params["I"], self.params["J"]]
        )
        self.dt = dt
        self._c = _bdf_alpha(dt, bdf_alpha)

    def _integrate(self, guess, c0, z_h):
        """
        Euler integration from base to tip
        :param guess: [n0, m0] at the base
        :param c0: BDF-alpha coefficient, 0 for statics
        :param z_h: history terms (N, 12) of z = [q, w, v, u]
        :return: y (N, 24) = [p, R, n, m, q, w] and z (N, 12)
        """
        prm = self.params
        ds = prm["L"] / (self.N - 1)
        Kse_inv = np.linalg.inv(prm["Kse"] + c0 * prm["Bse"])
        Kbt_inv = np.linalg.inv(prm["Kbt"] + c0 * prm["Bbt"])
        Kse_vstar = prm["Kse"] @ np.array([0, 0, 1.0])
        rhoA = prm["rho"] * prm["A"]
        rhoAg = rhoA * prm["g"].flatten()

        y = np.zeros((self.N, 24))
        z = np.zeros((self.N, 12))
        y[0, :18] = np.hstack(
            [
                self.inital_conditions["p0"].flatten(),
                self.inital_conditions["R0"].flatten(),
                guess,
            ]
        )  # q = w = 0 at the clamped base
        for j in range(self.N):
            R = y[j, 3:12].reshape(3, 3)
            n, m, q, w = y[j, 12:15], y[j, 15:18], y[j, 18:21], y[j, 21:24]
            u_star = np.array([0, prm["kappa"] if j * ds > prm["straight_length"] else 0, 0])

            v = Kse_inv @ (R.T @ n + Kse_vstar - prm["Bse"] @ z_h[j, 6:9])
            u = Kbt_inv @ (R.T @ m + prm["Kbt"] @ u_star - prm["Bbt"] @ z_h[j, 9:12])
            z[j] = np.hstack([q, w, v, u])
            if j == self.N - 1:
                break
            q_t, w_t, v_t, u_t = np.split(c0 * z[j] + z_h[j], 4)

            U, W = hat(u), hat(w)  # hat(a) @ b instead of np.cross: np.cross is ~5x slower on 3-vectors
            ps = R @ v
            Rs = R @ U
            ns = rhoA * R @ (q_t + W @ q) + R @ prm["C"] @ (q * np.abs(q)) - rhoAg
            ms = R @ (self.rhoJ @ w_t + W @ self.rhoJ @ w) - hat(ps) @ n
            qs = v_t - U @ q + W @ v
            ws = u_t - U @ w
            y[j + 1] = y[j] + ds * np.hstack([ps, Rs.flatten(), ns, ms, qs, ws])
        return y, z

    def _solve(self, c0, z_h, tip_wrench):
        tip_wrench = np.asarray(tip_wrench, dtype=float)
        solution = root(
            lambda guess: self._integrate(guess, c0, z_h)[0][-1, 12:18] - tip_wrench,
            self._guess,
            method="lm",
        )
        if not solution.success:
            raise RuntimeError(f"shooting method did not converge: {solution.message}")
        self._guess = solution.x  # warm start for the next time step
        self.y, self.z = self._integrate(solution.x, c0, z_h)
        return self.y[:, :3], self.y[:, 3:12], self.y[:, 12:18]

    def init_static(self, tip_wrench=np.zeros(6)):
        """
        Solve the static BVP and take it as rest state at t = 0
        :return: positions (N, 3), orientations (N, 9), wrenches (N, 6)
        """
        self._guess = np.zeros(6)
        out = self._solve(0.0, np.zeros((self.N, 12)), tip_wrench)
        _, c1, c2, _ = self._c
        self._z_old, self._z_h = self.z, (c1 + c2) * self.z
        return out

    def step(self, tip_wrench=np.zeros(6)):
        """
        Advance one time step dt. Call init_static first.
        :return: positions (N, 3), orientations (N, 9), wrenches (N, 6). Velocities q, w are in self.y[:, 18:24]
        """
        out = self._solve(self._c[0], self._z_h, tip_wrench)
        self._z_h, self._z_old = _advance(self._c, self.z, self._z_old, self._z_h), self.z
        return out


def _hat(u):
    """hat for an array of vectors (I, 3) -> (I, 3, 3)"""
    H = np.zeros((len(u), 3, 3))
    H[:, 0, 1], H[:, 0, 2], H[:, 1, 2] = -u[:, 2], u[:, 1], -u[:, 0]
    return H - H.transpose(0, 2, 1)


class DynamicConcentricTubeRobot(ConcentricTubeContinuumRobot):
    """
    Concentric tube robot dynamics, Till et al. 2020, eq. (13), (16), (17), (19), (21), (25) with Kirchhoff tubes.
    Tubes are ordered inner to outer and curved about their y axis (u* = [0, kappa, 0]) after straight_length.
    alphas may change between steps (rotate), betas are fixed after init_static (the paper assumes beta_t = 0).
    Not modelled: transmission below the base plate, friction, tip masses.
    Tip wrench [F, M] (world frame) acts on the end of the inner tube. Optional tube param: Bbt (damping, 3x3).

    y per node: p, R (of tube 1), q, w, n (world), m^b_xy (total, tube 1 frame), m^b_z, theta, gamma (per tube)
    z per interval between nodes (variables with time derivatives): q, w, u_xy (tube 1), u_z, gamma (per tube)
    """

    def __init__(self, tubes, dt=2e-3, bdf_alpha=-0.2, **kwargs):
        super().__init__(tubes, **kwargs)
        prm = [t.params for _, t in self.tubes]
        damping = [p.get("Bbt", np.zeros((3, 3))) for p in prm]
        self._EI = np.array([p["Kbt"][0, 0] for p in prm])
        self._GJ = np.array([p["Kbt"][2, 2] for p in prm])
        self._BB = np.array([b[0, 0] for b in damping])
        self._BT = np.array([b[2, 2] for b in damping])
        self._rhoI = np.array([p["rho"] * p["I"] for p in prm])
        self._kappa = np.array([p["kappa"] for p in prm], dtype=float)
        self._rhoA = sum(p["rho"] * p["A"] for p in prm)
        self._f = sum(p["rho"] * p["A"] * p["g"].flatten() for p in prm)  # gravity
        self.dt = dt
        self._c = _bdf_alpha(dt, bdf_alpha)

    def _grid(self):
        """Nodes self.s from the current betas, tubes present and u*_y per interval, residual rows -> nodes"""
        k = len(self.tubes)
        L = np.array([t.params["L"] for _, t in self.tubes])
        end = L * (1 - self.betas)
        curve = np.array([t.params["straight_length"] for _, t in self.tubes]) - L * self.betas
        borders = np.unique(np.concatenate([[0], end, curve]))
        borders = borders[(borders >= 0) & (borders <= end[0])]
        ds = self.tubes[0][1].params["s"]
        s, active, kappa = [np.zeros(1)], [], []
        for s0, s1 in zip(borders[:-1], borders[1:]):
            n = int(np.ceil((s1 - s0) / ds))
            s.append(np.linspace(s0, s1, n + 1)[1:])
            active += [end >= s1] * n
            kappa += [self._kappa * (end >= s1) * (curve < 0.5 * (s0 + s1))] * n
        self.s = np.concatenate(s)
        self._active, self._kappa_s = np.array(active, dtype=float), np.array(kappa)
        self._ends = np.searchsorted(self.s, end[1:])
        self._base_idx = np.r_[0:18, 23 + k : 23 + 3 * k]

        # every residual row depends on one or two neighbouring nodes, see _residual
        M, ny = len(self.s), 23 + 3 * k
        i = np.repeat(np.arange(M - 1), ny)
        a = np.concatenate([i, np.zeros(len(self._base_idx), int), self._ends, np.full(6, M - 1)])
        b = np.concatenate([i + 1, a[len(i) :]])
        self._row_node = [np.where(a % 2 == p, a, np.where(b % 2 == p, b, -1)) for p in (0, 1)]

    def _ode(self, y, z_h, c0):
        """eq. (13) with x_t = c0 * x + x_h, for all intervals at once: y (M - 1, ny), z_h (M - 1, nz)"""
        k = len(self.tubes)
        active, kappa = self._active, self._kappa_s
        R = y[:, 3:12].reshape(-1, 3, 3)
        q, w, n, mxy = y[:, 12:15], y[:, 15:18], y[:, 18:21], y[:, 21:23]
        mz, th, ga = y[:, 23 : 23 + k] * active, y[:, 23 + k : 23 + 2 * k], y[:, 23 + 2 * k :] * active
        EI, BB, rhoI = self._EI * active, self._BB * active, self._rhoI * active
        q_h, w_h, uxy_h, uz_h, ga_h = z_h[:, 0:3], z_h[:, 3:6], z_h[:, 6:8], z_h[:, 8 : 8 + k], z_h[:, 8 + k :]
        cos, sin = np.cos(th), np.sin(th)

        uz = (mz - self._BT * uz_h) / (self._GJ + c0 * self._BT) * active  # (16), u*_z = 0
        uxy = (
            mxy
            - BB.sum(1, keepdims=True) * uxy_h
            + np.column_stack([-(EI * kappa * sin).sum(1), (EI * kappa * cos).sum(1)])
        ) / (EI.sum(1) + c0 * BB.sum(1))[:, None]  # (17)
        q_t, w_t, uxy_t = c0 * q + q_h, c0 * w + w_h, c0 * uxy + uxy_h
        uz_t, ga_t = c0 * uz + uz_h, (c0 * ga + ga_h) * active
        u, u_t = np.column_stack([uxy, uz[:, 0]]), np.column_stack([uxy_t, uz_t[:, 0]])

        # curvature and bending moment of each tube in its own frame, (4) and (14)
        ux, uy = cos * uxy[:, :1] + sin * uxy[:, 1:], cos * uxy[:, 1:] - sin * uxy[:, :1]
        mx = EI * ux + BB * (cos * uxy_t[:, :1] + sin * uxy_t[:, 1:])
        my = EI * (uy - kappa) + BB * (cos * uxy_t[:, 1:] - sin * uxy_t[:, :1])

        nb = np.einsum("ikj,ik->ij", R, n)  # R^T n
        w_x_e3 = np.column_stack([w[:, 1], -w[:, 0], np.zeros(len(y))])
        e3_x_nb = np.column_stack([-nb[:, 1], nb[:, 0], np.zeros(len(y))])
        mbs = -np.cross(u, np.column_stack([mxy, mz.sum(1)])) - e3_x_nb + rhoI.sum(1, keepdims=True) * w_t

        ys = np.empty_like(y)
        ys[:, 0:3] = R[:, :, 2]
        ys[:, 3:12] = (R @ _hat(u)).reshape(-1, 9)
        ys[:, 12:15] = -np.cross(u, q) + w_x_e3
        ys[:, 15:18] = u_t - np.cross(u, w)
        ys[:, 18:21] = -self._f + self._rhoA * np.einsum("ijk,ik->ij", R, np.cross(w, q) + q_t)
        ys[:, 21:23] = mbs[:, :2] + w_x_e3[:, :2] * (rhoI.sum(1) * w[:, 2] + (rhoI * ga).sum(1))[:, None]
        ys[:, 23 : 23 + k] = -(ux * my - uy * mx) + 2 * rhoI * (w_t[:, 2:3] + ga_t)
        ys[:, 23 + k : 23 + 2 * k] = (uz - uz[:, :1]) * active
        ys[:, 23 + 2 * k :] = (uz_t - uz_t[:, :1]) * active
        return ys, np.column_stack([q, w, uxy, uz, ga])

    def _residual(self, x, c0, z_h, alphas_t, tip_wrench):
        """Midpoint rule between nodes (25), base (19), free torsion at the end of outer tubes, tip wrench (21)"""
        k = len(self.tubes)
        y = x.reshape(len(self.s), -1)
        ys, z = self._ode(0.5 * (y[1:] + y[:-1]), z_h, c0)
        base = np.concatenate(
            [
                self.p_init.flatten(),
                (self.R_init @ Rz(self.alphas[0])).flatten(),
                np.zeros(5),
                alphas_t[:1],
                self.alphas - self.alphas[0],
                alphas_t - alphas_t[0],
            ]
        )
        Mb = y[-1, 3:12].reshape(3, 3).T @ tip_wrench[3:]
        r = np.concatenate(
            [
                ((y[1:] - y[:-1]) / np.diff(self.s)[:, None] - ys).flatten(),
                y[0, self._base_idx] - base,
                y[self._ends, 23 + np.arange(1, k)],
                y[-1, 18:21] - tip_wrench[:3],
                y[-1, 21:23] - Mb[:2],
                [y[-1, 23] - Mb[2]],
            ]
        )
        return r, z

    def _jacobian(self, x, r, *args):
        """Finite differences, columns grouped: nodes of equal parity never share a residual row"""
        M = len(self.s)
        ny = len(x) // M
        rows, cols, vals = [], [], []
        for p, dep in enumerate(self._row_node):
            use = np.flatnonzero(dep >= 0)
            for c in range(ny):
                idx = np.arange(p, M, 2) * ny + c
                h = np.zeros(M)
                h[p::2] = 1e-8 * (1 + np.abs(x[idx]))
                x2 = x.copy()
                x2[idx] += h[p::2]
                dr = self._residual(x2, *args)[0] - r
                rows.append(use)
                cols.append(dep[use] * ny + c)
                vals.append(dr[use] / h[dep[use]])
        rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        return coo_matrix((vals, (rows, cols)), shape=(len(x), len(x))).tocsc()

    def _solve(self, c0, z_h, alphas_t, tip_wrench):
        """Newton's method on the discretized BVP, starting from the last solution"""
        args = (c0, z_h, alphas_t, np.asarray(tip_wrench, dtype=float))
        x = self.y.flatten()
        for _ in range(20):
            r, self.z = self._residual(x, *args)
            if np.abs(r).max() < 1e-9:
                break
            x = x - spsolve(self._jacobian(x, r, *args), r)
        else:
            raise RuntimeError(f"Newton's method did not converge, max residual {np.abs(r).max():.1e}")
        self.y = x.reshape(len(self.s), -1)
        k = len(self.tubes)
        wrenches = np.hstack([self.y[:, 18:23], self.y[:, 23 : 23 + k].sum(axis=1, keepdims=True)])
        return self.y[:, :3], self.y[:, 3:12], wrenches

    def init_static(self, tip_wrench=np.zeros(6)):
        """
        Solve the static BVP for the current alphas and betas and take it as rest state at t = 0
        :return: positions (M, 3), orientations (M, 9), wrenches [n, m^b] (M, 6) at the arc lengths self.s
        """
        self._grid()
        self._betas = self.betas.copy()
        k = len(self.tubes)
        R0 = self.R_init @ Rz(self.alphas[0])
        self.y = np.zeros((len(self.s), 23 + 3 * k))  # straight robot as first guess
        self.y[:, :3] = self.p_init.flatten() + np.outer(self.s, R0[:, 2])
        self.y[:, 3:12] = R0.flatten()
        self.y[:, 23 + k : 23 + 2 * k] = self.alphas - self.alphas[0]
        out = self._solve(0.0, np.zeros((len(self.s) - 1, 8 + 2 * k)), np.zeros(k), tip_wrench)
        _, c1, c2, _ = self._c
        self._z_old, self._z_h = self.z, (c1 + c2) * self.z
        self._a_old, self._a_h = self.alphas.copy(), (c1 + c2) * self.alphas
        return out

    def step(self, tip_wrench=np.zeros(6)):
        """
        Advance one time step dt with the current alphas (see rotate). Call init_static first.
        :return: positions (M, 3), orientations (M, 9), wrenches [n, m^b] (M, 6) at the arc lengths self.s
        """
        if not np.array_equal(self.betas, self._betas):
            raise ValueError("betas changed: insertion during dynamics is not modelled, call init_static again")
        c0 = self._c[0]
        out = self._solve(c0, self._z_h, c0 * self.alphas + self._a_h, tip_wrench)
        self._z_h, self._z_old = _advance(self._c, self.z, self._z_old, self._z_h), self.z
        self._a_h, self._a_old = _advance(self._c, self.alphas, self._a_old, self._a_h), self.alphas.copy()
        return out
