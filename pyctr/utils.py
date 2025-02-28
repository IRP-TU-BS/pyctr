import numpy as np
import matplotlib.pyplot as plt

def hat(mat):
    return np.asarray(
        [
            [0, -mat[2].item(), mat[1].item()],
            [mat[2].item(), 0, -mat[0].item()],
            [-mat[1].item(), mat[0].item(), 0],
        ],
        dtype=np.float64,
    )


def invhat(mat):
    return np.asarray([mat[2, 1], mat[0, 2], mat[1, 0]])


def Rz(theta):
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def plot_frame(ax, R, p):
    ex = R @ np.array([[0.1, 0, 0]]).T
    ey = R @ np.array([[0, 0.1, 0]]).T
    ez = R @ np.array([[0, 0, 0.1]]).T
    x = np.hstack([p.T, p.T + ex])
    y = np.hstack([p.T, p.T + ey])
    z = np.hstack([p.T, p.T + ez])
    ax.plot(x[0, :], x[1, :], x[2, :], color="b")
    ax.plot(y[0, :], y[1, :], y[2, :], color="g")
    ax.plot(z[0, :], z[1, :], z[2, :], color="r")
    return ax


def plot_all_frame(ax, states):
    for state in states:
        R = np.reshape(state[3:12], (3, 3))
        p = np.asarray([state[:3]])
        ex = R @ np.array([[0.01, 0, 0]]).T
        ey = R @ np.array([[0, 0.01, 0]]).T
        ez = R @ np.array([[0, 0, 0.01]]).T
        x = np.hstack([p.T, p.T + ex])
        y = np.hstack([p.T, p.T + ey])
        z = np.hstack([p.T, p.T + ez])
        ax.plot(x[0, :], x[1, :], x[2, :], color="b")
        ax.plot(y[0, :], y[1, :], y[2, :], color="g")
        ax.plot(z[0, :], z[1, :], z[2, :], color="r")
    return ax
