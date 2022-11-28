import numpy as np

def hat(mat):
    return np.array([[0, -mat[2], mat[1]],
                     [mat[2], 0 , -mat[0]],
                     [-mat[1], mat[0], 0]], dtype=np.float64)

def invhat(mat):
    return np.array([mat[2,1], mat[0,2], mat[1,0]])



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
        ax.plot(x[0, :], x[1, :], x[2, :], color='b')
        ax.plot(y[0, :], y[1, :], y[2, :], color='g')
        ax.plot(z[0, :], z[1, :], z[2, :], color='r')
    return ax


def single_curved_rod():
    rod = CurvedCosseratRod()
    # Arbitrary base frame assignment
    p0 = np.array([[0, 0, 0]])
    R0 = np.eye(3)

    rod.set_initial_conditions(p0, R0)
    rod.params['L'] = 1.

    kappa = np.deg2rad(90) / rod.params['L']
    rod.inital_conditions['kappa_0'] = np.array([0, 0, 0])

    rod.params['kappa'] = kappa
    rod.params['straight_length'] = 0.25

    states = rod.push_end(np.array([0, 0, 0, 0, 0, 0]))
    Rn0 = np.reshape(states[-1][3:12], (3, 3))
    pn0 = np.asarray([states[-1][:3]])

    ax = plt.figure().add_subplot(projection='3d')
    x_vals, y_vals, z_vals = states[:, 0], states[:, 1], states[:, 2]
    ax.plot(x_vals, y_vals, z_vals, label='parametric curve')
    plot_frame(ax, Rn0, pn0)

    for p in np.linspace(0.1, 1, 5):
        f = 1 * p
        r = 0  # np.pi/2 * p
        R0 = np.array([[np.cos(r), -np.sin(r), 0],
                       [np.sin(r), np.cos(r), 0],
                       [0, 0, 1]])
        rod.set_initial_conditions(p0, R0)
        rod.inital_conditions['kappa_0'] = np.array([0, 0, 0])
        states = rod.push_end(np.array([0, 0, 0, 0, 0, 0]))
        Rn0 = np.reshape(states[-1][3:12], (3, 3))
        wrench = np.array([0, -f, -f, 0, 0, 0])
        wrench[:3] = Rn0 @ wrench[:3]
        wrench[3:] = Rn0 @ wrench[3:]
        states = rod.push_end(wrench)
        p = np.asarray([states[-1][:3]])
        R = np.reshape(states[-1][3:12], (3, 3))
        plot_frame(ax, R, p)

        x_vals, y_vals, z_vals = states[:, 0], states[:, 1], states[:, 2]
        ax.plot(x_vals, y_vals, z_vals, label='parametric curve')
    plot_all_frame(ax, states)
    plot_frame(ax, R0, p0)
    ax.set_zlabel('Z')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set(xlim=[-0.2, 0.2], ylim=[-0.2, 0.2], zlim=[-0.2, 0.2])
    plt.show()


def two_curved_tubes():
    # Arbitrary base frame assignment
    p0 = np.array([[0, 0, 0]])
    R0 = np.eye(3)

    # define rods
    inner_rod = CurvedCosseratRod()

    inner_rod.set_initial_conditions(p0, R0)
    inner_rod.params['L'] = 178.80 * 1e-3  # 337e-3

    kappa = 10.47  # 3.4 #np.deg2rad(90) / rod.params['L']
    print(kappa)
    inner_rod.inital_conditions['kappa_0'] = np.array([0, 0, 0])

    beta = 1.0

    inner_rod.params['kappa'] = kappa
    inner_rod.params['alpha'] = np.deg2rad(0)
    inner_rod.params['beta'] = beta * 178.80 * 1e-3  # 337e-3
    inner_rod.params['straight_length'] = (178.80 - 150) * 1e-3  # 275e-3

    #####################
    outer_rod = CurvedCosseratRod()
    # Arbitrary base frame assignment
    outer_rod.set_initial_conditions(p0, R0)
    outer_rod.params['L'] = 84.3 * 1e-3  # 501e-3

    kappa = 6.98  # 7.3 #np.deg2rad(90) / rod2.params['L']
    outer_rod.inital_conditions['kappa_0'] = np.array([0, 0, 0])
    print(kappa)
    outer_rod.params['kappa'] = kappa
    outer_rod.params['alpha'] = 0
    outer_rod.params['beta'] = 0.
    outer_rod.params['straight_length'] = (84.3 - 75) * 1e-3  # 435e-3

    ctr = CombinedTubes((inner_rod, outer_rod))
    print(ctr.get_ordered_segments())

    states = ctr.calc_forward(R0, p0, np.array([0, 0, -0.0, 0, 0, 0]), 0.01)

    ax = plt.figure().add_subplot(projection='3d')
    x_vals, y_vals, z_vals = states.y.T[:, 0], states.y.T[:, 1], states.y.T[:, 2]
    ax.plot(x_vals, y_vals, z_vals, label='parametric curve')

    inner_rod.params['beta'] = 0.1
    outer_rod.params['alpha'] = np.deg2rad(180)
    states = ctr.calc_forward(R0, p0, np.array([0, 0, 0, 0, 0, 0]), 0.01)
    x_vals, y_vals, z_vals = states.y.T[:, 0], states.y.T[:, 1], states.y.T[:, 2]
    ax.plot(x_vals, y_vals, z_vals, label='parametric curve')

    ax.set_zlabel('Z')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set(xlim=[-0.2, 0.2], ylim=[-0.2, 0.2], zlim=[-0.2, 0.2])
    plt.show()

    print(ctr.tubes_end_indexes)