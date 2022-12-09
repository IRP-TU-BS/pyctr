import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from functools import partial
import numpy as np
import warnings


class Engine:
    def __init__(self) -> object:
        self.plotted_robots = {}

    def plot_robot_pos(self, pos, segments, radii, number_of_tubes, name="robot"):
        self.displ_fwd(pos, segments, radii, number_of_tubes, name)

    def displ_fwd(self, pos, segments, radii, number_of_tubes, name="robot"):
        pass


class MatplotLib(Engine):
    def __init__(self):
        super().__init__()
        self.colors = list(mcolors.XKCD_COLORS.keys())
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')

    def set_axes_equal(self, ax):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    def add_robot(self, radii, number_of_tubes, name="robot"):
        if not (name in self.plotted_robots):
            robot_seg_dict = {}
            for i in range(number_of_tubes):
                seg1, = self.ax.plot([0], [0], [0],
                                     linewidth=radii[i] * 1e3, label=name, c=self.colors[i])
                robot_seg_dict[name + 'tube' + str(i+1) + 'seg1'] = seg1
                seg2, = self.ax.plot([0], [0], [0],
                                     linewidth=radii[i] * 1e3, label=name, c=self.colors[i])
                robot_seg_dict[name + 'tube' + str(i+1) + 'seg2'] = seg2
                segshad1, = self.ax.plot([0], [0], zs=0, zdir='z', c='gray',
                                         alpha=0.7, linewidth=radii[i] * 1e3)
                robot_seg_dict[name + 'tube' + str(i+1) + 'segshad1'] = segshad1
                segshad2, = self.ax.plot([0], [0], zs=0, zdir='z', c='gray',
                                         alpha=0.7, linewidth=radii[i] * 1e3)
                robot_seg_dict[name + 'tube' + str(i+1) + 'segshad2'] = segshad2
            self.plotted_robots[name] = robot_seg_dict

    def displ_fwd(self, pos, segments, radii, number_of_tubes, name="robot"):
        last_ind = 0
        seg_data = []
        for i in range(number_of_tubes,0,-1):
            seg_dat = []
            for seg in segments[1:]:
                if seg[0] == i:
                    seg_dat.append(pos[last_ind:seg[1],:])
                    last_ind = seg[1]
            if len(seg_dat)<2:
                seg_dat.append(np.zeros((1,3)))
            seg_data.append(seg_dat)

        #print(seg_data)
        #print(segments)
        for i in range(number_of_tubes):
            for j in range(2):
                self.plotted_robots[name][name + 'tube' + str(i+1) + 'seg' + str(j+1)].set_data(seg_data[i][j][:,0],
                                                                                            seg_data[i][j][:, 1],
                                                                                            )
                self.plotted_robots[name][name + 'tube' + str(i+1) + 'seg' + str(j+1)].set_3d_properties(seg_data[i][j][:, 2]
                                                                                            )



        self.set_axes_equal(self.ax)

    def show(self, frames, robots):

        for name in frames.keys():
            ani = FuncAnimation(
                self.fig, partial(self.displ_fwd, segments=robots[name].get_segments(),
                                  radii=robots[name].get_radi_for_segments(),
                                  number_of_tubes=robots[name].num_of_tubes, name=name),
                frames=frames[name],
                blit=False)
        plt.show()


class Scene:
    """
    The base class for all visualizations in this package
    """

    def __init__(self, engine, path):
        self.scene_objects = {}
        self.robots = {}
        self.robot_frames = {}

        self.engine = engine

    def add_robot(self, name, robot):
        if name in self.robots.keys():
            warnings.warn("robot name already used. Replace robot.")
        self.robots[name] = robot
        self.robot_frames[name] = []
        self.engine.add_robot(self.robots[name].get_radi_for_segments(),
                              self.robots[name].num_of_tubes,
                              name)

    def update(self):
        for rob in self.robots.keys():
            positions, _, _, _, _ = self.robots[rob].calc_fwd()
            self.robot_frames[rob].append(positions)

    def show(self):
        self.engine.show(self.robot_frames, self.robots)


class Robot:
    def __init__(self, model_class_factory_function, config_path):
        self.model = model_class_factory_function(config_path)
        self.shape = {}
        self.configs = []
        self.num_of_tubes = len(self.model.tubes)

    def set_config(self, alphas, betas):
        self.configs.append([alphas, betas])
        self.model.rotate(alphas)
        self.model.translate(betas)

    def calc_fwd(self):
        return self.model.fwd_kinematic()

    def get_segments(self):
        return self.model.seg_indexes

    def get_radi_for_segments(self):
        return self.model.get_tube_outer_radii()



