# matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation

from vispy import app, scene
from copy import copy
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

class VisPy(Engine):
    def __init__(self):
        self.canvas = scene.SceneCanvas(title='Simple NetworkX Graph', size=(600, 600),
                                   bgcolor='white', show=True)
        self.view = self.canvas.central_widget.add_view('panzoom')

    def display_fwd(self, pos, segments, radii, number_of_tubes, name="robot"):
        self.robot_seg_dict = {}
        for i in range(number_of_tubes):
            tube = scene.Tube(pos=pos, radius=radii[i]*1e3, num_sides=12)

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
                seg, = self.ax.plot([0], [0], [0],
                                     linewidth=radii[i] * 1e3, label=name, c=self.colors[i])
                robot_seg_dict[name + 'tube' + str(i+1)] = seg
                segshad, = self.ax.plot([0], [0], zs=0, zdir='z', c='gray',
                                         alpha=0.7, linewidth=radii[i] * 1e3)
                robot_seg_dict[name + 'tube' + str(i+1) + '_shade'] = segshad
            self.plotted_robots[name] = robot_seg_dict

    def displ_fwd(self,frame, name="robot"):
        pos = frame[0]
        segments = frame[1]
        radii = frame[2]
        number_of_tubes = frame[3]
        cur_tube_num = 1
        tmp_segs = copy(segments)
        tmp_segs.reverse()
        tmp_segs.pop()
        while cur_tube_num <= number_of_tubes and len(tmp_segs) != 0:
            seg = tmp_segs.pop(0)
            if seg[0] == cur_tube_num:
                self.plotted_robots[name][name + 'tube' + str(cur_tube_num)].set_data(
                    pos[:seg[1], 0],
                    pos[:seg[1], 1],
                    )
                self.plotted_robots[name][name + 'tube' + str(cur_tube_num)].set_3d_properties(
                    pos[:seg[1], 2])
                cur_tube_num += 1
        self.set_axes_equal(self.ax)

    def show(self, frames):

        for name in frames.keys():
            ani = FuncAnimation(
                self.fig, partial(self.displ_fwd, name=name),
                frames=frames[name],
                blit=False)
        plt.show()


class Scene:
    """
    The base class for all visualizations in this package
    """

    def __init__(self, engine):
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
            segments =  self.robots[rob].get_segments(),
            radii =  self.robots[rob].get_radi_for_segments(),
            number_of_tubes =  self.robots[rob].num_of_tubes
            self.robot_frames[rob].append([positions, segments[0], radii, number_of_tubes])

    def show(self):
        self.engine.show(self.robot_frames)


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



