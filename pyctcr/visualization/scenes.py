import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import warnings
class Engine:
    def __init__(self) -> object:
        self.plotted_robots = {}

    def plot_robot_pos(self, pos, segments, radii, number_of_tubes, name ="robot"):
        self.displ_fwd(pos, segments, radii, number_of_tubes, name)

    def displ_fwd(self, pos, segments, radii, number_of_tubes, name ="robot"):
        pass

class MatplotLib(Engine):
    def __init__(self):
        super().__init__()
        self.colors = list(mcolors.XKCD_COLORS.keys())
        self.ax = plt.figure().add_subplot(projection='3d')


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

    def add_robot(self,radii, number_of_tubes, name ="robot"):
        if not (name in self.plotted_robots):
            robot_seg_dict = {}
            for i in number_of_tubes:
                seg1, = self.ax.plot([0], [0], [0],
                             linewidth=radii[i] * 1e3, label=name, c=self.colors[i])
                robot_seg_dict[name+'tube'+str(i)+'seg1'] = seg1
                seg2, = self.ax.plot([0], [0], [0],
                             linewidth=radii[i] * 1e3, label=name, c=self.colors[i])
                robot_seg_dict[name + 'tube' + str(i) + 'seg2'] = seg2
                segshad1, = self.ax.plot([0], [0], zs=0, zdir='z', c='gray',
                             alpha=0.7, linewidth=radii[i] * 1e3)
                robot_seg_dict[name + 'tube' + str(i) + 'segshad1'] = segshad1
                segshad2, = self.ax.plot([0], [0], zs=0, zdir='z', c='gray',
                             alpha=0.7, linewidth=radii[i] * 1e3)
                robot_seg_dict[name + 'tube' + str(i) + 'segshad2'] = segshad2
            self.plotted_robots[name] = robot_seg_dict

    def displ_fwd(self, pos, segments, radii, number_of_tubes, name ="robot"):
        last_ind = 0

        for i, seg in enumerate(segments[1:]):
            seg_index = seg[1]
            radindex = seg[0] - 1
            radi = radii[radindex]
            #if name in self.plotted_robots:

            #else:
            self.ax.plot(pos[last_ind:seg_index,0], pos[last_ind:seg_index,1], pos[last_ind:seg_index,2], linewidth=radi*1e3,  label=name, c=self.colors[radindex])
            self.ax.plot(pos[last_ind:seg_index,0], pos[last_ind:seg_index,1], zs=0, zdir='z', c='gray', alpha=0.7, linewidth=radi*1e3)
            last_ind = seg_index

        self.set_axes_equal(self.ax)

    def show(self):
        plt.show()



class Scene:
    """
    The base class for all visualizations in this package
    """

    def __init__(self, engine, path):
        self.scene_objects = {}
        self.robots = {}
        self.robot_fames = {}

        self.engine = engine


    def add_robot(self,name,  robot):
        if name in self.robots.keys():
            warnings.warn("robot name already used. Replace robot.")

        self.robots[name] = robot

    def update(self):
        for rob in self.robots.keys():
            positions, _, _, _, _ = self.robots[rob].calc_fwd()
            self.engine.plot_robot_pos(positions, self.robots[rob].get_segments(), self.robots[rob].get_radi_for_segments(),
                                       self.robots[rob].num_of_tubes)

    def show(self):

        self.engine.show()

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



