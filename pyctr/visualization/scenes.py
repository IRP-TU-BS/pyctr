# matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

from vispy import scene
from copy import copy
from functools import partial
import numpy as np
import warnings


class Engine:
    def __init__(self) -> object:
        self.plotted_robots = {}

    def plot_robot_pos(self, pos, segments, radii, number_of_tubes, name="robot"):
        self.displ_fwd(pos, segments, radii, number_of_tubes, name)

    def displ_fwd(self, frame, name="robot"):
        pass


class VisPy(Engine):
    def __init__(self):
        self.canvas = scene.SceneCanvas(
            title="PyCTCR - Vispy", size=(600, 600), bgcolor="white", show=True
        )
        self.view = self.canvas.central_widget.add_view("panzoom")
        self.view.camera = "arcball"

        self.tubes = {}

    def displ_fwd(self, frame, name="robot"):
        pos = frame[0]
        segments = frame[1]
        number_of_tubes = frame[2]

        tmp_segs = copy(segments)
        tmp_segs.reverse()
        tmp_segs.pop()

        cur_tube_num = 1
        while cur_tube_num <= number_of_tubes and len(tmp_segs) != 0:
            seg = tmp_segs.pop(0)
            if seg[0] == cur_tube_num:
                if len(self.tubes[name]) > 0:
                    self.tubes[name][cur_tube_num - 1].parent = None
                tube = scene.visuals.Tube(
                    points=pos[: seg[1], :],
                    radius=self.radii[cur_tube_num - 1] * 1e3,
                    parent=self.canvas.scene,
                )
                if len(self.tubes[name]) > 0:
                    self.tubes[name][cur_tube_num - 1] = tube
                else:
                    self.tubes[name].append(tube)
            self.canvas.scene

    def add_robot(self, radii, number_of_tubes, name="robot"):
        self.radii = radii
        self.tubes[name] = []
        # for i in range(number_of_tubes):
        #    self.tubes[name].append(scene.visuals.Tube(points=np.zeros(3), radius=radii[i]*1e3))

    def show(self, frames):
        for name in frames.keys():
            self.displ_fwd(frames[name][0], name)
        self.view.camera.set_range(x=[-3, 3])
        self.canvas.app.run()


class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


# def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
#'''Add an 3d arrow to an `Axes3D` instance.'''

# arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
# ax.add_artist(arrow)


# setattr(Axes3D, 'arrow3D', _arrow3D)


class MatplotLib(Engine):
    def __init__(self):
        super().__init__()
        self.colors = list(mcolors.XKCD_COLORS.keys())
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection="3d")
        self.ani = {}
        self.arrows = {}  # TODO redo.. currently only avalable for all forces and buggy

    def set_axes_equal(self, ax):
        """Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        """

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
        if name not in self.plotted_robots:
            robot_seg_dict = {}
            for i in range(number_of_tubes):
                (seg,) = self.ax.plot(
                    [0],
                    [0],
                    [0],
                    linewidth=radii[i] * 1e3,
                    label=name,
                    c=self.colors[i],
                )
                robot_seg_dict[name + "tube" + str(i + 1)] = seg
                (segshad,) = self.ax.plot(
                    [0],
                    [0],
                    zs=0,
                    zdir="z",
                    c="gray",
                    alpha=0.7,
                    linewidth=radii[i] * 1e3,
                )
                robot_seg_dict[name + "tube" + str(i + 1) + "_shade"] = segshad
            self.plotted_robots[name] = robot_seg_dict

    def remove_all_arrows(self):
        for name in self.arrows.keys():
            self.arrows[name].remove()

    def add_arrow(self, x, y, z, dx, dy, dz, name="tmp", color="red"):
        if name in self.arrows.keys():
            self.arrows[name].remove()
        arrow = Arrow3D(x, y, z, dx, dy, dz, mutation_scale=10, ec=color, fc=color)
        self.ax.add_artist(arrow)
        self.arrows[name] = arrow

    def displ_fwd(self, frame, name="robot", show_forces=False):
        pos = frame[0]
        segments = frame[1]
        number_of_tubes = frame[2]
        rots = frame[3]
        wrenches = frame[4]
        cur_tube_num = 1
        tmp_segs = copy(segments)
        tmp_segs.reverse()
        tmp_segs.pop()
        while cur_tube_num <= number_of_tubes and len(tmp_segs) != 0:
            seg = tmp_segs.pop(0)
            if seg[0] == cur_tube_num:
                self.plotted_robots[name][name + "tube" + str(cur_tube_num)].set_data(
                    pos[: seg[1], 0],
                    pos[: seg[1], 1],
                )
                self.plotted_robots[name][
                    name + "tube" + str(cur_tube_num)
                ].set_3d_properties(pos[: seg[1], 2])

                self.plotted_robots[name][
                    name + "tube" + str(cur_tube_num) + "_shade"
                ].set_data(
                    pos[: seg[1], 0],
                    pos[: seg[1], 1],
                )

                cur_tube_num += 1
        if show_forces:
            for i, p in enumerate(pos):
                transf = np.vstack(
                    [
                        np.hstack([rots[i].reshape(3, 3), p.reshape(-1, 1)]),
                        np.array([0, 0, 0, 1]),
                    ]
                )
                f_vector = np.hstack([wrenches[i][:3], np.ones(1)]).reshape(-1, 1)
                f_vector = transf @ f_vector
                self.add_arrow(
                    f_vector[0, 0],
                    f_vector[1, 0],
                    f_vector[2, 0],
                    p[0],
                    p[1],
                    p[2],
                    name=i,
                )
        self.set_axes_equal(self.ax)

    def show(self, frames, show_forces):
        for name in frames.keys():
            self.ani[name] = FuncAnimation(
                self.fig,
                partial(self.displ_fwd, name=name, show_forces=show_forces),
                frames=frames[name],
                blit=False,
            )
        plt.show()
        return self.fig, self.ani

    def stop_animations(self, name):
        if name in self.ani.keys():
            self.ani[name].event_source.stop()


class Scene:
    """
    The base class for all visualizations in this package
    """

    def __init__(self, engine):
        self.scene_objects = {}
        self.robots = {}
        self.robot_frames = {}

        self.engine = engine

        self._show_forces = False

    def add_robot(self, name, robot):
        if name in self.robots.keys():
            warnings.warn("robot name already used. Replace robot.")
        self.robots[name] = robot
        self.robot_frames[name] = []
        self.engine.add_robot(
            self.robots[name].get_radi_for_segments(),
            self.robots[name].num_of_tubes,
            name,
        )

    def update(self):
        for rob in self.robots.keys():
            positions, rotations, wrenches, _, _, _ = self.robots[rob].calc_fwd()
            segments = (self.robots[rob].get_segments(),)
            number_of_tubes = self.robots[rob].num_of_tubes
            self.robot_frames[rob].append(
                [positions, segments[0], number_of_tubes, rotations, wrenches]
            )

    def add_arrows(self, x, y, z, dx, dy, dz, color):
        self.engine.add_arrow(x, y, z, dx, dy, dz, color)

    def show(self):
        return self.engine.show(self.robot_frames, self._show_forces)

    def reset(self):
        for rob in self.robots.keys():
            self.robot_frames[rob] = []
            self.engine.stop_animations(rob)

    def show_forces(self, toggle=False):
        self._show_forces = toggle


class Robot:
    def __init__(self, model_class_factory_function, config_path):
        self.model = model_class_factory_function(config_path)
        self.shape = {}
        self.configs = []
        self.num_of_tubes = len(self.model.tubes)
        self.fwd_foo = self.model.fwd_kinematic

    def set_fwd_function(self, foo):
        self.fwd_foo = foo

    def set_config(self, alphas, betas):
        self.configs.append([alphas, betas])
        self.model.rotate(alphas)
        self.model.translate(betas)

    def calc_fwd(self):
        return self.fwd_foo()

    def apply_external_forces(self, gaussians):
        self.model.set_gaussians(gaussians)

    def release_external_forces(self):
        self.model.remove_gaussians()

    def apply_tip_wrench(self, wrench):
        self.fwd_foo = partial(self.model.push_end, wrench=wrench)

    def get_segments(self):
        return self.model.seg_indexes

    def get_radi_for_segments(self):
        return self.model.get_tube_outer_radii()
