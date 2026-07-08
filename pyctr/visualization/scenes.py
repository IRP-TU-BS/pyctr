# matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

from functools import partial
import numpy as np
import warnings


def tube_end_indices(seg_indexes, number_of_tubes):
    """
    For each tube (1-indexed, tube 1 = innermost/longest), determine the row index
    (exclusive, suitable for slicing ``pos[:end]``) up to which that tube is present in
    the stacked forward-integration output.

    ``seg_indexes`` looks like ``[0, (2, 3), (2, 11), (2, 13), (1, 21)]``: the first
    entry is the base row (0), and every following entry is
    ``(tubes_present_in_segment, cumulative_end_row_index)`` ordered base -> tip. Tube
    k is present in every segment whose tube-count is >= k, so its end index is the
    last such segment's cumulative end.

    :param seg_indexes: segment list as produced by ``ConcentricTubeContinuumRobot``
    :param number_of_tubes: total number of tubes in the robot
    :return: list of length ``number_of_tubes`` with an end index per tube, or ``None``
        for a tube that is never present (should not normally happen)
    """
    segments = seg_indexes[1:]
    ends = []
    for k in range(1, number_of_tubes + 1):
        end = None
        for count, seg_end in segments:
            if count >= k:
                end = seg_end
        ends.append(end)
    return ends


class Engine:
    def __init__(self) -> object:
        self.plotted_robots = {}

    def plot_robot_pos(self, frame, name="robot"):
        self.displ_fwd(frame, name)

    def displ_fwd(self, frame, name="robot", show_forces=False):
        pass


class VisPy(Engine):
    def __init__(self):
        super().__init__()
        try:
            from vispy import scene
        except ImportError as exc:
            raise ImportError(
                "vispy is required for the VisPy engine - pip install vispy"
            ) from exc
        self._scene_module = scene

        self.canvas = scene.SceneCanvas(
            title="PyCTCR - Vispy", size=(600, 600), bgcolor="white", show=True
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "arcball"

        self.tubes = {}

    def displ_fwd(self, frame, name="robot", show_forces=False):
        pos = frame[0]
        segments = frame[1]
        number_of_tubes = frame[2]

        for visual in self.tubes.get(name, []):
            visual.parent = None
        self.tubes[name] = []

        ends = tube_end_indices(segments, number_of_tubes)
        for tube_idx, end in enumerate(ends, start=1):
            if end is None:
                continue
            tube = self._scene_module.visuals.Tube(
                points=pos[:end, :],
                radius=self.radii[tube_idx - 1],
                parent=self.canvas.scene,
            )
            self.tubes[name].append(tube)

    def add_robot(self, radii, number_of_tubes, name="robot"):
        self.radii = radii
        self.tubes[name] = []

    def show(self, frames, show_forces=False):
        # vispy rendering here is a static snapshot (no animation support yet), so show
        # the last frame of each robot.
        all_positions = []
        for name in frames.keys():
            last_frame = frames[name][-1]
            self.displ_fwd(last_frame, name, show_forces=show_forces)
            all_positions.append(last_frame[0])

        if all_positions:
            pos = np.vstack(all_positions)
            span = np.ptp(pos, axis=0)
            margin = 0.1 * max(span.max(), 1e-6)
            mins = pos.min(axis=0) - margin
            maxs = pos.max(axis=0) + margin
            self.view.camera.set_range(
                x=[mins[0], maxs[0]], y=[mins[1], maxs[1]], z=[mins[2], maxs[2]]
            )
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
        self.arrows = {}
        self._bbox = None  # running [mins, maxs] over all rendered frames

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

    def _update_axes_limits(self, pos):
        """
        Grow a running bounding box over every position seen so far (plus the z=0
        ground plane, so the shadow stays visible) and apply it, with a margin, to the
        axes. Matplotlib's 3D line artists do not autoscale on set_data, so this has to
        be done explicitly.
        """
        pos_with_ground = np.vstack([pos, np.zeros((1, 3))])
        mins = pos_with_ground.min(axis=0)
        maxs = pos_with_ground.max(axis=0)
        if self._bbox is None:
            self._bbox = [mins.copy(), maxs.copy()]
        else:
            self._bbox[0] = np.minimum(self._bbox[0], mins)
            self._bbox[1] = np.maximum(self._bbox[1], maxs)

        bbox_min, bbox_max = self._bbox
        span = np.maximum(bbox_max - bbox_min, 1e-6)
        margin = 0.1 * span
        lo = bbox_min - margin
        hi = bbox_max + margin
        self.ax.set_xlim3d(lo[0], hi[0])
        self.ax.set_ylim3d(lo[1], hi[1])
        self.ax.set_zlim3d(lo[2], hi[2])

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
        for name in list(self.arrows.keys()):
            self.arrows[name].remove()
        self.arrows.clear()

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
        wrenches = frame[4] if len(frame) > 4 else None

        ends = tube_end_indices(segments, number_of_tubes)
        for tube_idx, end in enumerate(ends, start=1):
            if end is None:
                continue
            seg_pos = pos[:end]

            tube_line = self.plotted_robots[name][name + "tube" + str(tube_idx)]
            tube_line.set_data(seg_pos[:, 0], seg_pos[:, 1])
            tube_line.set_3d_properties(seg_pos[:, 2])

            shade_line = self.plotted_robots[name][
                name + "tube" + str(tube_idx) + "_shade"
            ]
            shade_line.set_data_3d(
                seg_pos[:, 0], seg_pos[:, 1], np.zeros(seg_pos.shape[0])
            )

        if show_forces and wrenches is not None:
            self.remove_all_arrows()
            bbox_diag = np.linalg.norm(pos.max(axis=0) - pos.min(axis=0))
            arrow_length = 0.1 * bbox_diag if bbox_diag > 1e-9 else 1.0
            step = max(1, len(pos) // 20)
            for i in range(0, len(pos), step):
                force = wrenches[i][:3]
                force_norm = np.linalg.norm(force)
                if force_norm < 1e-9:
                    continue
                direction = force / force_norm * arrow_length
                self.add_arrow(
                    pos[i, 0],
                    pos[i, 1],
                    pos[i, 2],
                    direction[0],
                    direction[1],
                    direction[2],
                    name=i,
                )
        else:
            self.remove_all_arrows()

        self._update_axes_limits(pos)
        self.set_axes_equal(self.ax)

        return list(self.plotted_robots[name].values()) + list(self.arrows.values())

    def show(self, frames, show_forces=False):
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

    def add_arrows(self, x, y, z, dx, dy, dz, color, name="tmp"):
        self.engine.add_arrow(x, y, z, dx, dy, dz, name=name, color=color)

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
