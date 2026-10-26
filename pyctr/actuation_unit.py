"""
Model of the IRP actuation unit (https://github.com/IRP-TU-BS/irp_concentric_tube_continuum_robot) for Drake.
The parts of Models/stl_files of that repository are placed as in its drawings (Container, AdapterEnd, Ctcr_v001).
Motors (NEMA 14) and guide rails have no STL file and are boxes; the gears on the square shafts are left out.
add_actuation_unit registers the parts as visual geometry of a MultibodyPlant body; ActuationUnitCarriages moves the
carriages with the tube translations (betas) and turns the tube shafts with the tube angles (alphas).
Needs the drake package and the STL files, which are converted to OBJ for Meshcat (it does not show STL).
"""

import hashlib
import os
import tempfile

import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.geometry import Box, Cylinder, FramePoseVector, GeometryFrame, GeometryInstance, Mesh
from pydrake.geometry import MakePhongIllustrationProperties
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.systems.framework import LeafSystem

# Housing frame C in mm: x across, y up (0 = bottom of the housing), z along the unit from the front face of
# AdapterEnd (z = 0, tube exit) to the motor end. Plates and carriages share their STL frame: their hole grid is at
# x = 0, 36, 72 and y = 0, -36.2, the tube hole at (36, 36), so they are shifted by PLATE.
PLATE = np.array([-36.0, 79.2, 0.0])
TUBE_AXIS = np.array([0.0, 115.2])  # x, y of the tube hole (Ø5) in C
SLEDGES_Z = (254.0, 184.0, 110.0)  # front faces of the carriage plates, rear to front, side view of Ctcr_v001
SPINDLES_X, SPINDLE_Y, SHAFT_Y = (-36.0, 0.0, 36.0), 43.0, 79.2  # trapezoidal spindles lower row, square shafts upper
# travel of the carriage front faces: the collet on the tube shaft (20.4 mm in front of the face, no STL) behind
# AdapterEnd (0..10), the gear and the foot (29.5 mm behind the face) in front of AdapterMid (310); on the tube axis
# the collet of a carriage meets the gear of the carriage in front of it
TRAVEL, SPACING = (31.0, 280.0), 50.0

WHITE, ALUMINIUM, STEEL, DARK = (0.93, 0.93, 0.9, 1.0), (0.72, 0.74, 0.77, 1.0), (0.55, 0.56, 0.58, 1.0), (0.15, 0.15, 0.17, 1.0)
_I = RotationMatrix()
_R_y = RotationMatrix.MakeYRotation(np.pi / 2)  # STL x axis -> -z of C (gear and tube shaft: thread to the exit)

# unit frame U: origin at the front bottom centre of the housing, x out of the tube hole, z up
X_UC = RigidTransform(RotationMatrix(np.array([[0, 0, -1.0], [-1, 0, 0], [0, 1, 0]])), np.zeros(3))
# tube exit E in C: z out of the unit, x up (alpha = 0 curves the robot upwards)
X_CE = RigidTransform(RotationMatrix(np.array([[0, 1.0, 0], [1, 0, 0], [0, 0, -1]])), [*TUBE_AXIS * 1e-3, 0])


def _fixed_parts():
    """(STL file or Box, rotation, translation in mm, colour) of the parts that do not move, in C"""
    parts = [
        ("ContainerBottom.stl", _I, (0, 5, 0), ALUMINIUM),
        ("ContainerLeft.stl", _I, (-59.5, 22.5, 0), ALUMINIUM),
        ("ContainerRight.stl", _I, (59.5, 22.5, 0), ALUMINIUM),
        ("AdapterEnd.stl", _I, PLATE, WHITE),
        ("AdapterMid.stl", _I, PLATE + (0, 0, 310), WHITE),
        ("AdapterMotor.stl", _I, PLATE + (0, 0, 350), WHITE),
    ]
    for x in (-59.5, 59.5):  # igus drylin TS-04-07 guide rails on the walls (no STL)
        parts.append((Box(7, 4, 300), _I, (x, 47, 160), STEEL))
    for x in SPINDLES_X:
        parts.append(("TrapezoidalThreadSpindle.stl", _I, (x, SPINDLE_Y, 0), STEEL))
        parts.append(("SquareShaft_4x330.stl", _I, (x, SHAFT_Y, 0), STEEL))
        for y in (SPINDLE_Y, SHAFT_Y):  # NEMA 14 stepper motors behind AdapterMotor (no STL)
            parts.append((Box(35.2, 35.2, 34), _I, (x, y, 377), DARK))
    return parts


def _carriage_parts(c):
    """Parts of carriage c (rear to front) relative to its front face at z = 0: (sliding, turning with the tube)"""
    sliding = [  # flange nut: body through the plate, flange behind it
        ("Sledge_V2.1.7.2.stl", _I, PLATE, WHITE),
        ("ThreadedNutWithQuadFlange.stl", RotationMatrix.MakeYRotation(np.pi), (SPINDLES_X[c], SPINDLE_Y, 14.5), STEEL),
    ]
    turning = [  # Shaft1: collet thread in front of the plate, flange behind it, gear on the flange
        ("Shaft1.stl", _R_y, (*TUBE_AXIS, 20.5), WHITE),
        ("6mmGear_6mmHole.stl", _R_y, (TUBE_AXIS[0] + 5.49, TUBE_AXIS[1] - 1.59, 26.5), WHITE),  # bore at y, z = 1.59, -5.49
    ]
    return sliding, turning


def read_stl(stl_path):
    """vertices (n, 3) and faces (m, 3) of a binary STL file"""
    data = np.fromfile(stl_path, dtype=np.uint8)
    n = int(data[80:84].view("<u4")[0])
    if len(data) != 84 + 50 * n:
        raise ValueError(f"{stl_path} is not a binary STL file")
    triangle = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    triangles = np.frombuffer(data[84:].tobytes(), dtype=triangle)["vertices"].reshape(-1, 3)
    vertices, faces = np.unique(triangles, axis=0, return_inverse=True)
    return vertices.astype(float), faces.reshape(-1, 3)


def _obj(stl_path, cache_dir):
    """OBJ copy of a binary STL file (Meshcat does not show STL), named after the STL path, size and time"""
    stat = os.stat(stl_path)  # copies of different STL files with the same name must not share an OBJ file
    key = hashlib.sha1(f"{os.path.abspath(stl_path)}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:10]
    obj = os.path.join(cache_dir, f"{os.path.splitext(os.path.basename(stl_path))[0]}_{key}.obj")
    if os.path.exists(obj):
        return obj
    vertices, faces = read_stl(stl_path)
    os.makedirs(cache_dir, exist_ok=True)
    with open(obj + ".tmp", "w") as f:
        np.savetxt(f, vertices, fmt="v %.6g %.6g %.6g")
        np.savetxt(f, faces + 1, fmt="f %d %d %d")
    os.replace(obj + ".tmp", obj)
    return obj


def _shape(part, stl_dir, cache_dir):
    if isinstance(part, Box):
        return Box(part.width() * 1e-3, part.depth() * 1e-3, part.height() * 1e-3)
    return Mesh(_obj(os.path.join(stl_dir, part), cache_dir or os.path.join(tempfile.gettempdir(), "pyctr_meshes")), 1e-3)


def _pose(R, p):
    return RigidTransform(R, np.asarray(p, dtype=float) * 1e-3)


def add_actuation_unit(
    plant, stl_dir, body=None, X_BodyUnit=RigidTransform(), tube_radii=(), cache_dir=None, moving_parts=False
):
    """
    Registers the actuation unit as visual geometry of body (default: the world) of plant, before plant.Finalize().
    :param stl_dir: Models/stl_files of the irp_concentric_tube_continuum_robot repository
    :param X_BodyUnit: pose of the unit frame in body: origin at the front bottom centre of the housing, x out of
        the tube hole, z up. With the default the unit stands on the ground and the tubes leave along x.
    :param tube_radii: outer radii of the tubes, inner tube first (m). They are drawn inside the unit from their
        carriage to the tube hole, the tubes are clamped in the front carriages (as in ActuationUnitCarriages).
    :param moving_parts: False: carriages and tubes at fixed places (overall drawing). True: only the fixed parts,
        the carriages and tubes come from ActuationUnitCarriages.
    :return: X_BodyExit, pose of the tube exit in body: the base plate of the robot model (z along the tubes, x up).
        Use it as robot.p_init, robot.R_init (body = world) or as X_BodyBase of ConcentricTubeRobotSystem.
    """
    body = plant.world_body() if body is None else body
    X_BodyC = X_BodyUnit.multiply(X_UC)
    parts = _fixed_parts()
    if not moving_parts:
        for c, z in enumerate(SLEDGES_Z):
            sliding, turning = _carriage_parts(c)
            parts += [(part, R, np.asarray(p) + (0, 0, z), color) for part, R, p, color in sliding + turning]
    for j, (part, R, p, color) in enumerate(parts):
        X = X_BodyC.multiply(_pose(R, p))
        plant.RegisterVisualGeometry(body, X, _shape(part, stl_dir, cache_dir), f"actuation_unit_{j}", np.array(color))
    k = len(tube_radii)
    for i, r in enumerate(tube_radii if not moving_parts else ()):  # from its carriage to the tube hole
        length = SLEDGES_Z[len(SLEDGES_Z) - k + i] * 1e-3
        X = X_BodyC.multiply(RigidTransform([*TUBE_AXIS * 1e-3, length / 2]))
        plant.RegisterVisualGeometry(body, X, Cylinder(r, length), f"actuation_unit_tube_{i}", np.array(WHITE))
    return X_BodyC.multiply(X_CE)


class ActuationUnitCarriages(LeafSystem):
    """
    Moving parts of the actuation unit, use with add_actuation_unit(..., moving_parts=True) and the same body and
    X_BodyUnit. The k tubes (inner first) are clamped in the k front carriages, carriages without a tube stay at the
    rear. The front face of the carriage of tube i is at z = d_i + L_i beta_i behind the tube hole: d_i is the tube
    length inside the unit at beta_i = 0 (transmission_lengths, default: carriages packed at the front). A ValueError
    is raised when a carriage leaves its travel (TRAVEL) or two carriages come closer than SPACING.
    The tube shaft and gear of carriage i turn with alpha_i. The tubes inside the unit are chains of cylinders.

    Input ports: betas, alphas (k, unconnected: the values of robot at construction); body_poses (only with plant,
        from its output port of the same name).
    Output port: geometry_pose, for scene_graph.get_source_pose_port(source_id).
    """

    def __init__(
        self,
        scene_graph,
        stl_dir,
        robot,
        plant=None,
        body=None,
        X_BodyUnit=RigidTransform(),
        transmission_lengths=None,
        cache_dir=None,
        name="actuation_unit_carriages",
        piece_length=10e-3,
    ):
        super().__init__()
        self.set_name(name)
        k = len(robot.tubes)
        if not 1 <= k <= len(SLEDGES_Z):
            raise ValueError(f"the actuation unit has {len(SLEDGES_Z)} carriages, the robot has {k} tubes")
        self._L = np.array([tube.params["L"] for _, tube in robot.tubes]) * 1e3  # mm
        default = TRAVEL[0] + SPACING * np.arange(k - 1, -1, -1)
        self._d = default if transmission_lengths is None else np.asarray(transmission_lengths, dtype=float) * 1e3
        self._parked = TRAVEL[1] - SPACING * np.arange(len(SLEDGES_Z) - k)  # carriages without a tube, rear first
        self._alphas0, self._betas0 = np.array(robot.alphas, dtype=float), np.array(robot.betas, dtype=float)
        self._X_BodyC = X_BodyUnit.multiply(X_UC)
        self._carriage_z(self._betas0)  # check the initial betas

        self.source_id = scene_graph.RegisterSource(name)
        self._geometry = []  # (frame, kind, index, pose in its moving frame)

        def register(shape, color, kind, index, X):
            frame = scene_graph.RegisterFrame(self.source_id, GeometryFrame(f"{name}_{len(self._geometry)}"))
            g = scene_graph.RegisterGeometry(self.source_id, frame, GeometryInstance(RigidTransform(), shape, f"{name}_{len(self._geometry)}"))
            scene_graph.AssignRole(self.source_id, g, MakePhongIllustrationProperties(np.array(color)))
            self._geometry.append((frame, kind, index, X))

        for c in range(len(SLEDGES_Z)):
            sliding, turning = _carriage_parts(c)
            for kind, parts in (("sliding", sliding), ("turning", turning)):
                for part, R, p, color in parts:
                    register(_shape(part, stl_dir, cache_dir), color, kind, c, _pose(R, p))
        self._length = piece_length
        for i, tube in enumerate(robot.tubes):
            for m in range(int(np.ceil(TRAVEL[1] * 1e-3 / piece_length))):
                register(Cylinder(tube[1].params["r_outer"], piece_length), WHITE, "tube", (i, m), RigidTransform())

        self._betas = self.DeclareVectorInputPort("betas", k)
        self._alphas = self.DeclareVectorInputPort("alphas", k)
        self._plant_body = None
        if plant is None and body is not None:
            raise ValueError("body needs plant (its body_poses output gives the pose of the unit)")
        if plant is not None and body is not None and body.index() != plant.world_body().index():
            self._plant_body = body
            self._poses = self.DeclareAbstractInputPort("body_poses", AbstractValue.Make([RigidTransform()]))
        self.DeclareAbstractOutputPort("geometry_pose", lambda: AbstractValue.Make(FramePoseVector()), self._calc_poses)

    def _carriage_z(self, betas):
        """Front face positions of all carriages (mm, rear to front), checked"""
        z = np.r_[self._parked, self._d + self._L * np.asarray(betas)]
        if z.min() < TRAVEL[0] or z.max() > TRAVEL[1] or (np.diff(z) > -SPACING).any():
            raise ValueError(
                f"carriage positions {np.round(z, 1)} mm (rear to front) for betas {np.round(betas, 4)}: travel "
                f"{TRAVEL} mm, spacing at least {SPACING} mm"
            )
        return z

    def _calc_poses(self, context, output):
        k = len(self._L)
        betas = self._betas.Eval(context) if self._betas.HasValue(context) else self._betas0
        alphas = self._alphas.Eval(context) if self._alphas.HasValue(context) else self._alphas0
        z = self._carriage_z(betas)
        X_WC = self._X_BodyC
        if self._plant_body is not None:
            X_WC = self._poses.Eval(context)[self._plant_body.index()].multiply(self._X_BodyC)
        tube_of_carriage = {len(z) - k + i: i for i in range(k)}
        axis = np.array([*TUBE_AXIS, 0.0]) * 1e-3
        poses = output.get_mutable_value()
        for frame, kind, index, X in self._geometry:
            if kind == "tube":  # piece m of tube i, from the tube hole back to the carriage
                i, m = index
                depth = z[len(z) - k + i] * 1e-3
                start = min(m * self._length, max(depth - self._length, 0))
                X_C = RigidTransform(axis + (0, 0, start + self._length / 2))
            else:
                X_C = RigidTransform([0, 0, z[index] * 1e-3]).multiply(X)
                i = tube_of_carriage.get(index)
                if kind == "turning" and i is not None:  # turn about the tube axis, alpha about -z of C (exit frame)
                    X_C = RigidTransform(axis).multiply(RigidTransform(RotationMatrix.MakeZRotation(-alphas[i])))
                    X_C = X_C.multiply(RigidTransform(-axis)).multiply(RigidTransform([0, 0, z[index] * 1e-3])).multiply(X)
            poses.set_value(frame, X_WC.multiply(X_C))
