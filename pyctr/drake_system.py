"""
Drake system for DynamicConcentricTubeRobot. Needs the drake package (pip install drake).
"""

import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.geometry import (
    Capsule,
    CollisionFilterDeclaration,
    FramePoseVector,
    GeometryFrame,
    GeometryInstance,
    GeometrySet,
    HalfSpace,
    MakePhongIllustrationProperties,
    ProximityProperties,
    QueryObject,
    Role,
)
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.math import SpatialForce, SpatialVelocity
from pydrake.multibody.plant import ExternallyAppliedSpatialForce
from pydrake.multibody.tree import BodyIndex
from pydrake.systems.framework import LeafSystem
from scipy.spatial.transform import Rotation


def _spatial_force(body_index, p_BoBq_B, torque_W, force_W):
    force = ExternallyAppliedSpatialForce()
    force.body_index, force.p_BoBq_B, force.F_Bq_W = body_index, p_BoBq_B, SpatialForce(torque_W, force_W)
    return force


class ConcentricTubeRobotSystem(LeafSystem):
    """
    Discrete system with period robot.dt, every update calls robot.step(). Call robot.init_static() before, with
    robot.p_init, robot.R_init at the initial pose of the base plate.
    The robot state is stored in the discrete state of the context, the robot object is only used for computing.
    Inputs are sampled at the beginning of every step.

    Input ports (optional, unconnected ports use the initial alphas and betas, a zero wrench and no contact):
        alphas: tube angles (k)
        betas: tube translations (k), see robot.translate; changes give a new grid of the robot model
        tip_wrench: [F, M] at the tip in the world frame (6)
        geometry_query: QueryObject of the scene_graph, for frictionless contact with the geometries that have a
            proximity role when the system is created (register obstacles and finalize a plant before). Per interval
            the closest geometry within contact_margin becomes the contact plane of the robot model, see
            DynamicConcentricTubeRobot. The margin must be larger than the motion within one step. The robot
            geometries are excluded from all other contact queries, a MultibodyPlant does not see them.
        body_poses, body_spatial_velocities: only with plant, from its output ports of the same name. The base
            plate is mounted on base_body at X_BodyBase (its z axis along the robot).
    Output ports:
        positions: backbone positions of all nodes, (M, 3) flattened
        geometry_pose: only with scene_graph, poses of the capsules: every tube is a chain of capsules (cylinder part
            piece_length, default: the largest node distance) along the backbone, the caps end at the base plate and
            at the tube end; capsules beyond the end lie on the last one (shapes cannot change their size, the tube
            ends move with betas). A tube shorter than piece_length plus its diameter sticks out of its capsule.
        applied_spatial_force: only with plant, for its applied_spatial_force input: the reaction of the robot on
            base_body and the reaction of the contact forces on bodies of the plant
    """

    def __init__(
        self,
        robot,
        scene_graph=None,
        plant=None,
        base_body=None,
        X_BodyBase=RigidTransform(),
        name="concentric_tube_robot",
        color=(0.75, 0.75, 0.75, 1.0),
        contact_margin=5e-3,
        piece_length=None,
    ):
        super().__init__()
        if not hasattr(robot, "s"):
            raise ValueError("call robot.init_static() before creating the system")
        self.set_name(name)
        self._robot = robot
        self._margin = contact_margin
        self._alphas0, self._betas0 = robot.alphas.copy(), robot.betas.copy()
        self._shape = robot.y.shape
        intervals = self._shape[0] - 1

        self._x = self.DeclareDiscreteState(robot.get_state())
        # reaction of the contact force per interval: body index (-1: none), point in the body frame, force (world)
        self._reactions = self.DeclareDiscreteState(np.tile([-1.0, 0, 0, 0, 0, 0, 0], intervals))
        self.DeclarePeriodicDiscreteUpdateEvent(robot.dt, 0.0, self._update)
        self._alphas = self.DeclareVectorInputPort("alphas", len(robot.tubes))
        self._betas = self.DeclareVectorInputPort("betas", len(robot.tubes))
        self._wrench = self.DeclareVectorInputPort("tip_wrench", 6)
        self._query = self.DeclareAbstractInputPort("geometry_query", AbstractValue.Make(QueryObject()))
        state = {self.all_state_ticket()}  # outputs depend on the state only, not on the inputs
        self.DeclareVectorOutputPort("positions", self._shape[0] * 3, self._calc_positions, state)

        self._piece = {}  # robot geometry id -> index of the capsule
        self._obstacles = {}  # obstacle geometry id -> is a half space
        if scene_graph is not None:
            inspector = scene_graph.model_inspector()
            others = inspector.GetAllGeometryIds(Role.kProximity)
            if plant is not None and base_body is not None:  # the robot leaves its base body, no contact with it
                body_of = lambda g: plant.GetBodyFromFrameId(inspector.GetFrameId(g))
                others_base = [g for g in others if body_of(g) is not None and body_of(g).index() == base_body.index()]
            else:
                others_base = []
            self._obstacles = {g: isinstance(inspector.GetShape(g), HalfSpace) for g in others if g not in others_base}
            self.source_id = scene_graph.RegisterSource(name)
            self._length = piece_length or np.diff(robot.s).max()
            self._L = np.array([tube.params["L"] for _, tube in robot.tubes])
            r_outer = [tube.params["r_outer"] for _, tube in robot.tubes]
            count = np.ceil(self._L / self._length).astype(int)  # enough for the whole tube (beta = 0)
            self._piece_tube = np.repeat(np.arange(len(count)), count)
            self._piece_index = np.concatenate([np.arange(c) for c in count])
            self._piece_radius = np.array(r_outer)[self._piece_tube]
            self._frames = []
            for q, (i, m) in enumerate(zip(self._piece_tube, self._piece_index)):
                frame = scene_graph.RegisterFrame(self.source_id, GeometryFrame(f"{name}_{i}_{m}"))
                capsule = Capsule(self._piece_radius[q], self._length)
                geometry = scene_graph.RegisterGeometry(
                    self.source_id, frame, GeometryInstance(RigidTransform(), capsule, f"{name}_{i}_{m}")
                )
                scene_graph.AssignRole(self.source_id, geometry, MakePhongIllustrationProperties(color))
                scene_graph.AssignRole(self.source_id, geometry, ProximityProperties())
                self._frames.append(frame)
                self._piece[geometry] = q
            self._proximity_count = len(others) + len(self._piece)
            robot_set = GeometrySet(list(self._piece))
            declaration = CollisionFilterDeclaration().ExcludeWithin(robot_set)
            if others:
                declaration.ExcludeBetween(robot_set, GeometrySet(others))
            scene_graph.collision_filter_manager().Apply(declaration)
            self.DeclareAbstractOutputPort(
                "geometry_pose", lambda: AbstractValue.Make(FramePoseVector()), self._calc_frame_poses, state
            )

        self._plant = plant
        if plant is not None:
            self._base_body, self._X_BodyBase = base_body, X_BodyBase
            self._obstacle_body = {}  # obstacle geometry id -> moving body of the plant
            for g in self._obstacles:
                body = plant.GetBodyFromFrameId(scene_graph.model_inspector().GetFrameId(g))
                if body is not None and body.index() != plant.world_body().index():
                    self._obstacle_body[g] = body
            self._poses = self.DeclareAbstractInputPort("body_poses", AbstractValue.Make([RigidTransform()]))
            self._velocities = self.DeclareAbstractInputPort(
                "body_spatial_velocities", AbstractValue.Make([SpatialVelocity()])
            )
            self.DeclareAbstractOutputPort(
                "applied_spatial_force",
                lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
                self._calc_applied_force,
                state,
            )

    def _y(self, context):
        y = context.get_discrete_state(self._x).get_value()[: self._shape[0] * self._shape[1]]
        return y.reshape(self._shape)

    def _arcs(self, context):
        """Arc lengths (start, end) of the cylinder parts of the capsules for the betas in the state"""
        betas = context.get_discrete_state(self._x).get_value()[-len(self._L) :]
        r, end = self._piece_radius, (self._L * (1 - betas))[self._piece_tube]
        start = np.maximum(np.minimum(self._piece_index * self._length, end - self._length - r), r)
        return start, start + self._length

    def _set_base(self, context):
        """Base plate pose (extrapolated to the end of the step, the robot model is implicit) and velocity"""
        index = self._base_body.index()
        X_WB = self._poses.Eval(context)[index]
        X_WP = X_WB.multiply(self._X_BodyBase)
        V_WP = self._velocities.Eval(context)[index].Shift(X_WB.rotation().multiply(self._X_BodyBase.translation()))
        v, w = V_WP.translational(), V_WP.rotational()
        robot = self._robot
        robot.p_init = (X_WP.translation() + v * robot.dt).reshape(3, 1)
        robot.R_init = Rotation.from_rotvec(w * robot.dt).as_matrix() @ X_WP.rotation().matrix()
        robot.base_velocity = (v, w)

    def _contacts(self, context):
        """Closest obstacle per interval: normals, points for the robot model, obstacle ids and surface points"""
        if not self._query.HasValue(context):
            return None
        query = self._query.Eval(context)
        if len(query.inspector().GetAllGeometryIds(Role.kProximity)) != self._proximity_count:
            raise RuntimeError(
                "geometries with proximity role were registered after the ConcentricTubeRobotSystem was created, "
                "register obstacles (and finalize a MultibodyPlant) before creating it"
            )
        if not self._obstacles:
            return None
        start, end = self._arcs(context)  # capsules as in the scene graph, s and p of the grid of the new betas
        s, p = self._robot.s, self._robot.y[:, :3]
        intervals = len(s) - 1
        interval = np.clip(np.searchsorted(s, 0.5 * (start + end)) - 1, 0, intervals - 1)  # interval of each capsule
        normals, points, surface = np.zeros((intervals, 3)), np.zeros((intervals, 3)), np.zeros((intervals, 3))
        ids, distances = [None] * intervals, np.full(intervals, np.inf)
        half, radius = self._length / 2, self._robot.radius
        for robot_id, q in self._piece.items():
            j, r = interval[q], self._piece_radius[q]
            X_WR = query.GetPoseInWorld(robot_id)
            centre, axis = X_WR.translation(), X_WR.rotation().col(2)
            for other, is_half_space in self._obstacles.items():
                if is_half_space:  # not supported by the signed distance query of SceneGraph
                    X_WH = query.GetPoseInWorld(other)
                    normal = X_WH.rotation().col(2)
                    z = -np.sign(normal @ axis) * half  # capsule end closest to the plane
                    distance = normal @ (centre + z * axis - X_WH.translation()) - r
                    p_other = centre + z * axis - (distance + r) * normal
                else:
                    pair = query.ComputeSignedDistancePairClosestPoints(robot_id, other)
                    distance = pair.distance
                    if pair.id_A == robot_id:
                        p_robot, p_other = pair.p_ACa, query.GetPoseInWorld(other) @ pair.p_BCb
                    else:
                        p_robot, p_other = pair.p_BCb, query.GetPoseInWorld(other) @ pair.p_ACa
                    z = np.clip(p_robot[2], -half, half)  # closest point on the capsule axis
                    # normal from the obstacle to the axis (nhat_BA_W can have the wrong sign near zero distance);
                    # the depth is far below the radius, so the axis stays outside the obstacle
                    normal = centre + z * axis - p_other
                    normal = normal / np.linalg.norm(normal)
                if distance < min(distances[j], self._margin):
                    # the model measures the depth from the midpoint of interval j with its radius: move the point so
                    # that the model depth equals the depth of this capsule now
                    middle = 0.5 * (p[j] + p[j + 1])
                    distances[j], normals[j], ids[j], surface[j] = distance, normal, other, p_other
                    points[j] = p_other - (centre + z * axis - middle) - (radius[j] - r) * normal
        return normals, points, ids, surface

    def _update(self, context, discrete_state):
        robot = self._robot
        robot.set_state(context.get_discrete_state(self._x).get_value())
        robot.rotate(self._alphas.Eval(context) if self._alphas.HasValue(context) else self._alphas0)
        robot.translate(self._betas.Eval(context) if self._betas.HasValue(context) else self._betas0)
        robot.remesh()  # contacts on the grid of the new betas
        wrench = self._wrench.Eval(context) if self._wrench.HasValue(context) else np.zeros(6)
        if self._plant is not None:
            self._set_base(context)
        contacts = self._contacts(context)
        robot.step(wrench, None if contacts is None else contacts[:2])
        discrete_state.get_mutable_vector(self._x).SetFromVector(robot.get_state())

        if self._plant is not None and contacts is not None:
            reactions = np.tile([-1.0, 0, 0, 0, 0, 0, 0], (len(contacts[2]), 1))
            poses = self._poses.Eval(context)
            for j, (other, p_W) in enumerate(zip(contacts[2], contacts[3])):
                body = self._obstacle_body.get(other)
                if body is not None and robot.contact_force[j].any():
                    p_B = poses[body.index()].inverse() @ p_W
                    reactions[j] = [int(body.index()), *p_B, *(-robot.contact_force[j])]
            discrete_state.get_mutable_vector(self._reactions).SetFromVector(reactions.ravel())

    def _calc_positions(self, context, output):
        output.SetFromVector(self._y(context)[:, :3].ravel())

    def _calc_frame_poses(self, context, output):
        start, end = self._arcs(context)
        self._robot.set_state(context.get_discrete_state(self._x).get_value())  # grid of the betas in the state
        s, p = self._robot.s, self._y(context)[:, :3]
        a = np.column_stack([np.interp(start, s, c) for c in p.T])
        b = np.column_stack([np.interp(end, s, c) for c in p.T])
        poses = output.get_mutable_value()
        for q, frame in enumerate(self._frames):
            axis = RotationMatrix.MakeFromOneVector(b[q] - a[q], 2)  # capsule z axis along the backbone
            poses.set_value(frame, RigidTransform(axis, 0.5 * (a[q] + b[q])))

    def _calc_applied_force(self, context, output):
        y, k = self._y(context), len(self._robot.tubes)
        moment = y[0, 3:12].reshape(3, 3) @ np.r_[y[0, 21:23], y[0, 23 : 23 + k].sum()]  # m(0) in the world frame
        forces = [_spatial_force(self._base_body.index(), self._X_BodyBase.translation(), moment, y[0, 18:21])]
        for body, *values in context.get_discrete_state(self._reactions).get_value().reshape(-1, 7):
            if body >= 0:
                forces.append(_spatial_force(BodyIndex(int(body)), values[:3], np.zeros(3), values[3:]))
        output.set_value(forces)
