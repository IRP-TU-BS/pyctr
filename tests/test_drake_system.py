import copy
import unittest
import numpy as np
from pyctr.cosserat_rod import CurvedCosseratRod
from pyctr.cosserat_rod_dynamics import DynamicConcentricTubeRobot

try:
    from pydrake.geometry import Box, GeometryInstance, ProximityProperties, SceneGraph, Sphere
    from pydrake.math import RigidTransform
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, CoulombFriction
    from pydrake.multibody.tree import SpatialInertia
    from pydrake.systems.analysis import Simulator
    from pydrake.systems.framework import DiagramBuilder
    from pydrake.systems.primitives import TrajectorySource
    from pydrake.trajectories import PiecewisePolynomial
    from pyctr.drake_system import ConcentricTubeRobotSystem
except ImportError:
    SceneGraph = None


def straight_robot(**kwargs):
    """Two straight tubes of 0.1 m along the z axis of the base plate"""
    nylon = {"E": 339889091.1107947, "G": 97898219.42862949, "rho": 1140, "g": 0, "kappa": 0.0, "s": 5e-3}
    tubes = [
        CurvedCosseratRod({**nylon, "L": 0.1, "r_inner": ri, "r_outer": ro})
        for ri, ro in ((0.5e-3, 1.25e-3), (1.5e-3, 2.5e-3))
    ]
    return DynamicConcentricTubeRobot(tubes, dt=1e-3, **kwargs)


def connect_to_plant(builder, plant, scene_graph, ctr):
    builder.Connect(ctr.GetOutputPort("geometry_pose"), scene_graph.get_source_pose_port(ctr.source_id))
    builder.Connect(scene_graph.get_query_output_port(), ctr.GetInputPort("geometry_query"))
    builder.Connect(plant.get_body_poses_output_port(), ctr.GetInputPort("body_poses"))
    builder.Connect(plant.get_body_spatial_velocities_output_port(), ctr.GetInputPort("body_spatial_velocities"))
    builder.Connect(ctr.GetOutputPort("applied_spatial_force"), plant.get_applied_spatial_force_input_port())


@unittest.skipIf(SceneGraph is None, "drake is not installed")
class TestConcentricTubeRobotSystem(unittest.TestCase):
    def test_simulator_matches_direct_steps(self):
        nylon = {"E": 339889091.1107947, "G": 97898219.42862949, "rho": 1140, "g": 0, "kappa": 11.636, "s": 5e-3}
        tubes = [
            CurvedCosseratRod({**nylon, "L": L, "straight_length": 0.03, "r_inner": ri, "r_outer": ro})
            for L, ri, ro in ((0.1, 0.5e-3, 1.25e-3), (0.06, 1.5e-3, 2.5e-3))
        ]
        robot = DynamicConcentricTubeRobot(tubes, dt=2e-3)
        robot.init_static()
        reference = copy.deepcopy(robot)

        builder = DiagramBuilder()
        scene_graph = builder.AddSystem(SceneGraph())
        ctr = builder.AddSystem(ConcentricTubeRobotSystem(robot, scene_graph))
        builder.Connect(ctr.GetOutputPort("geometry_pose"), scene_graph.get_source_pose_port(ctr.source_id))
        simulator = Simulator(builder.Build())
        context = ctr.GetMyMutableContextFromRoot(simulator.get_mutable_context())
        alphas, wrench = np.array([0.0, 1.0]), np.array([0, 0.01, 0, 0, 0, 0])
        ctr.GetInputPort("alphas").FixValue(context, alphas)
        ctr.GetInputPort("tip_wrench").FixValue(context, wrench)
        simulator.AdvanceTo(4.5 * robot.dt)  # updates at t = 0, dt, 2 dt, 3 dt, 4 dt

        reference.rotate(alphas)
        for _ in range(5):
            p_ref = reference.step(wrench)[0]
        p = ctr.GetOutputPort("positions").Eval(context).reshape(-1, 3)
        np.testing.assert_allclose(p, p_ref, atol=1e-12)

        # every tube is a chain of capsules on the backbone, the cap of the last one ends at the tube end
        poses = ctr.GetOutputPort("geometry_pose").Eval(context)
        length, s = ctr._length, reference.s
        self.assertEqual(poses.size(), sum(int(np.ceil(L / length)) for L in (0.1, 0.06)))
        fine = np.linspace(0, s[-1], 20001)
        polyline = np.column_stack([np.interp(fine, s, c) for c in p.T])
        for i, L in enumerate((0.1, 0.06)):
            frames = [(f, r) for f, tube, r in zip(ctr._frames, ctr._piece_tube, ctr._piece_radius) if tube == i]
            ends = []
            for frame, r in frames:
                X = poses.value(frame)
                self.assertLess(np.linalg.norm(polyline - X.translation(), axis=1).min(), 1e-4)
                ends.append(X.translation() + X.rotation().col(2) * (length / 2 + r))
            tip = [np.interp(L, s, c) for c in p.T]
            # the capsule and its cap follow the chord, not the curved backbone: about r^2 kappa off the tip
            self.assertLess(np.linalg.norm(np.array(ends) - tip, axis=1).min(), 1e-4)

    def test_contact_with_box_in_scene_graph(self):
        # straight robot along z, a tip force pushes it in +x against a fixed box whose face is 1 mm away
        nylon = {"E": 339889091.1107947, "G": 97898219.42862949, "rho": 1140, "g": 0, "kappa": 0.0, "s": 5e-3}
        tubes = [
            CurvedCosseratRod({**nylon, "L": 0.1, "r_inner": ri, "r_outer": ro})
            for ri, ro in ((0.5e-3, 1.25e-3), (1.5e-3, 2.5e-3))
        ]
        robot = DynamicConcentricTubeRobot(tubes, dt=1e-3)
        robot.init_static()
        face = 2.5e-3 + 1e-3

        builder = DiagramBuilder()
        scene_graph = builder.AddSystem(SceneGraph())
        source = scene_graph.RegisterSource("obstacles")  # obstacles before the robot system
        box = scene_graph.RegisterAnchoredGeometry(
            source, GeometryInstance(RigidTransform([face + 0.01, 0, 0.05]), Box(0.02, 0.05, 0.2), "box")
        )
        scene_graph.AssignRole(source, box, ProximityProperties())
        ctr = builder.AddSystem(ConcentricTubeRobotSystem(robot, scene_graph))
        builder.Connect(ctr.GetOutputPort("geometry_pose"), scene_graph.get_source_pose_port(ctr.source_id))
        builder.Connect(scene_graph.get_query_output_port(), ctr.GetInputPort("geometry_query"))
        simulator = Simulator(builder.Build())
        context = ctr.GetMyMutableContextFromRoot(simulator.get_mutable_context())
        ctr.GetInputPort("tip_wrench").FixValue(context, [0.05, 0, 0, 0, 0, 0])  # free deflection 1.7 mm, swing 3.4 mm

        depth = []
        for n in range(20):
            simulator.AdvanceTo((n + 0.5) * robot.dt)
            p = ctr.GetOutputPort("positions").Eval(context).reshape(-1, 3)
            depth.append((p[:, 0] + 2.5e-3 - face).max())
        self.assertGreater(max(depth), 0)  # it touches the box
        self.assertLess(max(depth), 1e-4)  # and does not go through

    def test_mounted_on_plant_body(self):
        # base plate on a body welded to the world at p_WB, static tip force F: reaction F and (p_tip - p_WB) x F
        p_WB, F = np.array([0.1, 0.2, 0.3]), np.array([0.05, 0, 0])
        robot = straight_robot()
        robot.p_init = p_WB.reshape(3, 1)
        p, _, _ = robot.init_static(np.r_[F, 0, 0, 0])

        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=robot.dt)
        base = plant.AddRigidBody("base", SpatialInertia.SolidBoxWithMass(0.1, 0.02, 0.02, 0.02))
        plant.WeldFrames(plant.world_frame(), base.body_frame(), RigidTransform(p_WB))
        plant.Finalize()
        ctr = builder.AddSystem(ConcentricTubeRobotSystem(robot, scene_graph, plant, base))
        connect_to_plant(builder, plant, scene_graph, ctr)
        simulator = Simulator(builder.Build())
        context = ctr.GetMyMutableContextFromRoot(simulator.get_mutable_context())

        reaction = ctr.GetOutputPort("applied_spatial_force").Eval(context)[0]
        self.assertEqual(reaction.body_index, base.index())
        np.testing.assert_allclose(reaction.F_Bq_W.translational(), F, atol=1e-9)
        np.testing.assert_allclose(reaction.F_Bq_W.rotational(), np.cross(p[-1] - p_WB, F), atol=1e-9)

        ctr.GetInputPort("tip_wrench").FixValue(context, np.r_[F, 0, 0, 0])
        simulator.AdvanceTo(2.5 * robot.dt)  # base pose and velocity from the plant
        np.testing.assert_allclose(ctr.GetOutputPort("positions").Eval(context).reshape(-1, 3), p, atol=1e-9)

    def test_pushes_free_body_of_plant(self):
        # straight robot along z, a tip force pushes it in +x against a free box (5 g, no gravity) 1 mm away
        robot = straight_robot()
        robot.init_static()
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=robot.dt)
        plant.mutable_gravity_field().set_gravity_vector([0, 0, 0])
        base = plant.AddRigidBody("base", SpatialInertia.SolidBoxWithMass(0.1, 0.02, 0.02, 0.02))
        plant.WeldFrames(plant.world_frame(), base.body_frame())
        box = plant.AddRigidBody("box", SpatialInertia.SolidBoxWithMass(0.005, 0.02, 0.02, 0.02))
        geometry = plant.RegisterCollisionGeometry(box, RigidTransform(), Box(0.02, 0.02, 0.02), "box", CoulombFriction())
        plant.Finalize()
        ctr = builder.AddSystem(ConcentricTubeRobotSystem(robot, scene_graph, plant, base))
        connect_to_plant(builder, plant, scene_graph, ctr)
        simulator = Simulator(builder.Build())
        root = simulator.get_mutable_context()
        plant_context = plant.GetMyMutableContextFromRoot(root)
        plant.SetFreeBodyPose(plant_context, box, RigidTransform([2.5e-3 + 1e-3 + 0.01, 0, 0.09]))
        ctr.GetInputPort("tip_wrench").FixValue(ctr.GetMyMutableContextFromRoot(root), [0.05, 0, 0, 0, 0, 0])

        gaps = []
        for n in range(20):
            simulator.AdvanceTo((n + 0.5) * robot.dt)
            query = scene_graph.get_query_output_port().Eval(scene_graph.GetMyContextFromRoot(root))
            gaps.append(min(query.ComputeSignedDistancePairClosestPoints(g, geometry).distance for g in ctr._piece))
        self.assertLess(min(gaps), 0)  # contact
        self.assertGreater(min(gaps), -1e-4)  # without going through
        self.assertGreater(plant.EvalBodySpatialVelocityInWorld(plant_context, box).translational()[0], 0)  # pushed


    def test_insertion_with_contact(self):
        # straight robot pushed in +x against a fixed sphere next to the outer tube while the outer tube is retracted;
        # the base body has a collision box right behind the base plate, which must not count as contact
        robot = straight_robot()
        robot.init_static()
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=robot.dt)
        base = plant.AddRigidBody("base", SpatialInertia.SolidBoxWithMass(0.1, 0.02, 0.02, 0.02))
        plant.WeldFrames(plant.world_frame(), base.body_frame())
        plant.RegisterCollisionGeometry(base, RigidTransform([0, 0, -0.01]), Box(0.02, 0.02, 0.02), "base", CoulombFriction())
        sphere = plant.RegisterCollisionGeometry(
            plant.world_body(), RigidTransform([2.5e-3 + 5e-3 + 0.3e-3, 0, 0.04]), Sphere(5e-3), "sphere", CoulombFriction()
        )
        plant.Finalize()
        ctr = builder.AddSystem(ConcentricTubeRobotSystem(robot, scene_graph, plant, base))
        connect_to_plant(builder, plant, scene_graph, ctr)
        betas = builder.AddSystem(TrajectorySource(PiecewisePolynomial.FirstOrderHold([0, 0.03], [[0, 0], [0, 0.5]])))
        builder.Connect(betas.get_output_port(), ctr.GetInputPort("betas"))
        simulator = Simulator(builder.Build())
        root = simulator.get_mutable_context()
        ctr.GetInputPort("tip_wrench").FixValue(ctr.GetMyMutableContextFromRoot(root), [0.05, 0, 0, 0, 0, 0])

        gaps, forces = [], []
        for n in range(40):
            simulator.AdvanceTo((n + 0.5) * robot.dt)
            query = scene_graph.get_query_output_port().Eval(scene_graph.GetMyContextFromRoot(root))
            gaps.append(min(query.ComputeSignedDistancePairClosestPoints(g, sphere).distance for g in ctr._piece))
            forces.append(np.linalg.norm(robot.contact_force, axis=1).max())
            self.assertFalse(robot.contact_force[:2].any())  # no contact with the base body
        self.assertLess(min(gaps), 0)  # the robot touches the sphere
        self.assertGreater(min(gaps), -1e-4)  # without going through it
        self.assertLess(max(forces), 0.2)  # no force peaks when the grid changes (tip force 0.05 N)
        np.testing.assert_allclose(robot.betas, [0, 0.5])


if __name__ == "__main__":
    unittest.main()
