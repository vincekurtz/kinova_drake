#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
from pydrake.all import *
from reduced_order_model import ReducedOrderModelPlant
from controller import Gen3Controller

############## Setup Parameters #################

sim_time = 5.0
dt = 2e-3
target_realtime_rate = 1.0

# Initial joint angles
q0 = np.array([0.0,0,np.pi/2,-np.pi/2,0.0,-np.pi/2,0])

# initial end-effector pose
x0 = np.array([np.pi,  
               0,
               np.pi/2,
               0.0,
               0.3,
               0.55])

# Target end-effector pose
x_target = np.array([np.pi,  
                     0.0,
                     np.pi/2,
                     0.0,
                     0.3,
                     0.1])

include_gripper = True

show_diagram = False
make_plots = False

#################################################

# Find the (local) description file relative to drake
robot_description_path = "./models/gen3_7dof/urdf/GEN3_URDF_V12.urdf"
drake_path = getDrakePath()
robot_description_file = "drake/" + os.path.relpath(robot_description_path, start=drake_path)

# Load the robot arm model from a urdf file
robot_urdf = FindResourceOrThrow(robot_description_file)
builder = DiagramBuilder()
scene_graph = builder.AddSystem(SceneGraph())
plant = builder.AddSystem(MultibodyPlant(time_step=dt))
plant.RegisterAsSourceForSceneGraph(scene_graph)
gen3 = Parser(plant=plant).AddModelFromFile(robot_urdf,"gen3")

# Load the gripper model from a urdf file
if include_gripper:
    gripper_file = "drake/" + os.path.relpath("./models/hande_gripper/urdf/robotiq_hande_static.urdf", start=drake_path)
    gripper_urdf = FindResourceOrThrow(gripper_file)
    gripper = Parser(plant=plant).AddModelFromFile(gripper_urdf,"gripper")

    # Fix the gripper to the manipulator arm
    X_EE = RigidTransform()
    plant.WeldFrames(plant.GetFrameByName("end_effector_link",gen3), plant.GetFrameByName("hande_base_link", gripper), X_EE)

# Fix the base of the manipulator to the world
plant.WeldFrames(plant.world_frame(),plant.GetFrameByName("base_link",gen3))

# Add a flat ground with friction
X_BG = RigidTransform()
surface_friction = CoulombFriction(
        static_friction = 0.7,
        dynamic_friction = 0.1)
plant.RegisterCollisionGeometry(
        plant.world_body(),      # the body for which this object is registered
        X_BG,                    # The fixed pose of the geometry frame G in the body frame B
        HalfSpace(),             # Defines the geometry of the object
        "ground_collision",      # A name
        surface_friction)        # Coulomb friction coefficients
plant.RegisterVisualGeometry(
        plant.world_body(),
        X_BG,
        HalfSpace(),
        "ground_visual",
        np.array([0.5,0.5,0.5,0.0]))    # Color set to be completely transparent

#plant.set_contact_model(ContactModel.kHydroelasticWithFallback)

plant.Finalize()
assert plant.geometry_source_is_registered()

# Add end-effector visualization
ee_source = scene_graph.RegisterSource("ee")
ee_frame = GeometryFrame("ee")
scene_graph.RegisterFrame(ee_source, ee_frame)

ee_shape = Mesh(os.path.abspath("./models/hande_gripper/meshes/hand-e_with_fingers.obj"),scale=1e-3)
ee_color = np.array([0.1,0.1,0.1,0.4])
X_ee = RigidTransform()

ee_geometry = GeometryInstance(X_ee, ee_shape, "ee")
ee_geometry.set_illustration_properties(MakePhongIllustrationProperties(ee_color))
scene_graph.RegisterGeometry(ee_source, ee_frame.id(), ee_geometry)

# Set up the Scene Graph
builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port())
builder.Connect(
        plant.get_geometry_poses_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id()))

# Connect controllers
rom_dof = 6   # degrees of freedom in the reduced-order model
rom = builder.AddSystem(ReducedOrderModelPlant(rom_dof, ee_frame.id()))
rom.set_name("RoM")
rom_ctrl = builder.AddSystem(PidController(kp=2*np.ones(rom_dof), 
                                           ki=np.zeros(rom_dof),
                                           kd=2*np.ones(rom_dof)))
rom_ctrl.set_name("RoM_controller")

builder.Connect(
        rom.GetOutputPort("ee_geometry"),
        scene_graph.get_source_pose_port(ee_source))

ctrl = Gen3Controller(plant,dt)
controller = builder.AddSystem(ctrl)
builder.Connect(
        plant.get_state_output_port(),
        controller.GetInputPort("arm_state"))
builder.Connect(
        controller.GetOutputPort("arm_torques"),
        plant.get_actuation_input_port(gen3))

#if include_gripper:
#    builder.Connect(
#            controller.GetOutputPort("gripper_forces"),
#            plant.get_actuation_input_port(gripper))

builder.Connect(rom_ctrl.get_output_port(), rom.GetInputPort("u"))
builder.Connect(rom.GetOutputPort("x"), rom_ctrl.get_input_port_estimated_state())

builder.Connect(rom.GetOutputPort("x"), controller.GetInputPort("rom_state"))
builder.Connect(rom_ctrl.get_output_port(), controller.GetInputPort("rom_input"))

# Set desired RoM state  
# TODO: replace with some sort of finite state machine
rom_target = builder.AddSystem(ConstantVectorSource(np.hstack([x_target,np.zeros(6)])))
builder.Connect(rom_target.get_output_port(),rom_ctrl.get_input_port_desired_state())

# Set up the Visualizer
visualizer_params = DrakeVisualizerParams(role=Role.kIllustration)  # kProximity for collision geometry,
                                                                    # kIllustration for visual geometry
DrakeVisualizer().AddToBuilder(builder=builder, scene_graph=scene_graph, params=visualizer_params)
ConnectContactResultsToDrakeVisualizer(builder, plant)

# Add loggers
rom_logger = LogOutput(rom.GetOutputPort("x"),builder)
rom_logger.set_name("rom_logger")

plant_logger = LogOutput(controller.GetOutputPort("end_effector"),builder)
plant_logger.set_name("plant_logger")

V_logger = LogOutput(controller.GetOutputPort("simulation_function"),builder)
V_logger.set_name("V_logger")

err_logger = LogOutput(controller.GetOutputPort("error"),builder)
err_logger.set_name("error_logger")

# Compile the diagram: no adding control blocks from here on out
diagram = builder.Build()
diagram.set_name("diagram")
diagram_context = diagram.CreateDefaultContext()

# Visualize the diagram
if show_diagram:
    plt.figure()
    plot_system_graphviz(diagram,max_depth=2)
    plt.show()

# Simulator setup
simulator = Simulator(diagram, diagram_context)
simulator.set_target_realtime_rate(target_realtime_rate)
simulator.set_publish_every_time_step(False)

# Set initial states
plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
plant.SetPositions(plant_context, gen3, q0)   # Manipulator arm
plant.SetVelocities(plant_context, gen3, np.zeros(7))

rom_context = diagram.GetMutableSubsystemContext(rom, diagram_context)
rom_context.SetContinuousState(np.hstack([x0,np.zeros(6,)]))

# Run the simulation
simulator.Initialize()
simulator.AdvanceTo(sim_time)

# Make some plots
if make_plots:
    t = rom_logger.sample_times()

    plt.figure()  # End effector rpy and angular velocity comparison
    plt.subplot(2,1,1)
    plt.plot(t, rom_logger.data()[:3,:].T,linewidth='2',linestyle='--')
    plt.gca().set_prop_cycle(None)  # reset the color cycle
    plt.plot(t, plant_logger.data()[:3,:].T,linewidth='2')
    #plt.legend(['RoM - x','RoM - y','RoM - z','Plant - x','Plant - y','Plant - z'])
    plt.xlabel("time (s)")
    plt.ylabel("End Effector RPY")

    plt.subplot(2,1,2)
    plt.plot(t, rom_logger.data()[6:9,:].T,linewidth='2',linestyle='--')
    plt.gca().set_prop_cycle(None)  # reset the color cycle
    plt.plot(t, plant_logger.data()[6:9,:].T,linewidth='2')
    #plt.legend(['RoM - x','RoM - y','RoM - z','Plant - x','Plant - y','Plant - z'])
    plt.xlabel("time (s)")
    plt.ylabel("End Effector Angular Velocity")

    plt.figure()  # End effector position and velocity comparison
    plt.subplot(2,1,1)
    plt.plot(t, rom_logger.data()[3:6,:].T,linewidth='2',linestyle='--')
    plt.gca().set_prop_cycle(None)  # reset the color cycle
    plt.plot(t, plant_logger.data()[3:6,:].T,linewidth='2')
    #plt.legend(['RoM - x','RoM - y','RoM - z','Plant - x','Plant - y','Plant - z'])
    plt.xlabel("time (s)")
    plt.ylabel("End Effector Position")

    plt.subplot(2,1,2)
    plt.plot(t, rom_logger.data()[9:,:].T,linewidth='2',linestyle='--')
    plt.gca().set_prop_cycle(None)  # reset the color cycle
    plt.plot(t, plant_logger.data()[9:,:].T,linewidth='2')
    #plt.legend(['RoM - x','RoM - y','RoM - z','Plant - x','Plant - y','Plant - z'])
    plt.xlabel("time (s)")
    plt.ylabel("End Effector Velocity")

    plt.figure()  # Simulation function and error comparison
    plt.plot(t, V_logger.data().T, linewidth='2',label="Simulation Function")
    plt.plot(t, err_logger.data().T, linewidth='2',label="Output Error")

    # Error bound is initial simulation function value, scaled by minimum eigenvalue of Kp
    #err_bound = V_logger.data()[0,0]/np.min(np.linalg.eigvals(controller.Kp))
    #plt.hlines(err_bound,t[0],t[-1],label="Error Bound",color="grey",linewidth=2,linestyle="--")

    plt.xlabel("time (s)")
    plt.legend()

    plt.show()
