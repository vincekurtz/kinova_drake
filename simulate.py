#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
from pydrake.all import *
from reduced_order_model import ReducedOrderModelPlant
from controller import Gen3Controller

# Find the (local) description file relative to drake
robot_description_path = "./model/urdf/GEN3_URDF_V12.urdf"
drake_path = getDrakePath()
robot_description_file = "drake/" + os.path.relpath(robot_description_path, start=drake_path)

# Load the kuka model from a urdf file
robot_urdf = FindResourceOrThrow(robot_description_file)
builder = DiagramBuilder()
scene_graph = builder.AddSystem(SceneGraph())
dt = 2e-3
plant = builder.AddSystem(MultibodyPlant(time_step=dt))
plant.RegisterAsSourceForSceneGraph(scene_graph)
gen3 = Parser(plant=plant).AddModelFromFile(robot_urdf,"gen3")

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

ee_shape = Sphere(0.03)
#ee_shape = Mesh("/home/vjkurtz/projects/kinova_drake/model/meshes/base_link.obj")
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
builder.Connect(                              # First input to controller is [q;qd] for the arm
        plant.get_state_output_port(gen3),
        controller.get_input_port(0))
builder.Connect(                              # First controller output is torques to the arm
        controller.get_output_port(0),
        plant.get_actuation_input_port(gen3))

builder.Connect(rom_ctrl.get_output_port(), rom.GetInputPort("u"))
builder.Connect(rom.GetOutputPort("x"), rom_ctrl.get_input_port_estimated_state())

builder.Connect(rom.GetOutputPort("x"), controller.GetInputPort("rom_state"))
builder.Connect(rom_ctrl.get_output_port(), controller.GetInputPort("rom_input"))

# Set desired RoM state  
# TODO: replace with some sort of finite state machine
p_nom = np.array([-np.pi/2,-np.pi/4,np.pi/4,0.5,0.5,0.7])
pd_nom = np.array([0,0,0,0,0,0])
rom_target = builder.AddSystem(ConstantVectorSource(np.hstack([p_nom,pd_nom])))
builder.Connect(rom_target.get_output_port(),rom_ctrl.get_input_port_desired_state())

# Set up the Visualizer
DrakeVisualizer().AddToBuilder(builder=builder, scene_graph=scene_graph)
ConnectContactResultsToDrakeVisualizer(builder, plant)

# Add loggers
rom_logger = LogOutput(rom.GetOutputPort("x"),builder)
rom_logger.set_name("rom_logger")

plant_logger = LogOutput(controller.GetOutputPort("end_effector"),builder)
plant_logger.set_name("plant_logger")

V_logger = LogOutput(controller.GetOutputPort("simulation_function"),builder)
V_logger.set_name("V_logger")
V_logger.set_publish_period(0.01)

# Compile the diagram: no adding control blocks from here on out
diagram = builder.Build()
diagram.set_name("diagram")
diagram_context = diagram.CreateDefaultContext()

# Visualize the diagram
#plt.figure()
#plot_system_graphviz(diagram,max_depth=2)
#plt.show()

# Simulator setup
simulator = Simulator(diagram, diagram_context)
simulator.set_target_realtime_rate(1.0)
simulator.set_publish_every_time_step(False)

# Set initial states
plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
plant.SetPositions(plant_context, gen3, np.array([0.0,0,np.pi/2,-np.pi/2,0.0,0,0]))   # Manipulator arm
plant.SetVelocities(plant_context, gen3, np.zeros(7))

rom_p0 = np.array([0.0,0.5,0.7])
rom_rpy0 = np.array([-np.pi/2,-np.pi/2,np.pi/2])
rom_context = diagram.GetMutableSubsystemContext(rom, diagram_context)
rom_context.SetContinuousState(np.hstack([rom_rpy0,rom_p0,np.zeros(6,)]))

# Run the simulation
simulator.Initialize()
simulator.AdvanceTo(5.00)

# Make some plots
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
plt.plot(V_logger.sample_times(), V_logger.data().T, linewidth='2',label="Simulation Function")
p_sim = plant_logger.data()[:3,:].T
p_des_sim = rom_logger.data()[:3,:].T
err = np.linalg.norm((p_sim - p_des_sim),axis=1)**2
plt.plot(t,err, linewidth='2',label="Output Error")

# Error bound is initial simulation function value, scaled by minimum eigenvalue of Kp
err_bound = V_logger.data()[0,0]/np.min(np.linalg.eigvals(controller.Kp))
plt.hlines(err_bound,t[0],t[-1],label="Error Bound",color="grey",linewidth=2,linestyle="--")

plt.xlabel("time (s)")
plt.legend()


plt.show()
