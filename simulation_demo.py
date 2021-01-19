#!/usr/bin/env python

##
#
# Simple example of using our kinova manipulation station in simulation
#
#
# The kinova manipulation station is a system with the following inputs
# and outputs:
#
#                              ---------------------------------
#                              |                               |
#                              |                               |
#                              |                               |
#                              |                               | --> measured_arm_position
#                              |                               | --> measured_arm_velocity
#   ee_target ---------------> |         KinovaStation         | --> measured_arm_torque
#   ee_target_type ----------> |                               |
#                              |                               |
#                              |                               | --> measured_ee_pose
#                              |                               | --> measured_ee_twist
#                              |                               | --> measured_ee_wrench
#   gripper_target ----------> |                               |
#   gripper_target_type -----> |                               |
#                              |                               | --> measured_gripper_position
#                              |                               | --> measured_gripper_velocity
#                              |                               |
#                              |                               | --> camera_rgb_image (TODO)
#                              |                               | --> camera_depth_image (TODO)
#                              |                               |
#                              |                               |
#                              ---------------------------------
#
# The end-effector target (ee_target) can be a pose, a wrench or a twist. 
# The gripper target (gripper_target) can be a position or a velocity. 
#
#
# See the "Parameters" section below for different ways of using and visualizing
# this system. 
##

from pydrake.all import *
import numpy as np
import matplotlib.pyplot as plt

from kinova_station import KinovaStation, EndEffectorTarget, GripperTarget

########################### Parameters #################################

# Make a plot of the inner workings of the station
show_station_diagram = True

# Make a plot of the diagram for this example, where only the inputs
# and outputs of the station are shown
show_toplevel_diagram = False

# Run a quick simulation
simulate = True

# If we're running a simulation, choose which sort of commands are
# sent to the arm and the gripper
ee_command_type = EndEffectorTarget.kTwist      # kPose, kTwist, or kWrench
gripper_command_type = GripperTarget.kPosition  # kPosition or kVelocity

########################################################################

# Set up the kinova station
station = KinovaStation(time_step=0.001)
station.SetupSinglePegScenario(gripper_type="2f_85")
station.AddCamera()
station.Finalize()

if show_station_diagram:
    # Show the station's system diagram
    plt.figure()
    plot_system_graphviz(station,max_depth=1)
    plt.show()

# Connect input ports to the kinova station
builder = DiagramBuilder()
builder.AddSystem(station)

# Set (constant) command to send to the system
if ee_command_type == EndEffectorTarget.kPose:
    pose_des = np.array([np.pi,0.0,0.0,
                            0.6,0.0,0.2])
    target_source = builder.AddSystem(ConstantVectorSource(pose_des))

elif ee_command_type == EndEffectorTarget.kTwist:
    twist_des = np.array([0,0,0.0,
                          0.0,0.0,0.0])
    target_source = builder.AddSystem(ConstantVectorSource(twist_des))

elif ee_command_type == EndEffectorTarget.kWrench:
    wrench_des = np.array([0,0,0.0,
                            0.1,0.0,0.0])
    target_source = builder.AddSystem(ConstantVectorSource(wrench_des))

else:
    raise RuntimeError("invalid end-effector target type")

# Send end-effector command and type
target_type_source = builder.AddSystem(ConstantValueSource(AbstractValue.Make(ee_command_type)))
builder.Connect(
        target_type_source.get_output_port(),
        station.GetInputPort("ee_target_type"))

builder.Connect(
        target_source.get_output_port(),
        station.GetInputPort("ee_target"))

target_source.set_name("ee_command_source")
target_type_source.set_name("ee_type_source")

# Set gripper command
if gripper_command_type == GripperTarget.kPosition:
    q_grip_des = np.array([0.06,0.06])  # Closed at [0,0], 
                                        # Hand-e open at [0.03,0.03], 
                                        # 2F-85 open at [0.06,0.06]
    gripper_target_source = builder.AddSystem(ConstantVectorSource(q_grip_des))

elif gripper_command_type == GripperTarget.kVelocity:
    v_grip_des = -np.array([0.01,0.01])
    gripper_target_source = builder.AddSystem(ConstantVectorSource(v_grip_des))

# Send gripper command and type
gripper_target_type_source = builder.AddSystem(ConstantValueSource(
                                         AbstractValue.Make(gripper_command_type)))
builder.Connect(
        gripper_target_type_source.get_output_port(),
        station.GetInputPort("gripper_target_type"))

builder.Connect(
        gripper_target_source.get_output_port(),
        station.GetInputPort("gripper_target"))

gripper_target_source.set_name("gripper_command_source")
gripper_target_type_source.set_name("gripper_type_source")

# Loggers force certain outputs to be computed
wrench_logger = LogOutput(station.GetOutputPort("measured_ee_wrench"),builder)
wrench_logger.set_name("wrench_logger")

pose_logger = LogOutput(station.GetOutputPort("measured_ee_pose"), builder)
pose_logger.set_name("pose_logger")

twist_logger = LogOutput(station.GetOutputPort("measured_ee_twist"), builder)
twist_logger.set_name("twist_logger")
    
# Build the system diagram
diagram = builder.Build()
diagram.set_name("toplevel_system_diagram")
diagram_context = diagram.CreateDefaultContext()

if show_toplevel_diagram:
    # Show the overall system diagram
    plt.figure()
    plot_system_graphviz(diagram,max_depth=1)
    plt.show()

if simulate:
    # Set default arm positions
    q0 = np.array([0.0, -np.pi/5, np.pi, -0.8*np.pi, np.pi, -0.1*np.pi, 0.5*np.pi])
    station.SetArmPositions(diagram, diagram_context, q0)

    # Set starting position for any objects in the scene
    station.SetManipulandStartPositions(diagram, diagram_context)

    # Set up simulation
    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(10.1)
    simulator.set_publish_every_time_step(False)

    # Run simulation
    simulator.Initialize()
    simulator.AdvanceTo(10.0)
