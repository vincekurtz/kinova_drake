#!/usr/bin/env python

##
#
# Simple example of using our kinova station and showing the 
# associated diagram.
#
##

from pydrake.all import *
from kinova_station import KinovaStation

from helpers import EndEffectorTargetType, GripperTargetType

import numpy as np
import matplotlib.pyplot as plt

#### Parameters ####

show_diagram = False
simulate = True

command_type = EndEffectorTargetType.kTwist  # kPose, kTwist, or kWrench

gripper_command_type = GripperTargetType.kVelocity  # kPosition or kVelocity

####################

# Set up the kinova station
station = KinovaStation(time_step=0.001)
station.AddArmWithHandeGripper()
station.AddGround()
station.ConnectToDrakeVisualizer()
station.Finalize()

if show_diagram:
    # Show the station's system diagram
    plt.figure()
    plot_system_graphviz(station,max_depth=1)
    plt.show()

# Connect input ports to the kinova station
builder = DiagramBuilder()
builder.AddSystem(station)

# Send the command type to the system
target_type_source = builder.AddSystem(ConstantValueSource(AbstractValue.Make(command_type)))
builder.Connect(
        target_type_source.get_output_port(),
        station.GetInputPort("ee_target_type"))

# Set (constant) command to send to the system
if command_type == EndEffectorTargetType.kPose:
    pose_des = np.array([np.pi,0.0,0.0,
                            0.6,0.0,0.2])
    target_source = builder.AddSystem(ConstantVectorSource(pose_des))

elif command_type == EndEffectorTargetType.kTwist:
    twist_des = np.array([0,0,0.2,
                          0.0,0.0,-0.05])
    target_source = builder.AddSystem(ConstantVectorSource(twist_des))

elif command_type == EndEffectorTargetType.kWrench:
    wrench_des = np.array([0,0,0.001,
                            0,0,0.0])
    target_source = builder.AddSystem(ConstantVectorSource(wrench_des))

else:
    raise RuntimeError("invalid end-effector target type")

builder.Connect(
        target_source.get_output_port(),
        station.GetInputPort("ee_target"))

# Send gripper command
if gripper_command_type == GripperTargetType.kPosition:
    q_grip_des = np.array([0.03,0.03])  # open at [0,0], closed at [0.03,0.03]
    gripper_target_source = builder.AddSystem(ConstantVectorSource(q_grip_des))

elif gripper_command_type == GripperTargetType.kVelocity:
    v_grip_des = np.array([0.01,0.01])
    gripper_target_source = builder.AddSystem(ConstantVectorSource(v_grip_des))

builder.Connect(
        gripper_target_source.get_output_port(),
        station.GetInputPort("gripper_target"))

gripper_target_type_source = builder.AddSystem(ConstantValueSource(
                                         AbstractValue.Make(gripper_command_type)))
builder.Connect(
        gripper_target_type_source.get_output_port(),
        station.GetInputPort("gripper_target_type"))

# Loggers force certain outputs to be computed
wrench_logger = LogOutput(station.GetOutputPort("measured_ee_wrench"),builder)
wrench_logger.set_name("wrench_logger")

pose_logger = LogOutput(station.GetOutputPort("measured_ee_pose"), builder)
pose_logger.set_name("pose_logger")

twist_logger = LogOutput(station.GetOutputPort("measured_ee_twist"), builder)
twist_logger.set_name("twist_logger")

if simulate:
    # Build the system diagram
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()

    # Set up simulation
    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(False)

    # Run simulation
    simulator.Initialize()
    simulator.AdvanceTo(10)

