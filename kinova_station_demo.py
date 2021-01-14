#!/usr/bin/env python

##
#
# Simple example of using our kinova station and showing the 
# associated diagram.
#
##

from pydrake.all import *
from kinova_station import KinovaStation

from helpers import EndEffectorTargetType

import numpy as np
import matplotlib.pyplot as plt

#### Parameters ####

show_diagram = False
simulate = True

command_type = EndEffectorTargetType.kWrench  # pose, twist, or wrench

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
target_type_sys = builder.AddSystem(ConstantValueSource(AbstractValue.Make(command_type)))
builder.Connect(
        target_type_sys.get_output_port(),
        station.GetInputPort("ee_target_type"))

# Set (constant) command to send to the system
if command_type == EndEffectorTargetType.kPose:
    pose_des = np.array([np.pi,0.0,0.0,
                            0.6,0.0,0.2])
    target_source = builder.AddSystem(ConstantVectorSource(pose_des))

elif command_type == EndEffectorTargetType.kTwist:
    twist_des = np.array([0,0,0,
                          0.0,0.0,0.0])
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
q_grip_des = np.array([0.03,0.03])  # open at [0,0], closed at [0.03,0.03]
v_grip_des = np.array([0,0])

target_gripper_position = builder.AddSystem(ConstantVectorSource(q_grip_des))
target_gripper_velocity = builder.AddSystem(ConstantVectorSource(v_grip_des))

builder.Connect(
        target_gripper_position.get_output_port(0),
        station.GetInputPort("target_gripper_position"))
builder.Connect(
        target_gripper_velocity.get_output_port(0),
        station.GetInputPort("target_gripper_velocity"))

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

    simulator.Initialize()
    simulator.AdvanceTo(10)

