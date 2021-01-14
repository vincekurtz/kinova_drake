#!/usr/bin/env python

##
#
# Simple example of using our kinova station and showing the 
# associated diagram.
#
##

from pydrake.all import *
from kinova_station import KinovaStation

import numpy as np
import matplotlib.pyplot as plt

#### Parameters ####

show_diagram = False
simulate = True

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

# Desired end-effector pose
rpy_xyz_des = np.array([np.pi,0.0,0.0,
                        0.6,0.0,0.2])
target_ee_pose = builder.AddSystem(ConstantVectorSource(rpy_xyz_des))
builder.Connect(
        target_ee_pose.get_output_port(0),
        station.GetInputPort("target_ee_pose"))

# Desired end-effector twist
# TODO

# Desired end-effector wrench
# TODO

# Send desired end-effector position and velocity
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

# Loggers
wrench_logger = LogOutput(station.GetOutputPort("measured_ee_wrench"),builder)
wrench_logger.set_name("wrench_logger")


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


