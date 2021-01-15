#!/usr/bin/env python

##
#
# Toward Bayesian parameter estimation for objects in the environment
#
##

import numpy as np
from pydrake.all import *
from kinova_station import *
from controllers.peg_pickup_controller import PegPickupController

# Set up the station
station = KinovaStation(time_step=0.001)
station.SetupSinglePegScenario()
station.Finalize()

# Set up the system diagram
builder = DiagramBuilder()
builder.AddSystem(station)

# Add peg pickup system (goes to peg, closes gripper, lifts peg up, waves around)
controller = builder.AddSystem(PegPickupController())

builder.Connect(                                     # Send state information
        station.GetOutputPort("measured_ee_pose"),   # to the controller
        controller.GetInputPort("ee_pose"))
builder.Connect(
        station.GetOutputPort("measured_ee_twist"),
        controller.GetInputPort("ee_twist"))
builder.Connect(
        station.GetOutputPort("measured_gripper_position"),
        controller.GetInputPort("gripper_position"))
builder.Connect(
        station.GetOutputPort("measured_gripper_velocity"),
        controller.GetInputPort("gripper_velocity"))

builder.Connect(                                  # Send commands to the station
        controller.GetOutputPort("ee_command"),
        station.GetInputPort("ee_target"))
builder.Connect(
        controller.GetOutputPort("ee_command_type"),
        station.GetInputPort("ee_target_type"))
builder.Connect(
        controller.GetOutputPort("gripper_command"),
        station.GetInputPort("gripper_target"))
builder.Connect(
        controller.GetOutputPort("gripper_command_type"),
        station.GetInputPort("gripper_target_type"))

# Add bayesian inference system (records wrenches, estimates inertial parameters)
wrench_logger = LogOutput(station.GetOutputPort("measured_ee_wrench"), builder)

# Set up system diagram
diagram = builder.Build()
diagram.set_name("toplevel_system_diagram")
diagram_context = diagram.CreateDefaultContext()

# Set initial positions
q0 = np.array([0.0, -0.2, 1, -0.8, 1, -0.1, 0.5])*np.pi
station.SetArmPositions(diagram, diagram_context, q0)
station.SetManipulandStartPositions(diagram, diagram_context)

# Set up simulation
simulator = Simulator(diagram, diagram_context)
simulator.set_target_realtime_rate(1.0)
simulator.set_publish_every_time_step(False)

# Run simulation
simulator.Initialize()
simulator.AdvanceTo(20)
