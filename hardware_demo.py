#!/usr/bin/env python

##
#
# Simple example of using our kinova manipulation station in real life
#
#
# The kinova manipulation station is a system with the following inputs
# and outputs:
#
#                              ------------------------------------
#                              |                                  |
#                              |                                  |
#                              |                                  |
#                              |                                  | --> measured_arm_position
#                              |                                  | --> measured_arm_velocity
#   ee_target ---------------> |  KinovaStationHardwareInterface  | --> measured_arm_torque
#   ee_target_type ----------> |                                  |
#                              |                                  |
#                              |                                  | --> measured_ee_pose
#                              |                                  | --> measured_ee_twist
#                              |                                  | --> measured_ee_wrench
#   gripper_target ----------> |                                  |
#   gripper_target_type -----> |                                  |
#                              |                                  | --> measured_gripper_position
#                              |                                  | --> measured_gripper_velocity
#                              |                                  |
#                              |                                  | --> camera_rgb_image (TODO)
#                              |                                  | --> camera_depth_image (TODO)
#                              |                                  |
#                              |                                  |
#                              ------------------------------------
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

from kinova_station import KinovaStationHardwareInterface, EndEffectorTarget, GripperTarget

########################### Parameters #################################

# Make a plot of the system diagram for this example
show_toplevel_diagram = False

# Run the example
run = True

# Choose which sort of commands are
# sent to the arm and the gripper
ee_command_type = EndEffectorTarget.kTwist      # kPose, kTwist, or kWrench
gripper_command_type = GripperTarget.kPosition  # kPosition or kVelocity

########################################################################

# Note that unlike the simulation station, the hardware station needs
# to be used within a 'with' block. This is to allow for cleaner error
# handling, since the connection with the hardware needs to be closed 
# properly even if there is an error (e.g. KeyboardInterrupt) during
# execution.
with KinovaStationHardwareInterface() as station:
   
    # First thing: send to home position
    station.go_home()

    # Set up the diagram builder
    builder = DiagramBuilder()
    builder.AddSystem(station)

    # Connect simple controllers to inputs

    # Connect loggers to outputs
    test_logger = LogOutput(station.GetOutputPort("test_output_port"), builder)
    test_logger.set_name("test_logger")

    # Build the system diagram
    diagram = builder.Build()
    diagram.set_name("toplevel_system_diagram")
    diagram_context = diagram.CreateDefaultContext()

    if show_toplevel_diagram:
        # Make a plot of the system diagram
        plt.figure()
        plot_system_graphviz(diagram, max_depth=1)
        plt.show()

    # Run the example
    if run:
        # We use a simulator instance to run the example, but no actual simulation 
        # is being done: it's all on the hardware. 
        simulator = Simulator(diagram, diagram_context)
        simulator.set_target_realtime_rate(1.0)
        simulator.set_publish_every_time_step(True)  # not sure if this is correct

        # We'll use a super simple integration scheme (since we only update a dummy state)
        # and set the maximum timestep to correspond to roughly 40Hz 
        integration_scheme = "explicit_euler"
        time_step = 0.025
        ResetIntegratorFromFlags(simulator, integration_scheme, time_step)

        simulator.Initialize()
        simulator.AdvanceTo(2.0)  # seconds

        # Print rate data
        print("")
        print("Target realtime rate: %s" % simulator.get_target_realtime_rate())
        print("Actual realtime rate: %s" % simulator.get_actual_realtime_rate())

