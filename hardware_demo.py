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

# Make a plot of the inner workings of the station
show_station_diagram = False

# Make a plot of the diagram for this example, where only the inputs
# and outputs of the station are shown
show_toplevel_diagram = False

# Choose which sort of commands are
# sent to the arm and the gripper
ee_command_type = EndEffectorTarget.kTwist      # kPose, kTwist, or kWrench
gripper_command_type = GripperTarget.kPosition  # kPosition or kVelocity

########################################################################

with KinovaStationHardwareInterface() as station:
    station.go_home("Home")
    station.send_ee_pose_example()

