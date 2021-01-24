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
gripper_command_type = GripperTarget.kVelocity  # kPosition or kVelocity

########################################################################

# Note that unlike the simulation station, the hardware station needs
# to be used within a 'with' block. This is to allow for cleaner error
# handling, since the connection with the hardware needs to be closed 
# properly even if there is an error (e.g. KeyboardInterrupt) during
# execution.
with KinovaStationHardwareInterface() as station:
   
    # Set up the diagram builder
    builder = DiagramBuilder()
    builder.AddSystem(station)

    # Connect end-effector target source
    if ee_command_type == EndEffectorTarget.kPose:
        pose_des = np.array([0.6*np.pi,0.0,np.pi/2, 0.58,0.0,0.2])
        target_source = builder.AddSystem(ConstantVectorSource(pose_des))
    elif ee_command_type == EndEffectorTarget.kTwist:
        twist_des = np.array([0.3,0,0.0,0.0,0.0,0.05])
        target_source = builder.AddSystem(ConstantVectorSource(twist_des))
    elif ee_command_type == EndEffectorTarget.kWrench:
        wrench_des = np.array([10.0, 0.0, 0, 0.0 , 0.0, 1.0])
        target_source = builder.AddSystem(ConstantVectorSource(wrench_des))
    target_source.set_name("ee_command_source")
    builder.Connect(
            target_source.get_output_port(),
            station.GetInputPort("ee_target"))

    # Connect end-effector type source
    target_type_source = builder.AddSystem(ConstantValueSource(AbstractValue.Make(ee_command_type)))
    target_type_source.set_name("ee_type_source")
    builder.Connect(
            target_type_source.get_output_port(),
            station.GetInputPort("ee_target_type"))

    # Connect gripper target source
    if gripper_command_type == GripperTarget.kPosition:
        q_grip_des = np.array([0.0])
        gripper_target_source = builder.AddSystem(ConstantVectorSource(q_grip_des))
    elif gripper_command_type == GripperTarget.kVelocity:
        v_grip_des = np.array([-0.1])
        gripper_target_source = builder.AddSystem(ConstantVectorSource(v_grip_des))
    gripper_target_source.set_name("gripper_command_source")
    builder.Connect(
            gripper_target_source.get_output_port(),
            station.GetInputPort("gripper_target"))

    # Connect gripper type source
    gripper_target_type_source = builder.AddSystem(ConstantValueSource(
                                             AbstractValue.Make(gripper_command_type)))
    gripper_target_type_source.set_name("gripper_type_source")
    builder.Connect(
            gripper_target_type_source.get_output_port(),
            station.GetInputPort("gripper_target_type"))

    # Connect loggers to outputs
    q_logger = LogOutput(station.GetOutputPort("measured_arm_position"), builder)
    q_logger.set_name("arm_position_logger")
    qd_logger = LogOutput(station.GetOutputPort("measured_arm_velocity"), builder)
    qd_logger.set_name("arm_velocity_logger")
    tau_logger = LogOutput(station.GetOutputPort("measured_arm_torque"), builder)
    tau_logger.set_name("arm_torque_logger")

    pose_logger = LogOutput(station.GetOutputPort("measured_ee_pose"), builder)
    pose_logger.set_name("pose_logger")
    twist_logger = LogOutput(station.GetOutputPort("measured_ee_twist"), builder)
    twist_logger.set_name("twist_logger")
    wrench_logger = LogOutput(station.GetOutputPort("measured_ee_wrench"), builder)
    wrench_logger.set_name("wrench_logger")

    #gp_logger = LogOutput(station.GetOutputPort("measured_gripper_position"), builder)
    #gp_logger.set_name("gripper_position_logger")
    #gv_logger = LogOutput(station.GetOutputPort("measured_gripper_velocity"), builder)
    #gv_logger.set_name("gripper_velocity_logger")

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
        # First thing: send to home position
        station.go_home()

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
        print("Target control frequency: %s Hz" % (1/time_step))
        print("Actual control frequency: %s Hz" % (1/time_step * simulator.get_actual_realtime_rate()))

