#!/usr/bin/env python

##
#
# Toward Bayesian parameter estimation for objects in the environment
#
##

import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import *

from kinova_station import *
from controllers.peg_pickup_controller import PegPickupController
from observers.bayes_observer import BayesObserver

# Set up the station
time_step = 0.002
station = KinovaStation(time_step=time_step)
station.AddGround()
station.AddArmWithHandeGripper()

# Weld peg to the end-effector
peg_urdf = "./models/manipulands/peg.sdf"
peg = Parser(plant=station.plant).AddModelFromFile(peg_urdf,"peg")

X_peg = RigidTransform()
X_peg.set_translation([0,0,0.13])
X_peg.set_rotation(RotationMatrix(RollPitchYaw([0,0,np.pi/2])))
station.plant.WeldFrames(station.plant.GetFrameByName("end_effector_link",station.arm),
                         station.plant.GetFrameByName("base_link", peg), X_peg)

#station.SetupSinglePegScenario(gripper_type="hande")
station.ConnectToDrakeVisualizer()
station.Finalize()

# Set up the system diagram
builder = DiagramBuilder()
builder.AddSystem(station)

# Add peg pickup system (goes to peg, closes gripper, lifts peg up, waves around)
controller = builder.AddSystem(PegPickupController())
controller.set_name("controller")

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


# Add bayesian inference system (records wrenches, estimates inertial parameters)
observer = builder.AddSystem(BayesObserver(time_step=time_step))
observer.set_name("bayesian_observer")

builder.Connect(
        station.GetOutputPort("measured_arm_position"),
        observer.GetInputPort("joint_positions"))
builder.Connect(
        station.GetOutputPort("measured_arm_velocity"),
        observer.GetInputPort("joint_velocities"))
builder.Connect(
        station.GetOutputPort("measured_arm_torque"),
        observer.GetInputPort("joint_torques"))

estimation_logger = LogOutput(observer.GetOutputPort("manipuland_parameter_estimate"), builder)
estimation_logger.set_name("estimation_logger")
covariance_logger = LogOutput(observer.GetOutputPort("manipuland_parameter_covariance"), builder)
covariance_logger.set_name("covariance_logger")

# Set up system diagram
diagram = builder.Build()
diagram.set_name("diagram")
diagram_context = diagram.CreateDefaultContext()

# DEBUG: show system diagram
#plt.figure()
#plot_system_graphviz(diagram, max_depth=1)
#plt.show()

# Set initial positions
q0 = np.array([0.0, -0.2, 1, -0.8, 1, -0.1, 0.5])*np.pi
station.SetArmPositions(diagram, diagram_context, q0)
station.SetManipulandStartPositions(diagram, diagram_context)

# Set up simulation
simulator = Simulator(diagram, diagram_context)
simulator.set_target_realtime_rate(10.0)
simulator.set_publish_every_time_step(False)

# Run simulation
try:
    simulator.Initialize()
    simulator.AdvanceTo(10)
except KeyboardInterrupt:
    pass

# Make plot of estimate
t = estimation_logger.sample_times()
m_hat = estimation_logger.data().flatten()
m_var = covariance_logger.data().flatten()

plt.plot(t,m_hat, label="Estimate")
plt.fill_between(t, m_hat-m_var, m_hat+m_var, label="Variance", color="green",alpha=0.5)
plt.gca().axhline(0.028, color="grey", linestyle="--", label="Ground Truth")

plt.xlabel("Time (s)")
plt.ylabel("Estimated Mass (kg)")

#plt.xlim(left=10)
plt.ylim(bottom=0, top=0.1)
plt.legend()

plt.show()
