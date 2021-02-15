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
from controllers import Command, CommandSequence, CommandSequenceController
from observers import BayesObserver

# Set up the station
time_step = 0.002
station = KinovaStation(time_step=time_step)
station.AddGround()
station.AddArmWithHandeGripper()

# Weld peg to the end-effector
peg_urdf = "./models/manipulands/peg.sdf"
peg = Parser(plant=station.plant).AddModelFromFile(peg_urdf,"peg")

X_peg = RigidTransform()
X_peg.set_translation([0.0,0.05,0.13])
X_peg.set_rotation(RotationMatrix(RollPitchYaw([0,0,np.pi/2])))
station.plant.WeldFrames(station.plant.GetFrameByName("end_effector_link",station.arm),
                         station.plant.GetFrameByName("base_link", peg), X_peg)

#station.SetupSinglePegScenario(gripper_type="hande")
station.ConnectToDrakeVisualizer()
station.Finalize()

# Set up the system diagram
builder = DiagramBuilder()
station = builder.AddSystem(station)

# Add controller which moves the peg around
cs = CommandSequence([
        Command(target_pose = np.array([-np.pi/2,0,0, 0.5,-0.12,0.7]),
                duration = 2,
                gripper_closed = False),
        Command(target_pose =  np.array([-np.pi,0,0, 0.5,-0.12,0.5]),
                duration = 2,
                gripper_closed = False),
        Command(target_pose = np.array([np.pi/2,0,np.pi/2,0.5,0.12,0.7]),
                duration = 2,
                gripper_closed = False),
        Command(target_pose = np.array([-np.pi/2,0,0, 0.5,-0.12,0.5]),
                duration = 2,
                gripper_closed = False)])

controller = builder.AddSystem(CommandSequenceController(cs))
controller.set_name("controller")

controller.ConnectToStation(builder, station)

# Add bayesian inference system (estimates inertial parameters)
observer = builder.AddSystem(BayesObserver(time_step=time_step, method="standard", estimator="bayes"))
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
    simulator.AdvanceTo(5)
except KeyboardInterrupt:
    pass

# Make plot of estimate
t = estimation_logger.sample_times()
theta_hat = estimation_logger.data()
theta_var = covariance_logger.data()
theta_std = np.sqrt(theta_var)

#plt.plot(t, theta_hat[0,:], label="px ")
#plt.plot(t, theta_hat[1,:], label="py ")
#plt.plot(t, theta_hat[2,:], label="pz ")
plt.plot(t, theta_hat[0,:], label="m")

# Show the ground truth values
true_com = X_peg.translation()
true_m = station.plant.GetBodyByName("base_link", peg).default_mass()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']      # Use the default color sequence
#plt.gca().axhline(true_com[0], linestyle="--", color=colors[0]) # to match estimates above
#plt.gca().axhline(true_com[1], linestyle="--", color=colors[1])
#plt.gca().axhline(true_com[2], linestyle="--", color=colors[2])
plt.gca().axhline(true_m, linestyle="--", color=colors[0])

# Show the covariances
plt.fill_between(t, theta_hat[0,:]-theta_std[0,:], theta_hat[0,:]+theta_std[0,:], color=colors[0], alpha=0.2)
#plt.fill_between(t, theta_hat[1,:]-theta_std[1,:], theta_hat[1,:]+theta_std[1,:], color=colors[1], alpha=0.2)
#plt.fill_between(t, theta_hat[2,:]-theta_std[2,:], theta_hat[2,:]+theta_std[2,:], color=colors[2], alpha=0.2)

plt.xlabel("Time (s)")
plt.ylabel("Parameter Estimate")

#plt.xlim(left=10)
plt.ylim(bottom=-0, top=0.05)
plt.legend()

plt.show()
