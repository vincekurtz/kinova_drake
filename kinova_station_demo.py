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

# Set up the kinova station
station = KinovaStation(time_step=0.001)
station.SetupArmOnly()
station.AddGround()
station.ConnectToDrakeVisualizer()
station.Finalize()

# Connect input ports to the kinova station
builder = DiagramBuilder()
builder.AddSystem(station)

# Desired end-effector pose
rpy_xyz_des = np.array([np.pi,-0.01,0.0,
                        0.5,0.1,0.2])
target_ee_pose = builder.AddSystem(ConstantVectorSource(rpy_xyz_des))
builder.Connect(
        target_ee_pose.get_output_port(0),
        station.GetInputPort("target_ee_pose"))

# Desired end-effector twist
# TODO

# Desired end-effector wrench
# TODO

diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()



# Set up simulation
simulator = Simulator(diagram, diagram_context)
simulator.set_target_realtime_rate(1.0)
simulator.set_publish_every_time_step(False)

simulator.Initialize()
simulator.AdvanceTo(10)


# Show the system diagram
#plt.figure()
#plot_system_graphviz(station,max_depth=1)
#plt.show()
