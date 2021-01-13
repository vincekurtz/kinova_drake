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
station.ConnectToDrakeVisualizer()
station.Finalize()

# Show the system diagram
#plt.figure()
#plot_system_graphviz(station,max_depth=1)
#plt.show()

# Set up simulation
diagram_context = station.CreateDefaultContext()

simulator = Simulator(station, diagram_context)
simulator.set_target_realtime_rate(1.0)
simulator.set_publish_every_time_step(False)

simulator.Initialize()
simulator.AdvanceTo(5)

