#!/usr/bin/env python

##
#
# Simple example of using Drake's manipulation station and showing the 
# associated diagram.
#
##

from pydrake.all import *
from pydrake.examples.manipulation_station import ManipulationStation

import numpy as np
import matplotlib.pyplot as plt

# Set up the manipulation station
station = ManipulationStation(time_step=0.01)
station.SetupManipulationClassStation()
station.Finalize()

# Show the system diagram
plt.figure()
plot_system_graphviz(station,max_depth=1)
plt.show()


print(type(station))
