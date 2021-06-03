##
#
# Simple example of using our kinova manipulation station to pick up a peg
# at an a-priori known location. Runs in simulation.
#
##

from pydrake.all import *
import numpy as np
import matplotlib.pyplot as plt

from kinova_station import KinovaStation, EndEffectorTarget, GripperTarget
from controllers import CommandSequenceController, CommandSequence, Command

########################### Parameters #################################

# Make a plot of the inner workings of the station
show_station_diagram = False

# Make a plot of the diagram for this example, where only the inputs
# and outputs of the station are shown
show_toplevel_diagram = False

# Which gripper to use (hande or 2f_85)
gripper_type = "hande"

########################################################################

# Set up the kinova station
station = KinovaStation(time_step=0.002)
station.SetupSinglePegScenario(gripper_type=gripper_type, arm_damping=False)
station.Finalize()

if show_station_diagram:
    # Show the station's system diagram
    plt.figure()
    plot_system_graphviz(station,max_depth=1)
    plt.show()

# Start assembling the overall system diagram
builder = DiagramBuilder()
station = builder.AddSystem(station)

# Create the command sequence
cs = CommandSequence([])
cs.append(Command(
    name="pregrasp",
    target_pose=np.array([0.5*np.pi, 0.0, 0.5*np.pi, 0.8, 0.0, 0.1]),
    duration=4,
    gripper_closed=False))
cs.append(Command(
    name="grasp",
    target_pose=np.array([0.5*np.pi, 0.0, 0.5*np.pi, 0.8, 0.0, 0.1]),
    duration=1,
    gripper_closed=True))
cs.append(Command(
    name="lift",
    target_pose=np.array([0.5*np.pi, 0.0, 0.5*np.pi, 0.5, 0.0, 0.5]),
    duration=2,
    gripper_closed=True))
cs.append(Command(
    name="move",
    target_pose=np.array([0.5*np.pi, 0.0, 0.1*np.pi, 0.5, -0.5, 0.5]),
    duration=2,
    gripper_closed=True))

# Create the controller and connect inputs and outputs appropriately
Kp = 10*np.eye(6)
Kd = 2*np.sqrt(Kp)

controller = builder.AddSystem(CommandSequenceController(
    cs,
    command_type=EndEffectorTarget.kTwist,  # Twist commands seem most reliable in simulation
    Kp=Kp,
    Kd=Kd))
controller.set_name("controller")
controller.ConnectToStation(builder, station)

# Build the system diagram
diagram = builder.Build()
diagram.set_name("system_diagram")
diagram_context = diagram.CreateDefaultContext()

if show_toplevel_diagram:
    # Show the overall system diagram
    plt.figure()
    plot_system_graphviz(diagram,max_depth=1)
    plt.show()

# Set default arm positions
station.go_home(diagram, diagram_context, name="Home")

# Set starting position for any objects in the scene
station.SetManipulandStartPositions(diagram, diagram_context)

# Set up simulation
simulator = Simulator(diagram, diagram_context)
simulator.set_target_realtime_rate(1.0)
simulator.set_publish_every_time_step(False)

# Run simulation
simulator.Initialize()
simulator.AdvanceTo(10.0)
