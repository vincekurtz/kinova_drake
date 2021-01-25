##
#
# Simple example of using our kinova manipulation station to pick up a peg
# at an a-priori known location. Runs on the real hardware.
#
##

from pydrake.all import *
import numpy as np
import matplotlib.pyplot as plt

from kinova_station import KinovaStationHardwareInterface, EndEffectorTarget, GripperTarget
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
with KinovaStationHardwareInterface() as station:

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
        target_pose=np.array([0.5*np.pi, 0.0, 0.5*np.pi, 0.68, 0.0, 0.08]),
        duration=4,
        gripper_closed=False))
    cs.append(Command(
        name="grasp",
        target_pose=np.array([0.5*np.pi, 0.0, 0.5*np.pi, 0.68, 0.0, 0.08]),
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
    Kp = np.diag([100, 100, 100, 200, 200, 200])  # high gains needed to overcome
    Kd = 2*np.sqrt(0.5*Kp)                        # significant joint friction

    controller = builder.AddSystem(CommandSequenceController(
        cs,
        command_type=EndEffectorTarget.kWrench,  # wrench commands work best on hardware
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
    station.go_home(name="Home")

    # Set up simulation
    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(False)

    integration_scheme = "explicit_euler"
    time_step = 0.10
    ResetIntegratorFromFlags(simulator, integration_scheme, time_step)

    # Run simulation
    simulator.Initialize()
    simulator.AdvanceTo(20.0)

    # Print rate data
    print("")
    print("Target control frequency: %s Hz" % (1/time_step))
    print("Actual control frequency: %s Hz" % (1/time_step * simulator.get_actual_realtime_rate()))

