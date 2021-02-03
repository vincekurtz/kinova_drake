##
#
# Simple example of creating a point cloud from the RGB-D camera
# and visualizing it with mesthcat.
#
##

from pydrake.all import *
import numpy as np
import matplotlib.pyplot as plt

from kinova_station import KinovaStation
from controllers import Command, CommandSequence, CommandSequenceController

########################### Parameters #################################

# Show the internal workings of the station
show_station_diagram = False

# Make a plot of the diagram for this example, where only the inputs
# and outputs of the station are shown
show_toplevel_diagram = False

# Which gripper to use (hande or 2f_85)
gripper_type = "hande"

########################################################################

# Set up the kinova station
station = KinovaStation(time_step=0.001)
station.SetupSinglePegScenario(gripper_type=gripper_type, arm_damping=False)
station.AddCamera()
station.ConnectToMeshcatVisualizer()
station.Finalize()

if show_station_diagram:
    plt.figure()
    plot_system_graphviz(station,max_depth=1)
    plt.show()


# Connect input ports to the kinova station
builder = DiagramBuilder()
builder.AddSystem(station)

# Define a sequence of end-effector and gripper targets
cs = CommandSequence([])
cs.append(Command(
    name="front_view",
    target_pose=np.array([0.7*np.pi, 0.0, 0.5*np.pi, 0.5, 0.0, 0.15]),
    duration=3,
    gripper_closed=False))
cs.append(Command(
    name="left_view",
    target_pose=np.array([0.7*np.pi, 0.0, 0.2*np.pi, 0.6, 0.3, 0.15]),
    duration=3,
    gripper_closed=False))
cs.append(Command(
    name="front_view",
    target_pose=np.array([0.7*np.pi, 0.0, 0.5*np.pi, 0.5, 0.0, 0.15]),
    duration=3,
    gripper_closed=False))
cs.append(Command(
    name="right_view",
    target_pose=np.array([0.7*np.pi, 0.0, 0.8*np.pi, 0.6, -0.3, 0.15]),
    duration=3,
    gripper_closed=False))
cs.append(Command(
    name="home",
    target_pose=np.array([0.5*np.pi, 0.0, 0.5*np.pi, 0.5, 0.0, 0.2]),
    duration=3,
    gripper_closed=False))

# Add an associated controller
controller = builder.AddSystem(CommandSequenceController(cs))
controller.set_name("controller")
controller.ConnectToStation(builder, station)

# Convert the depth image to a point cloud
# Note that this system block is slow
point_cloud_generator = builder.AddSystem(DepthImageToPointCloud(
                                    CameraInfo(width=480, height=270, fov_y=np.radians(40)),
                                    fields=BaseField.kXYZs | BaseField.kRGBs))
point_cloud_generator.set_name("point_cloud_generator")
builder.Connect(
        station.GetOutputPort("camera_depth_image"),
        point_cloud_generator.depth_image_input_port())
builder.Connect(
        station.GetOutputPort("camera_rgb_image"),
        point_cloud_generator.color_image_input_port())

# Connect camera pose to point cloud generator
builder.Connect(
        station.GetOutputPort("camera_transform"),
        point_cloud_generator.GetInputPort("camera_pose"))

# Visualize the point cloud with meshcat
meshcat_point_cloud = builder.AddSystem(MeshcatPointCloudVisualizer(
                                            station.meshcat,
                                            draw_period=0.2))
meshcat_point_cloud.set_name("point_cloud_viz")
builder.Connect(
        point_cloud_generator.point_cloud_output_port(),
        meshcat_point_cloud.get_input_port())

# Build the system diagram
diagram = builder.Build()
diagram.set_name("toplevel_system_diagram")
diagram_context = diagram.CreateDefaultContext()

if show_toplevel_diagram:
    # Show the overall system diagram
    plt.figure()
    plot_system_graphviz(diagram,max_depth=1)
    plt.show()

# Set starting positions
station.go_home(diagram, diagram_context, name="Home")
station.SetManipulandStartPositions(diagram, diagram_context)

# Set up simulation
simulator = Simulator(diagram, diagram_context)
simulator.set_target_realtime_rate(10.1)
simulator.set_publish_every_time_step(False)

# Run simulation
simulator.Initialize()
simulator.AdvanceTo(15.0)
