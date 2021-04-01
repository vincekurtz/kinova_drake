##
#
# Simple example of reading depth images from the hardware camera and using them
# to create and publish a point cloud, which can be visualized with Meshcat.
#
##


from pydrake.all import *
import numpy as np
import matplotlib.pyplot as plt

from kinova_station import KinovaStationHardwareInterface, EndEffectorTarget
from controllers import CommandSequenceController, CommandSequence, Command, PointCloudController

from meshcat.servers.zmqserver import start_zmq_server_as_subprocess

########################### Parameters #################################

# Make a plot of the inner workings of the station
show_station_diagram = False

# Make a plot of the diagram for this example, where only the inputs
# and outputs of the station are shown
show_toplevel_diagram = False

########################################################################

with KinovaStationHardwareInterface() as station:

    if show_station_diagram:
        # Show the station's system diagram
        plt.figure()
        plot_system_graphviz(station,max_depth=1)
        plt.show()

    # Start assembling the overall system diagram
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    station = builder.AddSystem(station)

    # Create the controller and connect inputs and outputs appropriately
    Kp = 1*np.diag([100, 100, 100, 200, 200, 200])  # high gains needed to overcome
    Kd = 2*np.sqrt(0.5*Kp)                          # significant joint friction

    controller = builder.AddSystem(PointCloudController(
        command_type=EndEffectorTarget.kWrench,  # wrench commands work best on hardware
        show_candidate_grasp=True,
        hardware=True,
        Kp=Kp,
        Kd=Kd))

    controller.set_name("controller")
    controller.ConnectToStation(builder, station)

    # Add converter from depth images to point clouds
    point_cloud_generator = builder.AddSystem(DepthImageToPointCloud(
                                    CameraInfo(width=480, height=270, fov_y=np.radians(40)),
                                    pixel_type=PixelType.kDepth16U,
                                    scale=0.001,
                                    fields=BaseField.kXYZs))
    point_cloud_generator.set_name("point_cloud_generator")

    builder.Connect(
            station.GetOutputPort("camera_depth_image"),
            point_cloud_generator.depth_image_input_port())
    builder.Connect(
            station.GetOutputPort("camera_transform"),
            point_cloud_generator.GetInputPort("camera_pose"))

    # Send generated point cloud and camera transform to the controller
    builder.Connect(
            point_cloud_generator.point_cloud_output_port(),
            controller.GetInputPort("point_cloud"))
    builder.Connect(
            station.GetOutputPort("camera_transform"),
            controller.GetInputPort("camera_transform"))

    # Connect meshcat visualizer
    #proc, zmq_url, web_url = start_zmq_server_as_subprocess()  # start meshcat from here
    # Alternative: start meshcat (in drake dir) with bazel run @meshcat_python//:meshcat-server
    zmq_url = "tcp://127.0.0.1:6000"

    meshcat = ConnectMeshcatVisualizer(
            builder=builder, 
            scene_graph=scene_graph,
            zmq_url=zmq_url)

    meshcat_point_cloud = builder.AddSystem(MeshcatPointCloudVisualizer(
                                                meshcat,
                                                draw_period=0.2))
    meshcat_point_cloud.set_name("point_cloud_viz")
    builder.Connect(
            point_cloud_generator.point_cloud_output_port(),
            meshcat_point_cloud.get_input_port())

    # Build the system diagram
    diagram = builder.Build()
    diagram.set_name("system_diagram")
    plant.Finalize()
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
    simulator.AdvanceTo(30.0)

    # Print rate data
    print("")
    print("Target control frequency: %s Hz" % (1/time_step))
    print("Actual control frequency: %s Hz" % (1/time_step * simulator.get_actual_realtime_rate()))

