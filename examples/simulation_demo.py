##
#
# Simple example of using our kinova manipulation station in simulation with various options
#
#
# The kinova manipulation station is a system with the following inputs
# and outputs:
#
#                              ---------------------------------
#                              |                               |
#                              |                               |
#                              |                               |
#                              |                               | --> measured_arm_position
#                              |                               | --> measured_arm_velocity
#   ee_target ---------------> |         KinovaStation         | --> measured_arm_torque
#   ee_target_type ----------> |                               |
#                              |                               |
#                              |                               | --> measured_ee_pose
#                              |                               | --> measured_ee_twist
#                              |                               | --> measured_ee_wrench
#   gripper_target ----------> |                               |
#   gripper_target_type -----> |                               |
#                              |                               | --> measured_gripper_position
#                              |                               | --> measured_gripper_velocity
#                              |                               |
#                              |                               | --> camera_rgb_image
#                              |                               | --> camera_depth_image
#                              |                               | --> camera_transform
#                              |                               |
#                              |                               |
#                              ---------------------------------
#
# The end-effector target (ee_target) can be a pose, a wrench or a twist. 
# The gripper target (gripper_target) can be a position or a velocity. 
#
#
# See the "Parameters" section below for different ways of using and visualizing
# this system. 
#
##

from pydrake.all import *
import numpy as np
import matplotlib.pyplot as plt

from kinova_station import KinovaStation, EndEffectorTarget, GripperTarget
from observers.camera_viewer import CameraViewer

########################### Parameters #################################

# Make a plot of the inner workings of the station
show_station_diagram = False

# Make a plot of the diagram for this example, where only the inputs
# and outputs of the station are shown
show_toplevel_diagram = False

# Run a quick simulation
simulate = True

# If we're running a simulation, choose which sort of commands are
# sent to the arm and the gripper
ee_command_type = EndEffectorTarget.kTwist      # kPose, kTwist, or kWrench
gripper_command_type = GripperTarget.kPosition  # kPosition or kVelocity

# If we're running a simulation, whether to include a simulated camera
# and show the associated image
include_camera = False
show_camera_window = False

# Which gripper to use (hande or 2f_85)
gripper_type = "hande"

########################################################################

# Set up the kinova station
station = KinovaStation(time_step=0.002, n_dof=7)
station.SetupSinglePegScenario(gripper_type=gripper_type, arm_damping=False)
if include_camera:
    station.AddCamera(show_window=show_camera_window)
    station.ConnectToMeshcatVisualizer()

station.Finalize()

if show_station_diagram:
    # Show the station's system diagram
    plt.figure()
    plot_system_graphviz(station,max_depth=1)
    plt.show()

# Connect input ports to the kinova station
builder = DiagramBuilder()
builder.AddSystem(station)

# Set (constant) command to send to the system
if ee_command_type == EndEffectorTarget.kPose:
    pose_des = np.array([np.pi,0.0,0.0,
                            0.6,0.0,0.2])
    target_source = builder.AddSystem(ConstantVectorSource(pose_des))

elif ee_command_type == EndEffectorTarget.kTwist:
    twist_des = np.array([0.0,0.1,0.0,
                          0.0,0.0,0.0])
    target_source = builder.AddSystem(ConstantVectorSource(twist_des))

elif ee_command_type == EndEffectorTarget.kWrench:
    wrench_des = np.array([0.0,0.0,0.0,
                            0.0,0.0,0.0])
    target_source = builder.AddSystem(ConstantVectorSource(wrench_des))

else:
    raise RuntimeError("invalid end-effector target type")

# Send end-effector command and type
target_type_source = builder.AddSystem(ConstantValueSource(AbstractValue.Make(ee_command_type)))
builder.Connect(
        target_type_source.get_output_port(),
        station.GetInputPort("ee_target_type"))

builder.Connect(
        target_source.get_output_port(),
        station.GetInputPort("ee_target"))

target_source.set_name("ee_command_source")
target_type_source.set_name("ee_type_source")

# Set gripper command
if gripper_command_type == GripperTarget.kPosition:
    q_grip_des = np.array([0])   # open at 0, closed at 1
    gripper_target_source = builder.AddSystem(ConstantVectorSource(q_grip_des))

elif gripper_command_type == GripperTarget.kVelocity:
    v_grip_des = np.array([1.0])
    gripper_target_source = builder.AddSystem(ConstantVectorSource(v_grip_des))

# Send gripper command and type
gripper_target_type_source = builder.AddSystem(ConstantValueSource(
                                         AbstractValue.Make(gripper_command_type)))
builder.Connect(
        gripper_target_type_source.get_output_port(),
        station.GetInputPort("gripper_target_type"))

builder.Connect(
        gripper_target_source.get_output_port(),
        station.GetInputPort("gripper_target"))

gripper_target_source.set_name("gripper_command_source")
gripper_target_type_source.set_name("gripper_type_source")

# Loggers force certain outputs to be computed
wrench_logger = LogVectorOutput(station.GetOutputPort("measured_ee_wrench"),builder)
wrench_logger.set_name("wrench_logger")

pose_logger = LogVectorOutput(station.GetOutputPort("measured_ee_pose"), builder)
pose_logger.set_name("pose_logger")

twist_logger = LogVectorOutput(station.GetOutputPort("measured_ee_twist"), builder)
twist_logger.set_name("twist_logger")

gripper_logger = LogVectorOutput(station.GetOutputPort("measured_gripper_velocity"), builder)
gripper_logger.set_name("gripper_logger")

if include_camera:
    # Camera observer allows us to access camera data, and must be connected
    # to view the camera stream.
    camera_viewer = builder.AddSystem(CameraViewer())
    camera_viewer.set_name("camera_viewer")

    builder.Connect(
            station.GetOutputPort("camera_rgb_image"),
            camera_viewer.GetInputPort("color_image"))
    builder.Connect(
            station.GetOutputPort("camera_depth_image"),
            camera_viewer.GetInputPort("depth_image"))

    # Convert the depth image to a point cloud
    point_cloud_generator = builder.AddSystem(DepthImageToPointCloud(
                                        CameraInfo(width=480, height=270, fov_y=np.radians(40)),
                                        fields=BaseField.kXYZs | BaseField.kRGBs))
    point_cloud_generator.set_name("point_cloud_generator")
    builder.Connect(
            station.GetOutputPort("camera_depth_image"),
            point_cloud_generator.depth_image_input_port())

    # Connect camera pose to point cloud generator
    builder.Connect(
            station.GetOutputPort("camera_transform"),
            point_cloud_generator.GetInputPort("camera_pose"))

    # Visualize the point cloud with meshcat
    meshcat_point_cloud = builder.AddSystem(
            MeshcatPointCloudVisualizer(station.meshcat, "point_cloud", 0.2))
    meshcat_point_cloud.set_name("point_cloud_viz")
    builder.Connect(
            point_cloud_generator.point_cloud_output_port(),
            meshcat_point_cloud.cloud_input_port())

# Build the system diagram
diagram = builder.Build()
diagram.set_name("toplevel_system_diagram")
diagram_context = diagram.CreateDefaultContext()

if show_toplevel_diagram:
    # Show the overall system diagram
    plt.figure()
    plot_system_graphviz(diagram,max_depth=1)
    plt.show()

if simulate:
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
    simulator.AdvanceTo(5.0)
