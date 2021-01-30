#!/usr/bin/env python

##
#
# Figure out how to handle depth images aquired from the hardware interface. 
#
##

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pydrake.all import *
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess

# load saved image
with open("depth_image_saved.npy", "rb") as f:
    frame = np.load(f)

# Create a drake depth image
pixel_type = PixelType.kDepth16U
depth_image = Image[pixel_type](width=frame.shape[1],height=frame.shape[0])
depth_image.mutable_data[:,:] = frame.T[np.newaxis].T

# Set up a point cloud publisher
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

depth_pub = builder.AddSystem(
        ConstantValueSource(AbstractValue.Make(depth_image)))
depth_pub.set_name("depth_publisher")

camera_info = CameraInfo(
        width=270,
        height=480,
        fov_y=np.radians(80))  # from https://www.intel.com/content/dam/support/us/en/documents/emerging-technologies/intel-realsense-technology/Intel-RealSense-D400-Series-Datasheet.pdf
point_cloud_gen = builder.AddSystem(
        DepthImageToPointCloud(camera_info, pixel_type, scale=1./1000))
point_cloud_gen.set_name("point_cloud_generator")

builder.Connect(
        depth_pub.get_output_port(),
        point_cloud_gen.depth_image_input_port())

# Set up meshcat viewer
proc, zmq_url, web_url = start_zmq_server_as_subprocess()

meshcat = ConnectMeshcatVisualizer(
        builder=builder, 
        scene_graph=scene_graph,
        zmq_url=zmq_url)
X_Camera = RigidTransform()
X_Camera.set_translation([0,0,0.2])
X_Camera.set_rotation(RotationMatrix(RollPitchYaw([-0.8*np.pi,0,0])))
meshcat_pointcloud = builder.AddSystem(MeshcatPointCloudVisualizer(meshcat, X_WP=X_Camera))

builder.Connect(
        point_cloud_gen.point_cloud_output_port(),
        meshcat_pointcloud.get_input_port())

diagram = builder.Build()

#plt.figure()
#plot_system_graphviz(diagram)
#plt.show()

# Evaluate camera outputs to get the image
plant.Finalize()
context = diagram.CreateDefaultContext()

diagram.Publish(context)

# Do something to avoid quitting immediately
plt.imshow(np.squeeze(depth_image.data))
plt.show()

