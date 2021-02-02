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

# load saved images
with open("depth_image_saved.npy", "rb") as f:
    depth_frame = np.load(f)

with open("color_image_saved.npy", "rb") as f:
    color_frame = np.load(f)

# Do some rescaling
pts1 = np.float32([[245,182],[367,147],[101,160],[251,234]])   # points on the depth image
pts2 = np.float32([[232,157],[391,93],[17,124],[230,232]])     # corresponding points on the color image

M = cv2.getPerspectiveTransform(pts1,pts2)

depth_frame = cv2.warpPerspective(depth_frame,M,(480,270))



# Show the resulting point cloud over meshcat
pixel_type = PixelType.kDepth16U
depth_image = Image[pixel_type](width=depth_frame.shape[1],height=depth_frame.shape[0])
depth_image.mutable_data[:,:,:] = depth_frame.reshape(270,480,1)

color_image = Image[PixelType.kRgba8U](width=color_frame.shape[1], height=color_frame.shape[0])
color_image.mutable_data[:,:,:] = color_frame

# Set up a point cloud publisher
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

depth_pub = builder.AddSystem(
        ConstantValueSource(AbstractValue.Make(depth_image)))
depth_pub.set_name("depth_publisher")

color_pub = builder.AddSystem(
        ConstantValueSource(AbstractValue.Make(color_image)))

camera_info = CameraInfo(
        width=480,
        height=270,
        fov_y=np.radians(40))  # from https://www.intel.com/content/dam/support/us/en/documents/emerging-technologies/intel-realsense-technology/Intel-RealSense-D400-Series-Datasheet.pdf
point_cloud_gen = builder.AddSystem(
        DepthImageToPointCloud(camera_info, 
                               pixel_type, 
                               scale=1./1000,
                               fields=BaseField.kXYZs | BaseField.kRGBs))
point_cloud_gen.set_name("point_cloud_generator")

builder.Connect(
        depth_pub.get_output_port(),
        point_cloud_gen.depth_image_input_port())
builder.Connect(
        color_pub.get_output_port(),
        point_cloud_gen.color_image_input_port())

# Set up meshcat viewer
#proc, zmq_url, web_url = start_zmq_server_as_subprocess()
zmq_url = "tcp://127.0.0.1:6000"

meshcat = ConnectMeshcatVisualizer(
        builder=builder, 
        scene_graph=scene_graph,
        zmq_url=zmq_url)
X_Camera = RigidTransform()
X_Camera.set_translation([0,0,0.25])
X_Camera.set_rotation(RotationMatrix(RollPitchYaw([0.5*np.pi,np.pi,0])))
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

# Plot the images
plt.subplot(2,1,1)
plt.imshow(color_frame)
plt.subplot(2,1,2)
plt.imshow(depth_frame.squeeze())
plt.show()
