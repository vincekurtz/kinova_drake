import numpy as np
from pydrake.all import *
from pydrake.geometry import (SceneGraph, MakeRenderEngineVtk, RenderEngineVtkParams, ClippingRange, RenderCameraCore, 
                              DepthRange, ColorRenderCamera, DepthRenderCamera, DrakeVisualizer, DrakeVisualizerParams,
                              Meshcat, MeshcatVisualizer)
from kinova_station.common import (EndEffectorTarget,
                                   JointTarget,
                                   GripperTarget, 
                                   EndEffectorWrenchCalculator,
                                   CameraPosePublisher)
from simulation_station import KinovaStation, CartesianController, GripperController, JointController, add_2f_85_bushings

import os
package_dir = os.path.dirname(os.path.abspath(__file__))


class ICLKinovaStation(KinovaStation):
    """
    System diagram for controlling 7 DoF Kinova Gen3 robot in ICL, modeled 
    after Drake's ManipulationStation, but with the kinova instead of a kuka arm.
   
                               ---------------------------------
                               |                               |
                               |                               |
                               |                               |
                               |                               | --> measured_arm_position
                               |                               | --> measured_arm_velocity
    ee/arm_target -----------> |        ICLKinovaStation       | --> measured_arm_torque
    ee/arm_target_type ------> |                               |
                               |                               |
                               |                               | --> measured_ee_pose
                               |                               | --> measured_ee_twist
                               |                               | --> measured_ee_wrench
    gripper_target ----------> |                               |
    gripper_target_type -----> |                               |
                               |                               | --> measured_gripper_position
                               |                               | --> measured_gripper_velocity
                               |                               |
                               |                               | --> camera_rgb_image
                               |                               | --> camera_depth_image
                               |                               | --> camera_transform
                               |                               |
                               |                               |
                               ---------------------------------

    The input ee_target can be a desired end-effector pose, twist or wrench, as specified by
    ee_target_type. The input arm_target can be desired joint positions, velocities or torques, 
    as specified by arm_target_type

    Similarly, the gripper_target can be a desired gripper position or velocity, as specified
    by gripper_target_type. 
   
    """
    def __init__(self, time_step=0.002, arm_controller_type='joint'):
        super().__init__(time_step=time_step, n_dof=7)  # kinova in ICL has n_dof = 7
        assert arm_controller_type=='joint' or arm_controller_type=='cartesian', \
                                        'Only joint controller and cartesian contoller are available'
        self.arm_controller_type = arm_controller_type
        self.set_name("ICL_kinova_manipulation_station")  # overwrite name

    def Finalize(self):
        """
        Do some final setup stuff. Must be called after making modifications
        to the station (e.g. adding arm, gripper, manipulands) and before using
        this diagram as a system. 
        """

        if self.gripper_type == "2f_85":
            # Add bushings to model the kinematic loop in the 2F-85 gripper.
            # This needs to be done pre-Finalize
            add_2f_85_bushings(self.plant, self.gripper)

        self.plant.Finalize()
        self.controller_plant.Finalize()
        
        # Set up the scene graph
        self.builder.Connect(
                self.scene_graph.get_query_output_port(),
                self.plant.get_geometry_query_input_port())
        self.builder.Connect(
                self.plant.get_geometry_poses_output_port(),
                self.scene_graph.get_source_pose_port(self.plant.get_source_id()))

        # Create controller
        if self.arm_controller_type=='cartesian':
            arm_controller = CartesianController(self.controller_plant, self.controller_arm)
            arm_controller.set_name("cartesian_controller")

            self.builder.AddSystem(arm_controller)
            # End effector target and target type go to the controller
            self.builder.ExportInput(arm_controller.ee_target_port,
                                    "ee_target")
            self.builder.ExportInput(arm_controller.ee_target_type_port,
                                    "ee_target_type")
        elif self.arm_controller_type=='joint':
            arm_controller = JointController(self.controller_plant, self.controller_arm)
            arm_controller.set_name("joint_controller")
            
            self.builder.AddSystem(arm_controller)
            # Arm (Joint) target and target type go to the controller
            self.builder.ExportInput(arm_controller.ee_target_port,
                                    "arm_target")
            self.builder.ExportInput(arm_controller.ee_target_type_port,
                                    "arm_target_type")
        else:
            raise ValueError(f"Invalid arm_controller_type: {self.arm_controller_type}")

        # Output measured arm position and velocity
        demux = self.builder.AddSystem(Demultiplexer(
                                        self.plant.num_multibody_states(self.arm),
                                        self.plant.num_positions(self.arm)))
        demux.set_name("demux")
        self.builder.Connect(
                self.plant.get_state_output_port(self.arm),
                demux.get_input_port(0))
        self.builder.ExportOutput(
                demux.get_output_port(0),
                "measured_arm_position")
        self.builder.ExportOutput(
                demux.get_output_port(1),
                "measured_arm_velocity")
        
        # Measured arm position and velocity are sent to the controller
        self.builder.Connect(
                demux.get_output_port(0),
                arm_controller.arm_position_port)
        self.builder.Connect(
                demux.get_output_port(1),
                arm_controller.arm_velocity_port)

        # Torques from controller go to the simulated plant
        self.builder.Connect(
                arm_controller.GetOutputPort("applied_arm_torque"),
                self.plant.get_actuation_input_port(self.arm))

        # Controller outputs measured arm torques, end-effector pose, end-effector twist
        self.builder.ExportOutput(
                arm_controller.GetOutputPort("applied_arm_torque"),
                "measured_arm_torque")
        self.builder.ExportOutput(
                arm_controller.GetOutputPort("measured_ee_pose"),
                "measured_ee_pose")
        self.builder.ExportOutput(
                arm_controller.GetOutputPort("measured_ee_twist"),
                "measured_ee_twist")
        
        # Create gripper controller
        gripper_controller = self.builder.AddSystem(GripperController(self.gripper_type))
        gripper_controller.set_name("gripper_controller")

        # Connect gripper controller to the diagram
        self.builder.ExportInput(
                gripper_controller.GetInputPort("gripper_target"),
                "gripper_target")
        self.builder.ExportInput(
                gripper_controller.GetInputPort("gripper_target_type"),
                "gripper_target_type")

        self.builder.Connect(
                self.plant.get_state_output_port(self.gripper),
                gripper_controller.GetInputPort("gripper_state"))
        self.builder.Connect(
                gripper_controller.GetOutputPort("applied_gripper_torque"),
                self.plant.get_actuation_input_port(self.gripper))
    
        # Send gripper position and velocity as an output
        self.builder.ExportOutput(
                gripper_controller.GetOutputPort("measured_gripper_position"),
                "measured_gripper_position")
        self.builder.ExportOutput(
                gripper_controller.GetOutputPort("measured_gripper_velocity"),
                "measured_gripper_velocity")
        
        # Compute and output end-effector wrenches based on measured joint torques
        wrench_calculator = self.builder.AddSystem(EndEffectorWrenchCalculator(
                self.controller_plant,
                self.controller_plant.GetFrameByName("end_effector")))
        wrench_calculator.set_name("wrench_calculator")

        self.builder.Connect(
                demux.get_output_port(0),
                wrench_calculator.GetInputPort("joint_positions"))
        self.builder.Connect(
                demux.get_output_port(1),
                wrench_calculator.GetInputPort("joint_velocities"))
        self.builder.Connect(
                arm_controller.GetOutputPort("applied_arm_torque"),
                wrench_calculator.GetInputPort("joint_torques"))

        self.builder.ExportOutput(
                wrench_calculator.get_output_port(),
                "measured_ee_wrench")

        # Configure camera
        if self.has_camera:
            
            # Create and add the camera system
            camera_parent_body_id = self.plant.GetBodyFrameIdIfExists(self.camera_parent_frame.body().index())
            camera = self.builder.AddSystem(RgbdSensor(camera_parent_body_id,
                                                        self.X_camera,
                                                        self.color_camera,
                                                        self.depth_camera))
            camera.set_name("camera")

            # Wire the camera to the scene graph
            self.builder.Connect(
                    self.scene_graph.get_query_output_port(),
                    camera.query_object_input_port())

            self.builder.ExportOutput(
                    camera.color_image_output_port(),
                    "camera_rgb_image")
            self.builder.ExportOutput(
                    camera.depth_image_32F_output_port(),
                    "camera_depth_image")

            # Send pose of camera in world frame as output
            X_ee_camera = self.X_ee.inverse().multiply(self.X_camera)
            camera_transform_pub = self.builder.AddSystem(CameraPosePublisher(X_ee_camera))
            camera_transform_pub.set_name("camera_transform_publisher")

            self.builder.Connect(
                    arm_controller.GetOutputPort("measured_ee_pose"),
                    camera_transform_pub.GetInputPort("ee_pose"))
            self.builder.ExportOutput(
                    camera_transform_pub.GetOutputPort("camera_transform"),
                    "camera_transform")

        # Build the diagram
        self.builder.BuildInto(self)
