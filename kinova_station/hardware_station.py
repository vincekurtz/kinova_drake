from pydrake.all import *

import time
import sys
import threading

from kinova_station.common import (EndEffectorTarget, 
                                   GripperTarget, 
                                   EndEffectorWrenchCalculator)

from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager

from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

from kortex_api.autogen.messages import DeviceConfig_pb2, Session_pb2, Base_pb2, VisionConfig_pb2

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import cv2

import matplotlib.pyplot as plt  # DEBUG


class KinovaStationHardwareInterface(LeafSystem):
    """
    A system block for controlling a 7 DoF Kinova Gen3 robot, modeled 
    after Drake's ManipulationStationHardwareInterface, but with the kinova 
    instead of a kuka arm.
   
                               ------------------------------------
                               |                                  |
                               |                                  |
                               |                                  |
                               |                                  | --> measured_arm_position
                               |                                  | --> measured_arm_velocity
    ee_target ---------------> |  KinovaStationHardwareInterface  | --> measured_arm_torque
    ee_target_type ----------> |                                  |
                               |                                  |
                               |                                  | --> measured_ee_pose
                               |                                  | --> measured_ee_twist
                               |                                  | --> measured_ee_wrench
    gripper_target ----------> |                                  |
    gripper_target_type -----> |                                  |
                               |                                  | --> measured_gripper_position
                               |                                  | --> measured_gripper_velocity
                               |                                  |
                               |                                  | --> camera_rgb_image
                               |                                  | --> camera_depth_image
                               |                                  | --> camera_transform
                               |                                  |
                               |                                  |
                               -----------------------------------

    The input ee_target can be a desired end-effector pose, twist or wrench, as specified by
    ee_target_type. 

    Similarly, the gripper_target can be a desired gripper position or velocity, as specified
    by gripper_target_type. 
   
    """
    def __init__(self):
        LeafSystem.__init__(self) 
        self.set_name("kinova_hardware_interface")

        # Declare input ports
        self.ee_target_port = self.DeclareVectorInputPort(
                                        "ee_target",
                                        BasicVector(6))
        self.ee_target_type_port = self.DeclareAbstractInputPort(
                                        "ee_target_type",
                                        AbstractValue.Make(EndEffectorTarget.kPose))
        
        self.gripper_target_port = self.DeclareVectorInputPort(
                                             "gripper_target",
                                             BasicVector(1))
        self.gripper_target_type_port = self.DeclareAbstractInputPort(
                                            "gripper_target_type",
                                            AbstractValue.Make(GripperTarget.kPosition))

        # Declare output ports
        self.DeclareVectorOutputPort(
                "measured_arm_position",
                BasicVector(7),
                self.CalcArmPosition)
        self.DeclareVectorOutputPort(
                "measured_arm_velocity",
                BasicVector(7),
                self.CalcArmVelocity)
        self.DeclareVectorOutputPort(
                "measured_arm_torque",
                BasicVector(7),
                self.CalcArmTorque)

        self.DeclareVectorOutputPort(
                "measured_ee_pose",
                BasicVector(6),
                self.CalcEndEffectorPose,
                {self.time_ticket()})
        self.DeclareVectorOutputPort(
                "measured_ee_twist",
                BasicVector(6),
                self.CalcEndEffectorTwist,
                {self.time_ticket()})
        self.DeclareVectorOutputPort(
                "measured_ee_wrench",
                BasicVector(6),
                self.CalcEndEffectorWrench)

        self.DeclareVectorOutputPort(
                "measured_gripper_position",
                BasicVector(1),
                self.CalcGripperPosition,
                {self.time_ticket()})
        self.DeclareVectorOutputPort(
                "measured_gripper_velocity",
                BasicVector(1),
                self.CalcGripperVelocity,
                {self.time_ticket()})

        sample_rgb_image = Image[PixelType.kRgba8U](width=480, height=270)
        sample_depth_image = Image[PixelType.kDepth16U](width=480, height=270)

        self.DeclareAbstractOutputPort(
                "camera_rgb_image",
                lambda: AbstractValue.Make(sample_rgb_image),
                self.CaptureRgbImage,
                {self.time_ticket()})
        self.DeclareAbstractOutputPort(
                "camera_depth_image",
                lambda: AbstractValue.Make(sample_depth_image),
                self.CaptureDepthImage,
                {self.time_ticket()})
        self.DeclareAbstractOutputPort(
                "camera_transform",
                lambda: AbstractValue.Make(RigidTransform()),   # TODO: use CameraPosePublisher?
                self.CalcCameraTransform,
                {self.time_ticket()})

        # Create a dummy continuous state so that the simulator
        # knows not to just jump to the last possible timestep
        self.DeclareContinuousState(1)

        # Each call to self.base_cyclic.RefreshFeedback() takes about 0.025s, so we'll
        # try to minimize redudant calls to the base as much as possible by storing
        # feedback from the base at each timestep.
        self.last_feedback_time = -np.inf
        self.feedback = None
    
        # Store end-effector pose (for computing camera pose)
        self.ee_pose = np.zeros(6)

    def __enter__(self):
        """
        Start an API instance and connect to the hardware. This is called
        at the beginning of a 'with' statement'.
        """
        print("Opening Hardware Connection")
        TCP_PORT = 10000   # UDP port is 10001

        # Set up API
        self.transport = TCPTransport()
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)
        self.transport.connect('192.168.1.10', TCP_PORT)

        # Create session
        session_info = Session_pb2.CreateSessionInfo()
        session_info.username = "admin"
        session_info.password = "admin"
        session_info.session_inactivity_timeout = 60000   # (milliseconds)
        session_info.connection_inactivity_timeout = 2000 # (milliseconds)

        self.session_manager = SessionManager(self.router)
        self.session_manager.CreateSession(session_info)

        # Create required services
        device_config = DeviceConfigClient(self.router)
        self.base = BaseClient(self.router)
        self.base_cyclic = BaseCyclicClient(self.router)

        if self.base.GetArmState().active_state != Base_pb2.ARMSTATE_SERVOING_READY:
            print("")
            print("ERROR: arm not in ready state.")

            print(self.base.GetArmState())

            print("Make sure there is nothing else currently sending commands (e.g. joystick, web interface), ")
            print("and clear any faults before trying again.")
            sys.exit(0)

        # Set up depth image pipeline. We use raw Gstramer for the depth image, since 
        # OpenCV only supports 8bit images and we want 16bit depth images for full resolution.
        Gst.init(None)
        depth_command = 'rtspsrc location=rtsp://192.168.1.10/depth latency=30 ! rtpgstdepay ! videoconvert ! appsink'

        depth_video_pipe = Gst.parse_launch(depth_command)
        depth_video_pipe.set_state(Gst.State.PLAYING)
        self.depth_video_sink = depth_video_pipe.get_by_name('appsink0')
        self.depth_video_sink.set_property('sync', False)
        self.depth_video_sink.set_property('max-buffers', 2)
        self.depth_video_sink.set_property('drop', True)

        # Set up the color image pipeline using OpenCV
        color_command = 'rtspsrc location=rtsp://192.168.1.10/color latency=30 ! rtph264depay ! avdec_h264 ! videoconvert ! appsink drop=true max-buffers=2'
        self.color_stream = cv2.VideoCapture(color_command)

        # DEBUG: set color and depth image sizes
        #device_manager = DeviceManagerClient(self.router)
        #vision_config = VisionConfigClient(self.router)

        #all_devices_info = device_manager.ReadAllDevices()
        #
        #vision_handles = [ hd for hd in all_devices_info.device_handle if hd.device_type == DeviceConfig_pb2.VISION ]
        #if len(vision_handles) == 0:
        #    raise RuntimeError("There is no vision device registered in the devices info")
        #elif len(vision_handles) > 1:
        #    raise RuntimeError("There is more than one vision device registered in the devices info")
        #
        #vision_device_id = vision_handles[0].device_identifier

        #profile_id = VisionConfig_pb2.IntrinsicProfileIdentifier()
        #profile_id.sensor = VisionConfig_pb2.SENSOR_COLOR
        #profile_id.resolution = VisionConfig_pb2.RESOLUTION_640x480
        #
        #intrinsics = vision_config.GetIntrinsicParametersProfile(profile_id, vision_device_id)
        #vision_config.SetIntrinsicParameters(intrinsics, vision_device_id)

        print("Hardware Connection Open.\n")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Disconnect from the API instance and close everything down. This
        is called at the end of a 'with' statement.
        """
        print("\nClosing Hardware Connection...")

        if self.session_manager is not None:
            router_options = RouterClientSendOptions()
            router_options.timeout_ms = 1000

            self.session_manager.CloseSession()

        self.transport.disconnect()
        print("Hardware Connection Closed.")

    def check_for_end_or_abort(self, e):
        """
        Return a closure checking for END or ABORT notifications

        Arguments:
        e -- event to signal when the action is completed
            (will be set when an END or ABORT occurs)
        """
        def check(notification, e = e):
            print("EVENT : " + \
                  Base_pb2.ActionEvent.Name(notification.action_event))
            if notification.action_event == Base_pb2.ACTION_END \
            or notification.action_event == Base_pb2.ACTION_ABORT:
                e.set()
        return check

    def GetFeedback(self, current_time):
        """
        Sets self.feedback with the latest data from the controller. 
        We also indicate the time at which this feedback was set, in an 
        effor to reduce redudnant calls to self.base_cyclic.RefreshFeedback(),
        which take about 25ms each. 
        """
        self.feedback = self.base_cyclic.RefreshFeedback()
        self.last_feedback_time = current_time

    def go_home(self, name="Home"):
        """
        Move the arm to the home position. Different positions can be
        specified using the 'name' parameter ('Home', 'Retract', 'Packaging', or 'Zero').
        """
        # Make sure the arm is in Single Level Servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        # Move arm to ready position
        print("Moving the arm to the home position")
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == name:
                action_handle = action.handle

        if action_handle == None:
            print("Invalid home position name: %s" % name)
            print("Must be one of ['Home','Retract','Packaging','Zero']")
            print("Exiting.")
            sys.exit(0)

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        self.base.ExecuteActionFromReference(action_handle)

        # Leave time to action to complete
        TIMEOUT_DURATION = 20  # seconds
        finished = e.wait(TIMEOUT_DURATION)   
        self.base.Unsubscribe(notification_handle)

        if finished:
            print("Home position reached")
        else:
            print("Timeout while moving to home position")
        return finished

    def send_gripper_command(self, mode, command):
        """
        Send a position or a velocity command to the gripper
        """
        assert (mode == Base_pb2.GRIPPER_POSITION) or (mode == Base_pb2.GRIPPER_SPEED)

        gripper_command = Base_pb2.GripperCommand()
        gripper_command.mode = mode
        
        finger = gripper_command.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = command

        self.base.SendGripperCommand(gripper_command)

    def send_gripper_position_command(self, position):
        """
        Convienience method for sending a position command 
        (real number in [0,1]) to the gripper. 
        """
        self.send_gripper_command(
                mode = Base_pb2.GRIPPER_POSITION,
                command = position)

    def send_gripper_velocity_command(self, velocity):
        """
        Convienience method for sending a velocity command
        to the gripper.
        """
        self.send_gripper_command(
                mode = Base_pb2.GRIPPER_SPEED,
                command = velocity)

    def send_pose_command(self, pose):
        """
        Convienience method for sending a target end-effector pose
        to the robot. 

        WARNING: unlike the twist and wrench commands, this command
        stops everything and just moves the end-effector to the 
        desired pose.
        """
        action = Base_pb2.Action()
        action.name = "End-effector pose command"
        action.application_data = ""
        
        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.theta_x = np.degrees(pose[0])
        cartesian_pose.theta_y = np.degrees(pose[1])
        cartesian_pose.theta_z = np.degrees(pose[2])
        cartesian_pose.x = pose[3]
        cartesian_pose.y = pose[4]
        cartesian_pose.z = pose[5]
        
        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        self.base.ExecuteAction(action)

        TIMEOUT_DURATION = 20  # seconds
        finished = e.wait(TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)
    
    def send_twist_command(self, cmd_twist):
        """
        Convienience method for sending an end-effector twist command
        to the robot. 
        """
        command = Base_pb2.TwistCommand()
        command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        command.duration = 0

        twist = command.twist
        twist.angular_x = np.degrees(cmd_twist[0])
        twist.angular_y = np.degrees(cmd_twist[1])
        twist.angular_z = np.degrees(cmd_twist[2])
        twist.linear_x = cmd_twist[3]
        twist.linear_y = cmd_twist[4]
        twist.linear_z = cmd_twist[5]

        # Note: this API call takes about 25ms
        self.base.SendTwistCommand(command)

    def send_wrench_command(self, cmd_wrench):
        """
        Convienience method for sending an end-effector wrench command
        to the robot. 
        
        WARNING: this method is experimental. Force control should probably
        be done with full torque-control over a 1kHz control loop (UDP) in C++. 
        """
        command = Base_pb2.WrenchCommand()
        command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        command.duration = 0

        wrench = command.wrench
        wrench.torque_x = cmd_wrench[0]
        wrench.torque_y = cmd_wrench[1]
        wrench.torque_z = cmd_wrench[2]
        wrench.force_x = cmd_wrench[3]
        wrench.force_y = cmd_wrench[4]
        wrench.force_z = cmd_wrench[5]

        self.base.SendWrenchCommand(command)

    def CalcArmPosition(self, context, output):
        """
        Compute the current joint angles and send as output.
        """
        # Get feedback from the base, but only if we haven't already this timestep
        t = context.get_time()
        if (self.last_feedback_time != t): self.GetFeedback(t)

        q = np.zeros(7)
        for i in range(7):
            q[i] = np.radians(self.feedback.actuators[i].position)  # Kortex provides joint angles
                                                                    # in degrees for some reason
        output.SetFromVector(q)

    def CalcArmVelocity(self, context, output):
        """
        Compute the current joint velocities and send as output.
        """
        t = context.get_time()
        if (self.last_feedback_time != t): self.GetFeedback(t)

        qd = np.zeros(7)
        for i in range(7):
            qd[i] = np.radians(self.feedback.actuators[i].velocity)  # Kortex provides joint angles
                                                                     # in degrees for some reason
        output.SetFromVector(qd)

    def CalcArmTorque(self, context, output):
        """
        Compute the current joint torques and send as output.
        """
        t = context.get_time()
        if (self.last_feedback_time != t): self.GetFeedback(t)

        tau = np.zeros(7)
        for i in range(7):
            tau[i] = np.radians(self.feedback.actuators[i].torque)  # in Nm

        output.SetFromVector(tau)

    def CalcEndEffectorPose(self, context, output):
        """
        Compute the current end-effector pose and send as output.
        """
        t = context.get_time()
        if (self.last_feedback_time != t): self.GetFeedback(t)

        ee_pose = np.zeros(6)
        ee_pose[0] = np.radians(self.feedback.base.tool_pose_theta_x)
        ee_pose[1] = np.radians(self.feedback.base.tool_pose_theta_y)
        ee_pose[2] = np.radians(self.feedback.base.tool_pose_theta_z)
        ee_pose[3] = self.feedback.base.tool_pose_x
        ee_pose[4] = self.feedback.base.tool_pose_y
        ee_pose[5] = self.feedback.base.tool_pose_z

        # Store the end-effector pose so we can use it to compute the camera pose
        self.ee_pose = ee_pose

        output.SetFromVector(ee_pose)

    def CalcEndEffectorTwist(self, context, output):
        """
        Compute the current end-effector twist and send as output
        """
        t = context.get_time()
        if (self.last_feedback_time != t): self.GetFeedback(t)
        
        ee_twist = np.zeros(6)
        ee_twist[0] = np.radians(self.feedback.base.tool_twist_angular_x)
        ee_twist[1] = np.radians(self.feedback.base.tool_twist_angular_y)
        ee_twist[2] = np.radians(self.feedback.base.tool_twist_angular_z)
        ee_twist[3] = self.feedback.base.tool_twist_linear_x
        ee_twist[4] = self.feedback.base.tool_twist_linear_y
        ee_twist[5] = self.feedback.base.tool_twist_linear_z

        output.SetFromVector(ee_twist)

    def CalcEndEffectorWrench(self, context, output):
        """
        Compute the current end-effector wrench and send as output
        """
        t = context.get_time()
        if (self.last_feedback_time != t): self.GetFeedback(t)
        
        ee_wrench = np.zeros(6)
        ee_wrench[0] = self.feedback.base.tool_external_wrench_torque_x
        ee_wrench[1] = self.feedback.base.tool_external_wrench_torque_y
        ee_wrench[2] = self.feedback.base.tool_external_wrench_torque_z
        ee_wrench[3] = self.feedback.base.tool_external_wrench_force_x
        ee_wrench[4] = self.feedback.base.tool_external_wrench_force_y
        ee_wrench[5] = self.feedback.base.tool_external_wrench_force_z

        output.SetFromVector(ee_wrench)

    def CalcGripperPosition(self, context, output):
        """
        Compute the current gripper position and send as output

        Note that this method is fairly slow: sending and recieving the
        MeasuredGripperMovement takes about 25ms.
        """
        # Position is 0 full open, 1 fully closed
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
        
        output.SetFromVector([gripper_measure.finger[0].value])

    def CalcGripperVelocity(self, context, output):
        """
        Compute the current gripper velocity and send as output.

        Note that this method is fairly slow: sending and recieving the
        MeasuredGripperMovement takes about 25ms.
        """
        # TODO: this just gives us a speed, but not a direction!
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_SPEED
        gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)

        output.SetFromVector([gripper_measure.finger[0].value])

    def CaptureRgbImage(self, context, output):
        """
        Capture and send as output a color image from the camera.
        """
        # Read data vida OpenCV
        ret, frame = self.color_stream.read()
        if not ret:
            raise RuntimeError("Unable to pull color image")

        # Resize to match resolution of the depth image, and add an alpha channel
        frame = cv2.resize(frame, (480, 270))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        # Transform to match depth image (roughly)
        #tx = 0
        #ty = 0
        #M = np.array([[1, 0, tx],   # transformation matrix
        #              [0, 1, ty]])

        #rows, cols, _ = frame.shape
        #frame = cv2.warpAffine(frame, M, (rows,cols))

        #with open('color_image_saved.npy', 'wb') as f:
        #    np.save(f, frame)

        color_image = output.get_mutable_value()   # 270 x 480 x 4
        color_image.mutable_data[:,:,:] = frame
    
    def CaptureDepthImage(self, context, output):
        """
        Capture and send as output a depth image from the camera.
        """
        # Get raw byte stream from the camera
        timeout = 2.0  # max seconds to wait for an image
        sample = self.depth_video_sink.emit('try-pull-sample', timeout)

        if sample is None:
            raise RuntimeError("Unable to pull depth image: %s second timeout exceeded" % timeout)

        # Convert to a 16bit image (consistent with OpenCV, for example)
        buf = sample.get_buffer()
        caps = sample.get_caps()

        frame = np.ndarray(
                (
                    caps.get_structure(0).get_value('height'),
                    caps.get_structure(0).get_value('width'),
                    1
                ),
                buffer=buf.extract_dup(0, buf.get_size()), dtype=np.uint16)
        
        #with open('depth_image_saved.npy', 'wb') as f:
        #    np.save(f, frame)

        # Convert to a Drake DepthImage
        depth_image = output.get_mutable_value()
        depth_image.mutable_data[:,:,:] = frame

    def CalcCameraTransform(self, context, output):
        """
        Compute and send as output the current pose of the camera in the world.
        """
        ee_rpy = self.ee_pose[:3]
        ee_xyz = self.ee_pose[3:]

        # pose of end-effector in the world frame
        X_WE = RigidTransform(
                        RotationMatrix(RollPitchYaw(ee_rpy)),
                        ee_xyz)

        # pose of camera in the end-effector frame
        # TODO: get precise transform data from the datasheet
        X_EC = RigidTransform(
                        RotationMatrix(RollPitchYaw([0,0,np.pi])),  # This seems strange...
                        [0.03,0.065,-0.14])

        # Compute pose of camera in the world frame
        X_WC = X_WE.multiply(X_EC)

        output.set_value(X_WC)
    
    def DoCalcTimeDerivatives(self, context, continuous_state):
        """
        This method gets called every timestep. Its nominal purpose
        is to update the (dummy) continuous variable for the simulator, 
        but here we'll use it to parse inputs and send the corresponding 
        commands to the robot. 
        """
        print("time is: %s" % context.get_time())

        # Get data from input ports
        ee_target_type = self.ee_target_type_port.Eval(context)
        ee_target = self.ee_target_port.Eval(context)

        gripper_target_type = self.gripper_target_type_port.Eval(context)
        gripper_target = self.gripper_target_port.Eval(context)

        # Send commands to the base consistent with these targets
        if gripper_target_type == GripperTarget.kPosition:
            self.send_gripper_position_command(gripper_target[0])
        elif gripper_target_type == GripperTarget.kVelocity:
            self.send_gripper_velocity_command(gripper_target[0])
        else:
            raise RuntimeError("Invalid gripper target type %s" % gripper_target_type)

        if ee_target_type == EndEffectorTarget.kPose:
            # WARNING: unlike the twist and wrench commands, this command
            # stops everything and just moves the end-effector to the 
            # desired pose.
            self.send_pose_command(ee_target)

        elif ee_target_type == EndEffectorTarget.kTwist:
            self.send_twist_command(ee_target)

        elif ee_target_type == EndEffectorTarget.kWrench:
            # WARNING: this method is experimental. Full torque control via a 1kHz
            # UDP connection and C code is probably preferable for force control.
            self.send_wrench_command(ee_target)
        else:
            raise RuntimeError("Invalid end-effector target type %s" % ee_target_type)


