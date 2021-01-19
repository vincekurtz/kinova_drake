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
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

from kortex_api.autogen.messages import DeviceConfig_pb2, Session_pb2, Base_pb2



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
                               |                                  | --> camera_rgb_image (TODO)
                               |                                  | --> camera_depth_image (TODO)
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
                                             BasicVector(6))
        self.gripper_target_type_port = self.DeclareAbstractInputPort(
                                            "gripper_target_type",
                                            AbstractValue.Make(GripperTarget.kPosition))

        # Declare output ports
        self.DeclareVectorOutputPort(
                "test_output_port",
                BasicVector(1),
                self.CalcTestOutput)

        # DEBUG
        self.DeclareDiscreteState(np.zeros(1))

    def DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        # DEBUG
        print("in discrete variable updates")

    def DoCalcTimeDerivatives(self, context, continuous_state):
        # DEBUG
        print("in continuous variable updates")

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

    def send_ee_pose_example(self):
        print("Starting Cartesian action movement ...")
        action = Base_pb2.Action()
        action.name = "Example Cartesian action movement"
        action.application_data = ""

        feedback = self.base_cyclic.RefreshFeedback()

        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = feedback.base.tool_pose_x          # (meters)
        cartesian_pose.y = feedback.base.tool_pose_y - 0.1    # (meters)
        cartesian_pose.z = feedback.base.tool_pose_z - 0.2    # (meters)
        cartesian_pose.theta_x = feedback.base.tool_pose_theta_x # (degrees)
        cartesian_pose.theta_y = feedback.base.tool_pose_theta_y # (degrees)
        cartesian_pose.theta_z = feedback.base.tool_pose_theta_z # (degrees)

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        print("Executing action")
        self.base.ExecuteAction(action)

        print("Waiting for movement to finish ...")
        TIMEOUT_DURATION = 20  # seconds
        finished = e.wait(TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)

        if finished:
            print("Cartesian movement completed")
        else:
            print("Timeout on action notification wait")
        return finished

        pass

    def send_ee_twist_example(self):
        command = Base_pb2.TwistCommand()
        command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE  # is this consistent w/ the simulation?
        command.duration = 0

        twist = command.twist
        twist.linear_x = 0
        twist.linear_y = 0.03
        twist.linear_z = 0
        twist.angular_x = 0
        twist.angular_y = 0
        twist.angular_z = 5

        print ("Sending the twist command for 5 seconds...")
        self.base.SendTwistCommand(command)

        # Let time for twist to be executed
        time.sleep(5)

        print ("Stopping the robot...")
        self.base.Stop()
        time.sleep(1)

    def send_ee_wrench_example(self):
        pass

    def send_gripper_position_target_example(self):
        # Create the GripperCommand we will send
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()

        # Close the gripper with position increments
        print("Performing gripper test in position mode...")
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        position = 0.00
        finger.finger_identifier = 1
        while position < 1.0:
            finger.value = position
            print("Going to position {:0.2f}...".format(finger.value))
            self.base.SendGripperCommand(gripper_command)
            position += 0.1
            time.sleep(1)

    def send_gripper_velocity_target_example(self):
        # Create the GripperCommand we will send
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()

        # Set speed to open gripper
        print ("Opening gripper using speed command...")
        gripper_command.mode = Base_pb2.GRIPPER_SPEED
        finger.value = 0.1
        self.base.SendGripperCommand(gripper_command)
        gripper_request = Base_pb2.GripperRequest()

        # Wait for reported position to be opened
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        while True:
            gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
            if len (gripper_measure.finger):
                print("Current position is : {0}".format(gripper_measure.finger[0].value))
                if gripper_measure.finger[0].value < 0.01:
                    break
            else: # Else, no finger present in answer, end loop
                break

        # Set speed to close gripper
        print ("Closing gripper using speed command...")
        gripper_command.mode = Base_pb2.GRIPPER_SPEED
        finger.value = -0.1
        self.base.SendGripperCommand(gripper_command)

        # Wait for reported speed to be 0
        # Note: this is a nicer way of doing grasping, since we stop once an object is firmly
        # in the gripper. 
        gripper_request.mode = Base_pb2.GRIPPER_SPEED
        while True:
            gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
            if len (gripper_measure.finger):
                print("Current speed is : {0}".format(gripper_measure.finger[0].value))
                if gripper_measure.finger[0].value == 0.0:
                    break
            else: # Else, no finger present in answer, end loop
                break

    def calc_arm_position_example(self):
        feedback = self.base_cyclic.RefreshFeedback()
        q = np.zeros(7)
        for i in range(7):
            q[i] = np.radians(feedback.actuators[i].position)  # Kortex provides joint angles
                                                               # in degrees for some reason
        print("q: %s" % q)

    def calc_arm_velocity_example(self):
        feedback = self.base_cyclic.RefreshFeedback()
        qd = np.zeros(7)
        for i in range(7):
            qd[i] = np.radians(feedback.actuators[i].velocity)  # Kortex provides joint angles
                                                                # in degrees for some reason
        print("qd: %s" % qd)

    def calc_arm_torque_example(self):
        feedback = self.base_cyclic.RefreshFeedback()
        tau = np.zeros(7)
        for i in range(7):
            tau[i] = feedback.actuators[i].torque  # in Nm
                                    
        print("tau: %s" % tau)

    def calc_ee_pose_example(self):
        feedback = self.base_cyclic.RefreshFeedback()
        ee_pose = np.zeros(6)

        ee_pose[0] = np.radians(feedback.base.tool_pose_theta_x)
        ee_pose[1] = np.radians(feedback.base.tool_pose_theta_y)
        ee_pose[2] = np.radians(feedback.base.tool_pose_theta_z)
        ee_pose[3] = feedback.base.tool_pose_x
        ee_pose[4] = feedback.base.tool_pose_y
        ee_pose[5] = feedback.base.tool_pose_z

        print("end-effector pose: %s" % ee_pose)

    def calc_ee_twist_example(self):
        # This is the twist expressed in the base frame
        feedback = self.base_cyclic.RefreshFeedback()
        ee_twist = np.zeros(6)

        ee_twist[0] = np.radians(feedback.base.tool_twist_angular_x)
        ee_twist[1] = np.radians(feedback.base.tool_twist_angular_y)
        ee_twist[2] = np.radians(feedback.base.tool_twist_angular_z)
        ee_twist[3] = feedback.base.tool_twist_linear_x
        ee_twist[4] = feedback.base.tool_twist_linear_y
        ee_twist[5] = feedback.base.tool_twist_linear_z

        print("end-effector twist: %s" % ee_twist)

    def calc_ee_wrench_example(self):
        feedback = self.base_cyclic.RefreshFeedback()

        ee_wrench = np.zeros(6)

        ee_wrench[0] = feedback.base.tool_external_wrench_torque_x
        ee_wrench[1] = feedback.base.tool_external_wrench_torque_y
        ee_wrench[2] = feedback.base.tool_external_wrench_torque_z
        ee_wrench[3] = feedback.base.tool_external_wrench_force_x
        ee_wrench[4] = feedback.base.tool_external_wrench_force_y
        ee_wrench[5] = feedback.base.tool_external_wrench_force_z

        print("end-effector wrench: %s" % ee_wrench)

    def calc_gripper_position_example(self):
        # Position is 0 full open, 1 fully closed
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
        
        if len(gripper_measure.finger):
            print("Current gripper position is : {0}".format(gripper_measure.finger[0].value))
        else: # Else, no finger present in answer, end loop
            print("No gripper detected")
        pass

    def calc_gripper_velocity_example(self):
        # Positive velocities close, negative velocities open
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_SPEED
        gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
        
        if len(gripper_measure.finger):
            print("Current gripper speed is : {0}".format(gripper_measure.finger[0].value))
        else: # Else, no finger present in answer, end loop
            print("No gripper detected")

    def get_camera_rbg_image_example(self):
        # Note: can fetch camera params, see example 01-vision_intrinsics.py
        pass

    def get_camera_depth_image_example(self):
        pass

    def CalcTestOutput(self, context, output):
        print("hello world!")
        print("time is %s" % context.get_time())

        output.SetFromVector([3.14])

