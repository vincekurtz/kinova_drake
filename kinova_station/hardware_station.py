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

from kortex_api.autogen.messages import DeviceConfig_pb2, Session_pb2, Base_pb2



class KinovaStationHardwareInterface(Diagram):
    """
    A template system diagram for controlling a 7 DoF Kinova Gen3 robot, modeled 
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
        Diagram.__init__(self) 
        self.set_name("kinova_manipulation_station_hardware_interface")

        # Set up controller model with robot arm + gripper mass only
        self.arg = "hello"

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

    def Finalize(self):
        pass

        # Finalize the controller plant

        # Set up input and output ports 

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


    def send_twist_example(self):
        
        command = Base_pb2.TwistCommand()
        command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL
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

