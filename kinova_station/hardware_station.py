from pydrake.all import *

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

        print(device_config.GetDeviceType())
        print(self.base.GetArmState())

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Disconnect from the API instance and close everything down. This
        is called at the end of a 'with' statement.
        """
        print("Closing Hardware Connection...")

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

