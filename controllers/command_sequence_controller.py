
from controllers.basic_controller import *
from controllers.command_sequence import *

class CommandSequenceController(BasicController):
    """
    A simple controller which tracks a sequence of target end-effector poses
    and gripper states (open or closed). See BasicController for IO details. 

    Sends exclusively gripper position commands and end-effector twist commands.
    """
    def __init__(self, command_sequence):
        BasicController.__init__(self)
        self.cs = command_sequence   # a CommandSequence object

    def CalcGripperCommand(self, context, output):
        t = context.get_time()

        if self.cs.gripper_closed(t):
            cmd_pos = np.array([1.0])  # closed 
        else:
            cmd_pos = np.array([0.0])  # open

        output.SetFromVector(cmd_pos)

    def CalcEndEffectorCommand(self, context, output):
        """
        Compute and send an end-effector twist command. This is just a
        simple PD controller.
        """
        t = context.get_time()

        # Get target end-effector pose and twist
        target_pose = self.cs.target_pose(t)
        target_twist = np.zeros(6)

        # Get current end-effector pose and twist
        current_pose = self.ee_pose_port.Eval(context)
        current_twist = self.ee_twist_port.Eval(context)

        # Set commanded end-effector twist using a PD controller
        Kp = np.diag([100,100,100, 200,200,200])
        Kd = 2*np.sqrt(0.5*Kp)
        
        cmd_twist = Kp@(target_pose - current_pose) + Kd@(target_twist - current_twist)

        output.SetFromVector(cmd_twist)

    def ConnectToStation(self, builder, station):
        """
        Connect inputs and outputs of this controller to the given kinova station (either
        hardware or simulation). 
        """
        builder.Connect(                                  # Send commands to the station
                self.GetOutputPort("ee_command"),
                station.GetInputPort("ee_target"))
        builder.Connect(
                self.GetOutputPort("ee_command_type"),
                station.GetInputPort("ee_target_type"))
        builder.Connect(
                self.GetOutputPort("gripper_command"),
                station.GetInputPort("gripper_target"))
        builder.Connect(
                self.GetOutputPort("gripper_command_type"),
                station.GetInputPort("gripper_target_type"))

        builder.Connect(                                     # Send state information
                station.GetOutputPort("measured_ee_pose"),   # to the controller
                self.GetInputPort("ee_pose"))
        builder.Connect(
                station.GetOutputPort("measured_ee_twist"),
                self.GetInputPort("ee_twist"))





