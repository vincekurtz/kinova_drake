##
#
# A simple controller which sends commands to a KinovaStation to pick up and
# move around a peg in a pre-ordained location.
#
##

from pydrake.all import *
from kinova_station import EndEffectorTarget, GripperTarget

class PegPickupController(LeafSystem):
    """
    A simple controller which sends commands to a KinovaStation to pick up and
    move around a peg in a pre-ordained location.

                            -------------------------
                            |                       |
                            |                       |
    ee_pose --------------> |                       | ---> ee_command (desired twist)
    ee_twist -------------> |                       | ---> ee_command_type
                            |  PegPickupController  |
                            |                       |
    gripper_position -----> |                       | ---> gripper_command
    gripper_velocity -----> |                       | ---> gripper_command_type
                            |                       |
                            |                       |
                            |                       |
                            -------------------------

    """
    def __init__(self):
        LeafSystem.__init__(self)

        # Declare input ports (current robot state)
        self.ee_pose_port = self.DeclareVectorInputPort(
                                    "ee_pose",
                                    BasicVector(6))
        self.ee_twist_port = self.DeclareVectorInputPort(
                                    "ee_twist",
                                    BasicVector(6))
        self.gripper_pos_port = self.DeclareVectorInputPort(
                                        "gripper_position",
                                        BasicVector(2))
        self.gripper_vel_port = self.DeclareVectorInputPort(
                                        "gripper_velocity",
                                        BasicVector(2))

        # Declare output ports (desired end-effector and gripper behavior)
        self.DeclareVectorOutputPort(
                "ee_command",
                BasicVector(6),
                self.CalcEndEffectorCommand)
        self.DeclareAbstractOutputPort(
                "ee_command_type",
                lambda: AbstractValue.Make(EndEffectorTarget.kTwist),
                self.SetEndEffectorCommandType)
        self.DeclareVectorOutputPort(
                "gripper_command",
                BasicVector(2),
                self.CalcGripperCommand)
        self.DeclareAbstractOutputPort(
                "gripper_command_type",
                lambda: AbstractValue.Make(GripperTarget.kPosition),
                self.SetGripperCommandType)

        # Define command sequence
        # TODO: consider creating a separate command sequence class
        self.command_sequence = {
                "lineup" : {"target_pose" : np.array([-np.pi,0,0, 0.5,-0.1,0.3]),
                              "start_time" : 0,
                              "end_time" : 5,
                              "gripper_closed" : False },
                "pregrasp" : {"target_pose" : np.array([-np.pi/2,0,0, 0.5,-0.12,0.1]),
                               "start_time" : 5,
                               "end_time" : 8,
                               "gripper_closed" : False },
                "grasp" : {"target_pose" : np.array([-np.pi/2,0,0, 0.5,-0.12,0.5]),
                               "start_time" : 8,
                               "end_time" : np.inf,
                               "gripper_closed" : True },
                #"move_1" : {"target_pose" : np.array([-np.pi/2,0,0, 0.5,-0.12,0.7]),
                #               "start_time" : 10,
                #               "end_time" : 12,
                #               "gripper_closed" : True },
                #"move_2" : {"target_pose" : np.array([-np.pi/2,0,0, 0.5,-0.12,0.5]),
                #               "start_time" : 12,
                #               "end_time" : 14,
                #               "gripper_closed" : True },
                #"move_3" : {"target_pose" : np.array([-np.pi/2,0,0, 0.5,-0.12,0.7]),
                #               "start_time" : 14,
                #               "end_time" : 16,
                #               "gripper_closed" : True },
                #"move_4" : {"target_pose" : np.array([-np.pi/2,0,0, 0.5,-0.12,0.5]),
                #               "start_time" : 16,
                #               "end_time" : np.inf,
                #               "gripper_closed" : True },
                #"move" : {"target_pose" : np.array([-np.pi/2,0,np.pi/2, 0.0,0.5,0.5]),
                #               "start_time" : 10,
                #               "end_time" : 15,
                #               "gripper_closed" : True },
                #"move_back" : {"target_pose" : np.array([-np.pi/2,0,0, 0.5,0.0,0.5]),
                #               "start_time" : 15,
                #               "end_time" : np.inf,
                #               "gripper_closed" : True },
                }

    def get_current_mode(self, t):
        """
        Searches over self.command_sequence to find the control mode
        corresponding to the given timestep. 
        """
        current_mode = None

        for mode in self.command_sequence:
            st = self.command_sequence[mode]["start_time"]
            et = self.command_sequence[mode]["end_time"]

            if (st <= t) and (t < et):
                # Found a potentially valid mode
                if current_mode is None:
                    current_mode = mode
                else:
                    raise RuntimeError("Conflicting Modes. Modes '%s' and '%s' both hold at time %s" % (current_mode, mode, t))
        
        if current_mode is None:
            raise RuntimeError("No valid mode found at t=%s" % t)

        return current_mode


    def SetGripperCommandType(self, context, output):
        command_type = GripperTarget.kPosition
        output.SetFrom(AbstractValue.Make(command_type))

    def SetEndEffectorCommandType(self, context, output):
        command_type = EndEffectorTarget.kTwist
        output.SetFrom(AbstractValue.Make(command_type))

    def CalcGripperCommand(self, context, output):
        # Get current mode (including gripper state)
        mode = self.get_current_mode(context.get_time())
       
        # Set gripper target position accordingly
        if self.command_sequence[mode]["gripper_closed"]:
            cmd_pos = np.zeros(2)
        else:
            cmd_pos = np.array([0.04,0.04])

        output.SetFromVector(cmd_pos)

    def CalcEndEffectorCommand(self, context, output):

        # Get target end-effector pose and twist
        mode = self.get_current_mode(context.get_time())
        target_pose = self.command_sequence[mode]["target_pose"]
        target_twist = np.zeros(6)

        # Get current end-effector pose and twist
        current_pose = self.ee_pose_port.Eval(context)
        current_twist = self.ee_twist_port.Eval(context)

        # Set commanded end-effector twist using a PD controller
        Kp = 10*np.eye(6)
        Kd = 2*np.sqrt(Kp)
        
        cmd_twist = Kp@(target_pose - current_pose) + Kd@(target_twist - current_twist)

        output.SetFromVector(cmd_twist)

