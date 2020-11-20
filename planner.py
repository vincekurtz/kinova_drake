# A simple high-level planner which sets desired pose and gripper state.

from pydrake.all import *
import numpy as np

class SimplePlanner(LeafSystem):
    """
    This is a simple system block with no inputs. It simply outpus

        1) A desired end-effector pose [roll;pitch;yaw;x;y;z] (and pose dot)
        2) A desired gripper state (open or closed)
    """
    def __init__(self):
        LeafSystem.__init__(self)

        self.target_pose = np.array([np.pi,  
                                     0.0,
                                     np.pi/2,
                                     0.0,
                                     0.4,
                                     0.15])
        self.target_twist = np.zeros(6)

        self.DeclareVectorOutputPort(
                "end_effector_setpoint",
                BasicVector(12),
                self.SetEndEffectorOutput)

        self.DeclareAbstractOutputPort(
                "gripper_command",
                lambda : AbstractValue.Make(True),
                self.SetGripperOutput)

    def SetEndEffectorOutput(self, context, output):
        if context.get_time() > 3.5:
            self.target_pose += np.array([0,0,0,0,0,0.0001])
        target_state = np.hstack([self.target_pose, self.target_twist])
        output.SetFromVector(target_state)

    def SetGripperOutput(self, context, output):
        gripper_closed = False

        if context.get_time() > 3.5:
            gripper_closed = True

        output.set_value(gripper_closed)



