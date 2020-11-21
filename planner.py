# A simple high-level planner which sets desired pose and gripper state.

from pydrake.all import *
import numpy as np
import tkinter as tk

class SimplePlanner(LeafSystem):
    """
    This is a simple system block with no inputs. It simply outpus

        1) A desired end-effector pose [roll;pitch;yaw;x;y;z] (and pose dot)
        2) A desired gripper state (open or closed)
    """
    def __init__(self):
        LeafSystem.__init__(self)

        # Set nominal poses and gripper state
        self.pose_nom = np.array([np.pi,  
                                  0.0,
                                  np.pi/2,
                                  0.0,
                                  0.3,
                                  0.55])
        self.twist_nom = np.zeros(6)

        self.gripper_closed = False


        # Declare Drake input and output ports
        self.DeclareVectorOutputPort(
                "end_effector_setpoint",
                BasicVector(12),
                self.SetEndEffectorOutput)

        self.DeclareAbstractOutputPort(
                "gripper_command",
                lambda : AbstractValue.Make(True),
                self.SetGripperOutput)

        # Set up interactive window using Tkinter
        self.window = tk.Tk()
        self.window.title("Planner")

        self.DeclarePeriodicPublish(0.01, 0.0)   # schedule window updates via self.DoPublish

        self.roll = tk.Scale(self.window, 
                     from_=-2*np.pi, 
                     to=2*np.pi,
                     resolution=-1,
                     label="Roll",
                     length=400,
                     orient=tk.HORIZONTAL)
        self.roll.pack()
        self.roll.set(self.pose_nom[0])

        self.pitch = tk.Scale(self.window, 
                     from_=-np.pi/2+0.3,    # restrictive pitch limits to 
                     to=np.pi/2-0.3,        # avoid gimbal lock issues
                     resolution=-1,
                     label="Pitch",
                     length=400,
                     orient=tk.HORIZONTAL)
        self.pitch.pack()
        self.pitch.set(self.pose_nom[1])

        self.yaw = tk.Scale(self.window, 
                     from_=-2*np.pi, 
                     to=2*np.pi,
                     resolution=-1,
                     label="Yaw",
                     length=400,
                     orient=tk.HORIZONTAL)
        self.yaw.pack()
        self.yaw.set(self.pose_nom[2])

        self.x = tk.Scale(self.window, 
                     from_=-0.5, 
                     to=0.5,
                     resolution=-1,
                     label="X",
                     length=400,
                     orient=tk.HORIZONTAL)
        self.x.pack()
        self.x.set(self.pose_nom[3])

        self.y = tk.Scale(self.window, 
                     from_=-0.5, 
                     to=0.5,
                     resolution=-1,
                     label="Y",
                     length=400,
                     orient=tk.HORIZONTAL)
        self.y.pack()
        self.y.set(self.pose_nom[4])

        self.z = tk.Scale(self.window, 
                     from_=0.0, 
                     to=0.7,
                     resolution=-1,
                     label="Z",
                     length=400,
                     orient=tk.HORIZONTAL)
        self.z.pack()
        self.z.set(self.pose_nom[5])

        self.gripper_button = tk.Button(self.window,
                                    text="Toggle Gripper",
                                    state=tk.NORMAL,
                                    command=self.toggle_gripper_state)
        self.gripper_button.pack()

    def toggle_gripper_state(self):
        self.gripper_button.configure(state=tk.NORMAL)
        self.gripper_closed = not self.gripper_closed

    def DoPublish(self, context, output):
        self.window.update_idletasks()
        self.window.update()

    def SetEndEffectorOutput(self, context, output):
        target_state = np.hstack([
            self.roll.get(),
            self.pitch.get(),
            self.yaw.get(),
            self.x.get(),
            self.y.get(),
            self.z.get(),
            self.twist_nom])
        output.SetFromVector(target_state)

    def SetGripperOutput(self, context, output):
        output.set_value(self.gripper_closed)



