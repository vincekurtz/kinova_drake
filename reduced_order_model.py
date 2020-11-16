# Describes the dynamics of a double integrator reduced-order model

from pydrake.all import *
import numpy as np

class ReducedOrderModelPlant(LeafSystem):
    """
    A system representing the reduced order model double integrator dynamics. 
    Input is a desired acceleration pdd_des. Outputs are the position p_des and velocity
    pd_des of the reduced-order model.
    """
    def __init__(self, task_space_size, ee_frame_id):
        LeafSystem.__init__(self)

        self.n = task_space_size    # number of variables in the task space, i.e., size 
                                    # of p_des. For example, this is 3 if the task-space is CoM position.

        self.DeclareContinuousState(2*self.n)     # state variable is [p_des,pd_des]
        self.frame_id = ee_frame_id

        # Dynamics xdot = A*x + B*u
        self.A = np.block([[np.zeros((self.n,self.n)),np.eye(self.n)],
                           [np.zeros((self.n,2*self.n))             ]])
        self.B = np.block([[np.zeros((self.n,self.n))],
                           [np.eye(self.n)           ]])

        # Output is state x = [p_des, pd_des]
        self.DeclareVectorOutputPort("x", BasicVector(2*self.n), 
                                    self.CopyStateOut,
                                    {self.all_state_ticket()})

        # Geometry output port for visualization
        fpv = FramePoseVector()
        fpv.set_value(self.frame_id, RigidTransform())

        self.DeclareAbstractOutputPort(
                "ee_geometry",
                lambda: AbstractValue.Make(fpv),
                self.SetGeometryOutputs)

        # Input is u = pdd_des
        self.DeclareVectorInputPort("u", BasicVector(self.n))

    def DoCalcTimeDerivatives(self, context, derivatives):
        """
        Set the equations of motion xdot = f(x,u)
        """
        u = self.EvalVectorInput(context,0).get_value()
        x = context.get_continuous_state_vector().get_value()

        xdot = self.A@x + self.B@u

        derivatives.get_mutable_vector().SetFromVector(xdot)

    def CopyStateOut(self, context, output):
        state = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(state)

    def SetGeometryOutputs(self, context, output):
        fpv = output.get_mutable_value()

        x = context.get_continuous_state_vector().get_value()
        rpy = x[0:3]
        pos = x[3:6]
        
        X = RigidTransform()
        X.set_rotation(RollPitchYaw(rpy))
        X.set_translation(pos)

        fpv.set_value(self.frame_id, X)

