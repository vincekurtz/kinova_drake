# Miscellaneous helper functions

from pydrake.all import *
import numpy as np
from enum import Enum

class EndEffectorTargetType(Enum):
    kPose = 1
    kTwist = 2
    kWrench = 3

class GripperTargetType(Enum):
    kPosition = 1
    kVelocity = 2

class EndEffectorWrenchCalculator(LeafSystem):
    """
    A simple system which takes as input joint torques and outputs the corresponding
    wrench applied to the end-effector. 

                       ---------------------------------
                       |                               |
                       |                               |
                       |                               |
    joint_positions -> |  EndEffectorWrenchCalculator  | ---> end_effector_wrench
    joint_angles ----> |                               | 
    joint_torques ---> |                               |
                       |                               |
                       |                               |
                       ---------------------------------
    """
    def __init__(self, plant, end_effector_frame):
        LeafSystem.__init__(self)

        self.plant = plant
        self.context = self.plant.CreateDefaultContext()
        self.ee_frame = end_effector_frame

        # Inputs are joint positions, angles and torques
        self.q_port = self.DeclareVectorInputPort(
                                "joint_positions",
                                BasicVector(plant.num_positions()))
        self.v_port = self.DeclareVectorInputPort(
                                "joint_velocities",
                                BasicVector(plant.num_velocities()))
        self.tau_port = self.DeclareVectorInputPort(
                                "joint_torques",
                                BasicVector(plant.num_actuators()))

        # Output is applied wrench at the end-effector
        self.DeclareVectorOutputPort(
                "end_effector_wrench",
                BasicVector(6),
                self.CalcEndEffectorWrench)

    def CalcEndEffectorWrench(self, context, output):
        # Gather inputs
        q = self.q_port.Eval(context)
        v = self.v_port.Eval(context)
        tau = self.tau_port.Eval(context)

        # Set internal model state
        self.plant.SetPositions(self.context, q)
        self.plant.SetVelocities(self.context, v)

        # Some dynamics computations
        M = self.plant.CalcMassMatrixViaInverseDynamics(self.context)
        tau_g = -self.plant.CalcGravityGeneralizedForces(self.context)

        # Compute end-effector jacobian
        J = self.plant.CalcJacobianSpatialVelocity(self.context,
                                                   JacobianWrtVariable.kV,
                                                   self.ee_frame,
                                                   np.zeros(3),
                                                   self.plant.world_frame(),
                                                   self.plant.world_frame())

        # Compute jacobian pseudoinverse
        Minv = np.linalg.inv(M)
        Lambda = np.linalg.inv(J@Minv@J.T)
        Jbar = Lambda@J@Minv

        # Compute wrench (spatial force) applied at end-effector
        w = Jbar@(tau-tau_g)

        output.SetFromVector(w)
