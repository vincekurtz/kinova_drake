import time
import numpy as np
from pydrake.all import *
from helpers import *

class Gen3Controller(LeafSystem):
    """
    This class describes a controller for a Kinova Gen3 manipulator arm based on
    a double integrator reduced-order model.

                          -----------------------------
                          |                           |
             [q,qd] ----> |                           | ----> tau
                          |      Gen3Controller       |
      [p_des,pd_des] ---> |                           | ----> [p,pd]
                          |                           |
           [pdd_des] ---> |                           |
                          |                           |
                          -----------------------------

    """
    def __init__(self, plant, dt):
        LeafSystem.__init__(self)

        self.dt = dt
        self.plant = plant
        self.context = self.plant.CreateDefaultContext()  # stores q, qd

        self.solver = GurobiSolver()

        # Tuning parameters
        self.Kp = np.block([[  0.0*np.eye(3)  , np.zeros((3,3))],
                            [np.zeros((3,3)),  10.0*np.eye(3)  ]])
        self.Kd = np.block([[  0.0*np.eye(3)  , np.zeros((3,3))],
                            [np.zeros((3,3)),  20.0*np.eye(3)  ]])

        # AutoDiff plant and context for values that require automatic differentiation
        self.plant_autodiff = plant.ToAutoDiffXd()
        self.context_autodiff = self.plant_autodiff.CreateDefaultContext()

        # Robot arm dimensions
        self.arm_model_index = self.plant.GetModelInstanceByName("gen3")
        self.np_arm = self.plant.num_positions(self.arm_model_index)
        self.nv_arm = self.plant.num_velocities(self.arm_model_index)
        self.nu_arm = 7

        # First input port takes in robot arm state ([q;qd])
        self.DeclareVectorInputPort(
                "arm_state",
                BasicVector(self.np_arm + self.nv_arm))
        
        # First output port maps to torques on robot arm
        self.DeclareVectorOutputPort(
                "arm_torques",
                BasicVector(self.nu_arm),
                self.DoCalcArmOutput)

        # Input for RoM state x_rom = [p_des,pd_des]
        self.DeclareVectorInputPort(
                "rom_state",
                BasicVector(12))

        # Input for RoM input u_rom = [pdd_des]
        self.DeclareVectorInputPort(
                "rom_input",
                BasicVector(6))

        # Output end effector state [p_des,pd_des]
        self.DeclareVectorOutputPort(
                "end_effector",
                BasicVector(12),
                self.DoCalcEndEffectorOutput)

        # Output simulation function V([q,qd],[p_des,pd_des])
        self.V = 0
        self.DeclareVectorOutputPort(
                "simulation_function",
                BasicVector(1),
                self.DoCalcSimFcnOutput)

        # Relevant frames
        self.world_frame = self.plant.world_frame()
        self.end_effector_frame = self.plant.GetFrameByName("end_effector_link")

        self.world_frame_autodiff = self.plant_autodiff.world_frame()
        self.end_effector_frame_autodiff = self.plant_autodiff.GetFrameByName("end_effector_link")

    def AddDynamicsConstraint(self, M, qdd, Cqd, tau_g, tau):
        """
        Add a linear dynamics constraint

            M*qdd + Cqd + tau_g = tau

        to the whole-body QP. 
        """
        Aeq = np.hstack([M, np.eye(len(tau))])
        beq = -Cqd - tau_g
        x = np.vstack([qdd, tau])

        return self.mp.AddLinearEqualityConstraint(Aeq, beq, x)
    
    def UpdateStoredContext(self, context):
        """
        Use the data in the given input context to update self.context.
        This should be called at the beginning of each timestep.
        """
        state = self.EvalVectorInput(context, 0).get_value()
        q = state[:self.plant.num_positions()]
        qd = state[-self.plant.num_velocities():]

        self.plant.SetPositions(self.context, q)
        self.plant.SetVelocities(self.context, qd)

    def CalcDynamics(self):
        """
        Compute dynamics quantities, M, Cv, tau_g, and S such that the
        robot's dynamics are given by 

            M(q)vd + C(q,v)v + tau_g = S'u + tau_ext.

        Assumes that self.context has been set properly. 
        """
        M = self.plant.CalcMassMatrixViaInverseDynamics(self.context)
        Cv = self.plant.CalcBiasTerm(self.context)
        tau_g = -self.plant.CalcGravityGeneralizedForces(self.context)
        S = self.plant.MakeActuationMatrix().T

        return M, Cv, tau_g, S

    def CalcCoriolisMatrix(self):
        """
        Compute the coriolis matrix C(q,qd) using autodiff.
        
        Assumes that self.context has been set properly.
        """
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)

        def Cv_fcn(v):
            self.plant_autodiff.SetPositions(self.context_autodiff, q)
            self.plant_autodiff.SetVelocities(self.context_autodiff, v)
            return self.plant_autodiff.CalcBiasTerm(self.context_autodiff)

        C = 0.5*jacobian(Cv_fcn,v)
        return C

    def CalcFramePositionQuantities(self, frame):
        """
        Compute the position (p), jacobian (J) and 
        jacobian-time-derivative-times-v (Jdv) for the given frame
        
        Assumes that self.context has been set properly. 
        """
        p = self.plant.CalcPointsPositions(self.context,
                                           frame,
                                           np.array([0,0,0]),
                                           self.world_frame)
        J = self.plant.CalcJacobianTranslationalVelocity(self.context,
                                                         JacobianWrtVariable.kV,
                                                         frame,
                                                         np.array([0,0,0]),
                                                         self.world_frame,
                                                         self.world_frame)
        Jdv = self.plant.CalcBiasTranslationalAcceleration(self.context,
                                                           JacobianWrtVariable.kV,
                                                           frame,
                                                           np.array([0,0,0]),
                                                           self.world_frame,
                                                           self.world_frame)
        return p, J, Jdv

    def CalcFramePositionJacobianDot(self, frame):
        """
        Compute the time derivative of the given frame's position Jacobian (Jd)
        directly using autodiff. 

        Note that `frame` must be an autodiff type frame. 

        Assumes that self.context has been set properly. 
        """
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)

        def J_fcn(q):
            self.plant_autodiff.SetPositions(self.context_autodiff, q)
            self.plant_autodiff.SetVelocities(self.context_autodiff, v)
            return self.plant_autodiff.CalcJacobianTranslationalVelocity(self.context_autodiff,
                                                                         JacobianWrtVariable.kV,
                                                                         frame,
                                                                         np.zeros(3,),
                                                                         self.world_frame_autodiff,
                                                                         self.world_frame_autodiff)
        Jd = jacobian2(J_fcn,q)@self.plant.MapVelocityToQDot(self.context,v)
        return Jd
    
    def CalcFramePoseJacobianDot(self, frame):
        """
        Compute the time derivative of the given frame's pose Jacobian (Jd)
        directly using autodiff. 

        Note that `frame` must be an autodiff type frame. 

        Assumes that self.context has been set properly. 
        """
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)

        def J_fcn(q):
            self.plant_autodiff.SetPositions(self.context_autodiff, q)
            self.plant_autodiff.SetVelocities(self.context_autodiff, v)
            return self.plant_autodiff.CalcJacobianSpatialVelocity(self.context_autodiff,
                                                                   JacobianWrtVariable.kV,
                                                                   frame,
                                                                   np.zeros(3,),
                                                                   self.world_frame_autodiff,
                                                                   self.world_frame_autodiff)
        Jd = jacobian2(J_fcn,q)@self.plant.MapVelocityToQDot(self.context,v)
        return Jd
    
    def CalcFramePoseQuantities(self, frame):
        """
        Compute the pose (position + orientation), spatial jacobian (J) and,
        spatial jacobian-time-derivative-times-v (Jdv) for the given frame. 
        
        Assumes that self.context has been set properly. 
        """
        pose = self.plant.CalcRelativeTransform(self.context,
                                           self.world_frame,
                                           frame)
        J = self.plant.CalcJacobianSpatialVelocity(self.context,
                                                   JacobianWrtVariable.kV,
                                                   frame,
                                                   np.array([0,0,0]),
                                                   self.world_frame,
                                                   self.world_frame)
        Jdv = self.plant.CalcBiasSpatialAcceleration(self.context,
                                                     JacobianWrtVariable.kV,
                                                     frame,
                                                     np.array([0,0,0]),
                                                     self.world_frame,
                                                     self.world_frame)

        return pose, J, Jdv.get_coeffs()

    def DoCalcEndEffectorOutput(self, context, output):
        """
        This method is called at every timestep, and records
        the current end effector state [p,pd].
        """
        pass
        #self.UpdateStoredContext(context)
        #qd = self.plant.GetVelocities(self.context, self.arm_model_index)

        #p = self.CalcEndEffectorPose()
        #J = self.CalcEndEffectorJacobian()

        #pd = J @ qd

        #output.SetFromVector(np.hstack([p,pd]))

    def DoCalcSimFcnOutput(self, context, output):
        """
        Output the current value of the simulation function.
        """
        output.SetFromVector([self.V])
        
    def DoCalcArmOutput(self, context, output):
        """
        This method is called at every timestep, and determines
        output torques to control the robot arm. 
        """
        self.UpdateStoredContext(context)
        q = self.plant.GetPositions(self.context, self.arm_model_index)
        qd = self.plant.GetVelocities(self.context, self.arm_model_index)

        # Dynamics Computations 
        M, Cqd, tau_g, _ = self.CalcDynamics()
        C = self.CalcCoriolisMatrix()

        # Current end-effector state
        p, J, Jdqd = self.CalcFramePositionQuantities(self.end_effector_frame)
        Jd = self.CalcFramePositionJacobianDot(self.end_effector_frame_autodiff)
        x = p.flatten()
        xd = J@qd

        # Desired end-effector state 
        x_xd_nom = self.EvalVectorInput(context,1).get_value()
        x_nom = x_xd_nom[3:6]
        xd_nom = x_xd_nom[9:]
        xdd_nom = self.EvalVectorInput(context,2).get_value()[3:]

        # End-effector errors
        x_tilde = x - x_nom
        xd_tilde = xd - xd_nom

        # Additional Dynamics Terms
        Minv = np.linalg.inv(M)
        Lambda = np.linalg.inv(J@Minv@J.T)
        Jbar = Minv@J.T@Lambda
        Q = J@Minv@C - Jd
        
        # Desired end-effector force (really wrench)
        Kp = 10*np.eye(3)
        Kd = 1*np.eye(3)
        f_des = Lambda@xdd_nom + Lambda@Q@(qd - Jbar@xd_tilde) + Jbar.T@tau_g - Kp@x_tilde - Kd@xd_tilde

        # Solve QP to find corresponding joint torques
        self.mp = MathematicalProgram()
        tau = self.mp.NewContinuousVariables(self.plant.num_actuators(), 1, 'tau')
        qdd = self.mp.NewContinuousVariables(self.plant.num_velocities(), 1, 'qdd')
       
        # min || qdd - qdd_nom ||^2
        qdd_nom = 100.0*qd
        self.mp.AddQuadraticErrorCost(Q=np.eye(self.plant.num_velocities()),
                                      x_desired=qdd_nom,
                                      vars=qdd)
        
        # min || tau ||^2
        #self.mp.AddQuadraticErrorCost(Q=np.eye(self.plant.num_actuators()),
        #                              x_desired=np.zeros(self.plant.num_actuators()),
        #                              vars=tau)

        # s.t. M*qdd + Cqd + tau_g = tau
        self.AddDynamicsConstraint(M, qdd, Cqd, tau_g, tau)

        # s.t. Jbar'*tau = f_des
        self.mp.AddLinearEqualityConstraint(Aeq=Jbar.T,
                                            beq=f_des,
                                            vars=tau)

        result = self.solver.Solve(self.mp)
        assert result.is_success()
        tau = result.GetSolution(tau)
        
        output.SetFromVector(tau)

