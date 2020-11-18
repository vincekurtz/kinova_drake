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

        self.solver = OsqpSolver()

        # AutoDiff plant and context for values that require automatic differentiation
        self.plant_autodiff = plant.ToAutoDiffXd()
        self.context_autodiff = self.plant_autodiff.CreateDefaultContext()

        self.arm_index = self.plant.GetModelInstanceByName("gen3")

        # Test whether a gripper is attached
        try:
            self.gripper_index = self.plant.GetModelInstanceByName("gripper")
            self.has_gripper = True
        except RuntimeError:
            self.has_gripper = False

        # First input port takes in robot state ([q;qd])
        self.DeclareVectorInputPort(
                "arm_state",
                BasicVector(self.plant.num_positions()+self.plant.num_velocities()))
        
        # First output port maps to torques on robot arm
        self.DeclareVectorOutputPort(
                "arm_torques",
                BasicVector(7),
                self.DoCalcArmOutput)

        # Second output port maps to gripper forces
        self.DeclareVectorOutputPort(
                "gripper_forces",
                BasicVector(2),
                self.DoCalcGripperOutput)

        # Input for RoM state x_rom = [x_des,xd_des]
        self.DeclareVectorInputPort(
                "rom_state",
                BasicVector(12))

        # Input for RoM input u_rom = [xdd_des]
        self.DeclareVectorInputPort(
                "rom_input",
                BasicVector(6))

        # Output end effector state [x,xd]
        self.x = np.zeros(6)
        self.xd = np.zeros(6)
        self.DeclareVectorOutputPort(
                "end_effector",
                BasicVector(12),
                self.DoCalcEndEffectorOutput)

        # Output simulation function V
        self.V = 0
        self.DeclareVectorOutputPort(
                "simulation_function",
                BasicVector(1),
                self.DoCalcSimFcnOutput)
        
        # Output tracking error x_tilde
        self.err = 0
        self.DeclareVectorOutputPort(
                "error",
                BasicVector(1),
                self.DoCalcErrOutput)

        # Relevant frames
        self.world_frame = self.plant.world_frame()
        self.end_effector_frame = self.plant.GetFrameByName("end_effector_link")

        self.world_frame_autodiff = self.plant_autodiff.world_frame()
        self.end_effector_frame_autodiff = self.plant_autodiff.GetFrameByName("end_effector_link")

    def AddDynamicsConstraint(self, M, qdd, Cqd, tau_g, S, tau):
        """
        Add a linear dynamics constraint

            M*qdd + Cqd + tau_g = S.T*tau

        to the whole-body QP. 
        """
        Aeq = np.hstack([M, -S.T])
        beq = -Cqd - tau_g
        x = np.vstack([qdd, tau])

        return self.mp.AddLinearEqualityConstraint(Aeq, beq, x)
        
    def AddEndEffectorForceCost(self, Jbar, tau, f_des, weight=1.0):
        """
        Add a quadratic cost

            w*|| Jbar'*tau - f_des ||^2

        to the whole-body QP
        """
        # We'll write as 1/2 x'*Q*x + c'*x
        Q = Jbar@Jbar.T
        c = -(f_des.T@Jbar.T)[np.newaxis].T

        return self.mp.AddQuadraticCost(Q,c,tau)
    
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
        output.SetFromVector(np.hstack([self.x,self.xd]))

    def DoCalcSimFcnOutput(self, context, output):
        """
        Output the current value of the simulation function.
        """
        output.SetFromVector([self.V])
    
    def DoCalcErrOutput(self, context, output):
        """
        Output the current value of the simulation function.
        """
        output.SetFromVector([self.err])

    def DoCalcGripperOutput(self, context, output):
        """
        This method is called at every timestep, and determines
        output torques to control the gripper.
        """
        pass
        
    def DoCalcArmOutput(self, context, output):
        """
        This method is called at every timestep, and determines
        output torques to control the robot arm. 
        """
        ################## Tuning Parameters #################

        Kp_p = 50
        Kd_p = 20

        Kp_rpy = 1.0
        Kd_rpy = 0.5

        Kd_qd = 1.0

        ######################################################

        self.UpdateStoredContext(context)
        q = self.plant.GetPositions(self.context)
        qd = self.plant.GetVelocities(self.context)

        # Dynamics Computations 
        M, Cqd, tau_g, S = self.CalcDynamics()
        C = self.CalcCoriolisMatrix()

        # Current end-effector state
        X, J, Jdqd = self.CalcFramePoseQuantities(self.end_effector_frame)
        Jd = self.CalcFramePoseJacobianDot(self.end_effector_frame_autodiff)

        p = X.translation()
        pd = (J@qd)[3:]

        R = X.rotation()
        RPY = RollPitchYaw(R)
        rpy = RPY.vector()
        w = (J@qd)[:3]
        rpyd = RPY.CalcRpyDtFromAngularVelocityInParent(w)

        x = np.hstack([rpy, p])
        xd = np.hstack([w, pd])

        # Desired end-effector state 
        rom_state = self.EvalVectorInput(context,1).get_value()
        rom_input = self.EvalVectorInput(context,2).get_value()

        rpy_nom = rom_state[:3]
        p_nom = rom_state[3:6]
        rpyd_nom = rom_state[6:9]
        pd_nom = rom_state[9:]
       
        RPY_nom = RollPitchYaw(rpy_nom)
        R_nom = RotationMatrix(RPY_nom)
        w_nom = RPY_nom.CalcAngularVelocityInParentFromRpyDt(rpyd_nom)

        x_nom = np.hstack([rpy_nom, p_nom])
        xd_nom = np.hstack([w_nom, pd_nom])

        rpydd_nom = rom_input[:3]
        pdd_nom = rom_input[3:]
        wd_nom = RPY_nom.CalcAngularVelocityInParentFromRpyDt(rpydd_nom)
        
        xdd_nom = np.hstack([wd_nom, pdd_nom])

        # End-effector errors
        rpy_tilde = RollPitchYaw(RotationMatrix(RollPitchYaw(rpy-rpy_nom))).vector()  # converting to Rotation matrix and back
                                                                                      # helps avoid strangeness around pi,-pi.
        x_tilde = np.hstack([RPY.CalcAngularVelocityInParentFromRpyDt(rpy_tilde),
                             p - p_nom])
        xd_tilde = xd - xd_nom

        # Additional Dynamics Terms
        Minv = np.linalg.inv(M)
        Lambda = np.linalg.inv(J@Minv@J.T)
        Jbar = Minv@J.T@Lambda
        Q = J@Minv@C - Jd
        
        # Desired end-effector force (really wrench)
        Kp = np.block([[Kp_rpy*np.eye(3), np.zeros((3,3))],
                       [np.zeros((3,3)),  Kp_p*np.eye(3) ]])
        Kd = np.block([[Kd_rpy*np.eye(3), np.zeros((3,3))],
                       [np.zeros((3,3)),  Kd_p*np.eye(3) ]])
        f_des = Lambda@xdd_nom + Lambda@Q@(qd - Jbar@xd_tilde) + Jbar.T@tau_g - Kp@x_tilde - Kd@xd_tilde

        # Solve QP to find corresponding joint torques
        self.mp = MathematicalProgram()
        tau = self.mp.NewContinuousVariables(self.plant.num_actuators(), 1, 'tau')
        qdd = self.mp.NewContinuousVariables(self.plant.num_velocities(), 1, 'qdd')
       
        # min || qdd - qdd_nom ||^2
        qdd_nom = -Kd_qd*qd
        self.mp.AddQuadraticErrorCost(Q=1.0*np.eye(self.plant.num_velocities()),
                                      x_desired=qdd_nom,
                                      vars=qdd)
        
        # min || tau ||^2
        #self.mp.AddQuadraticErrorCost(Q=1e-2*np.eye(self.plant.num_actuators()),
        #                              x_desired=np.zeros(self.plant.num_actuators()),
        #                              vars=tau)

        # min w*|| Jbar'*tau - f_des ||
        #self.AddEndEffectorForceCost(Jbar, tau, f_des, weight=1000.0)

        # s.t. M*qdd + Cqd + tau_g = tau
        self.AddDynamicsConstraint(M, qdd, Cqd, tau_g, S, tau)
        

        # s.t. Jbar'*tau = f_des
        self.mp.AddLinearEqualityConstraint(Aeq=Jbar.T,
                                            beq=f_des,
                                            vars=tau)

        # s.t. tau_min <= tau <= tau_max
        #tau_min = -50
        #tau_max = 50
        #self.mp.AddLinearConstraint(A=np.eye(self.plant.num_actuators()),
        #                            lb=tau_min*np.ones(self.plant.num_actuators()),
        #                            ub=tau_max*np.ones(self.plant.num_actuators()),
        #                            vars=tau)

        result = self.solver.Solve(self.mp)
        assert result.is_success()
        tau = result.GetSolution(tau)
       
        output.SetFromVector(tau)

        # Record stuff for plots
        self.V = 0.5*xd_tilde.T@Lambda@xd_tilde + 0.5*x_tilde.T@Kp@x_tilde
        self.x = x
        self.xd = xd
        self.err = x_tilde.T@x_tilde

        #Vdot = xd_tilde.T@(Jbar.T@tau - Jbar.T@tau_g + Lambda@Q@(Jbar@xd_tilde - qd) - Lambda@xdd_nom + Kp@x_tilde)
        #print(Vdot <= 0)

