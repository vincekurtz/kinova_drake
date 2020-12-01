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

        #self.solver = OsqpSolver()
        self.solver = GurobiSolver()

        # AutoDiff plant and context for values that require automatic differentiation
        self.plant_autodiff = plant.ToAutoDiffXd()
        self.context_autodiff = self.plant_autodiff.CreateDefaultContext()

        self.arm_index = self.plant.GetModelInstanceByName("gen3")
        self.gripper_index = self.plant.GetModelInstanceByName("gripper")

        # Input port for robot state ([q;qd])
        self.arm_state_port = self.DeclareVectorInputPort(
                                      "arm_state",
                                      BasicVector(self.plant.num_positions(self.arm_index)
                                                  +self.plant.num_velocities(self.arm_index)))
        
        # Output port for torques on robot arm
        self.DeclareVectorOutputPort(
                "arm_torques",
                BasicVector(7),
                self.DoCalcArmOutput)

        # Input port for in gripper states
        self.grip_state_port = self.DeclareVectorInputPort(
                                      "gripper_state",
                                      BasicVector(4))

        # Output port for gripper forces
        self.DeclareVectorOutputPort(
                "gripper_forces",
                BasicVector(2),
                self.DoCalcGripperOutput)

        # Input port for RoM state x_rom = [x_des,xd_des]
        self.rom_state_port = self.DeclareVectorInputPort(
                                     "rom_state",
                                     BasicVector(12))

        # Input port for RoM target input u_rom = [xdd_des]
        self.rom_input_port = self.DeclareVectorInputPort(
                                      "rom_input",
                                      BasicVector(6))

        # Output port for resolved RoM input
        self.xdd_nom = np.zeros(6)
        self.DeclareVectorOutputPort(
                "resolved_rom_input",
                BasicVector(6),
                self.SetRomOutput)

        # Input port for gripper command (open or closed)
        self.grip_cmd_port = self.DeclareAbstractInputPort(
                                     "gripper_command",
                                     AbstractValue.Make(True))

        # Output port for end effector state [x,xd] (for logging)
        self.x = np.zeros(6)
        self.xd = np.zeros(6)
        self.DeclareVectorOutputPort(
                "end_effector",
                BasicVector(12),
                self.DoCalcEndEffectorOutput)

        # Output port for storage function V (for logging)
        self.V = 0
        self.DeclareVectorOutputPort(
                "storage_function",
                BasicVector(1),
                self.DoCalcStorageFcnOutput)
        
        # Output port for  tracking error x_tilde (for logging)
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

        # Get joint angle and velocity limits based on the plant, which in
        # turn were loaded from a urdf
        self.GetJointLimits()

        # Parameter estimation stuff
        self.qd_last = np.zeros(self.plant.num_velocities())
        self.tau_last = np.zeros(self.plant.num_actuators())
        self.Ys = []
        self.Fs = []

    def GetJointLimits(self):
        """
        Iterate through self.plant to establish joint angle
        and velocity limits. 

        Sets:

            self.q_min
            self.q_max
            self.qd_min
            self.qd_max

        """
        q_min = []
        q_max = []
        qd_min = []
        qd_max = []

        joint_indices = self.plant.GetJointIndices(self.arm_index)

        for idx in joint_indices:
            joint = self.plant.get_joint(idx)
            
            if joint.type_name() == "revolute":  # ignore the joint welded to the world
                q_min.append(joint.position_lower_limit())
                q_max.append(joint.position_upper_limit())
                qd_min.append(joint.velocity_lower_limit())
                qd_max.append(joint.velocity_upper_limit())

        # Add (nonexistant) joint limits for the gripper
        q_min.extend([-np.inf, -np.inf])
        q_max.extend([np.inf, np.inf])
        qd_min.extend([-np.inf, -np.inf])
        qd_max.extend([np.inf, np.inf])

        self.q_min = np.array(q_min)
        self.q_max = np.array(q_max)
        self.qd_min = np.array(qd_min)
        self.qd_max = np.array(qd_max)
        
    def AddTaskForceConstraint(self, Jbar, tau, Lambda, xdd_nom, Q, qd, 
                                            xd_tilde, tau_g, Kp, x_tilde, Kd):
        """
        Add a linear constraint
        
            Jbar'*tau = f_des

        to the whole-body QP, where 

            f_des = Lambda*xdd_nom + Lambda*Q*(qd - Jbar*xd_tilde) + Jbar.T*tau_g 
                                                            - Kp*x_tilde - Kd*xd_tilde

        are desired task-space forces and both tau and xdd_nom are optimization variables
        """
        # we'll write as A*x = b
        A = np.hstack([Jbar.T, -Lambda])
        x = np.vstack([tau,xdd_nom])
        b = Lambda@Q@(qd-Jbar@xd_tilde) + Jbar.T@tau_g - Kp@x_tilde - Kd@xd_tilde

        return self.mp.AddLinearEqualityConstraint(A,b,x)

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
        
    def AddJointVelCBFConstraint(self, qdd, ah_qd_min, ah_qd_max):
        """
        Add a CBF constraint which enforces joint limits

            qdd >= -ah_qd_min
           -qdd >= -ah_qd_max

        where 
            
            ah_qd_min = alpha(qd - qd_min)
            qh_qd_max = alpha(qd_max - qd)

        for some class-K function alpha.
        """
        A = np.eye(self.plant.num_velocities())
        lb = -ah_qd_min
        ub = ah_qd_max

        return self.mp.AddLinearConstraint(A=A,lb=lb,ub=ub,vars=qdd)
    
    def UpdateStoredContext(self, context):
        """
        Use the data in the given input context to update self.context.
        This should be called at the beginning of each timestep.
        """
        arm_state = self.arm_state_port.Eval(context)
        gripper_state = self.grip_state_port.Eval(context)
      
        q_arm = arm_state[:self.plant.num_positions(self.arm_index)]
        q_gripper = gripper_state[:self.plant.num_positions(self.gripper_index)]

        qd_arm = arm_state[self.plant.num_positions(self.arm_index):]
        qd_gripper = gripper_state[self.plant.num_positions(self.gripper_index):]

        q = np.hstack([q_arm,q_gripper])
        qd = np.hstack([qd_arm,qd_gripper])

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

    def DoCalcStorageFcnOutput(self, context, output):
        """
        Output the current value of the storage function.
        """
        output.SetFromVector([self.V])
    
    def DoCalcErrOutput(self, context, output):
        """
        Output the current value of the simulation function.
        """
        output.SetFromVector([self.err])

    def SetRomOutput(self, context, output):
        output.SetFromVector(self.xdd_nom)

    def DoParameterEstimation(self, f, v, a):
        """
        Given current estimates of applied spatial forces f,
        spatial velocities v, and spatial accelerations a, 
        estimate the (lumped) inertial parameters of the 
        manipuland. 
        """
        y = single_body_regression_matrix(a,v)
        self.Ys.append(y)
        self.Fs.append(f)

        Y = np.vstack(self.Ys)
        F = np.hstack(self.Fs)

        Yinv = np.linalg.inv(Y.T@Y)@Y.T
        theta_hat = Yinv@F

        m_hat = theta_hat[0]     # mass
        mc_hat = theta_hat[1:4]  # mass*(position of CoM in end-effector frame)
        c_hat = mc_hat/m_hat
        print(c_hat)


    def DoCalcGripperOutput(self, context, output):
        """
        This method is called at every timestep, and determines
        output torques to control the gripper.
        """
        # Tuning gains
        Kp = 100
        Kd = 10

        # Get gripper command
        gripper_closed = self.grip_cmd_port.Eval(context)
        if gripper_closed:
            q_nom = np.array([0.024,0.024])
        else:
            q_nom = np.array([-0.0,-0.0])

        v_nom = np.array([0,0])

        # Get current gripper state
        qv = self.grip_state_port.Eval(context)
        q = qv[:2];  v = qv[2:]

        # Set output with pd controller
        f = -Kp*(q-q_nom) - Kd*(v-v_nom)
        output.SetFromVector(f)
        
    def DoCalcArmOutput(self, context, output):
        """
        This method is called at every timestep, and determines
        output torques to control the robot arm. 
        """
        ################## Tuning Parameters #################

        Kp_p = 50      # End effector position stiffness
        Kd_p = 20      # and damping

        Kp_rpy = 1.0   # End effector orientation stiffness
        Kd_rpy = 0.5   # and damping

        Kd_qd = 10.0    # Joint velocity damping

        w_qd = 1e-3    # joint velocity damping weight
        w_xdd = 10     # desired RoM input tracking weight

        alpha_qd = lambda h : 1*h  # CBF class-K functions
        alpha_q = lambda h : 1*h
        beta_q = lambda h : 1*h

        ######################################################
        
        self.UpdateStoredContext(context)
        q = self.plant.GetPositions(self.context)
        qd = self.plant.GetVelocities(self.context)

        # Unpack PD gains 
        Kp = np.block([[Kp_rpy*np.eye(3), np.zeros((3,3))],
                       [np.zeros((3,3)),  Kp_p*np.eye(3) ]])
        Kd = np.block([[Kd_rpy*np.eye(3), np.zeros((3,3))],
                       [np.zeros((3,3)),  Kd_p*np.eye(3) ]])

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
        rom_state = self.rom_state_port.Eval(context)
        rom_input = self.rom_input_port.Eval(context)

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
        
        xdd_target = np.hstack([wd_nom, pdd_nom])

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

        # Solve QP to find joint torques and input to RoM
        self.mp = MathematicalProgram()
        tau = self.mp.NewContinuousVariables(self.plant.num_actuators(), 1, 'tau')
        qdd = self.mp.NewContinuousVariables(self.plant.num_velocities(), 1, 'qdd')
        xdd_nom = self.mp.NewContinuousVariables(6,1,'xdd_nom')
       
        # min w_qd*|| qdd - qdd_nom ||^2
        qdd_nom = -Kd_qd*qd
        self.mp.AddQuadraticErrorCost(Q=w_qd*np.eye(self.plant.num_velocities()),
                                      x_desired=qdd_nom,
                                      vars=qdd)

        # min w_xdd*|| xdd_nom - xdd_target ||^2
        self.mp.AddQuadraticErrorCost(Q=w_xdd*np.eye(6),
                                      x_desired=xdd_target,
                                      vars=xdd_nom)
        
        # min || tau ||^2
        #self.mp.AddQuadraticErrorCost(Q=1e-2*np.eye(self.plant.num_actuators()),
        #                              x_desired=np.zeros(self.plant.num_actuators()),
        #                              vars=tau)

        # min w_f*|| Jbar'*tau - f_des ||
        #self.AddEndEffectorForceCost(Jbar, tau, f_des, weight=w_f)

        # s.t. M*qdd + Cqd + tau_g = tau
        self.AddDynamicsConstraint(M, qdd, Cqd, tau_g, S, tau)
      
        ## s.t. qdd >= -alpha_qd(qd - qd_min)   (joint velocity CBF constraint)
        ##     -qdd >= -alpha_qd(qd_max - qd)
        #ah_qd_min = alpha_qd(qd - self.qd_min)
        #ah_qd_max = alpha_qd(self.qd_max - qd)
        #self.AddJointVelCBFConstraint(qdd, ah_qd_min, ah_qd_max)
        #
        ## s.t. qdd >= -beta_q( qd + alpha_q(q - q_min) )   (joint angle CBF constraint)
        ##     -qdd >= -beta_q( alpha_q(q_max - q) - qd )
        #ah_q_min = beta_q( qd + alpha_q(q - self.q_min) )
        #ah_q_max = beta_q( alpha_q(self.q_max - q) - qd )
        #self.AddJointVelCBFConstraint(qdd, ah_q_min, ah_q_max)
        
        # s.t. Jbar'*tau = f_des, where
        # f_des = Lambda*xdd_nom + Lambda*Q*(qd - Jbar*xd_tilde) + Jbar.T*tau_g 
        #                                                   - Kp*x_tilde - Kd*xd_tilde
        self.AddTaskForceConstraint(Jbar, tau, Lambda, xdd_nom, Q, qd, 
                                       xd_tilde, tau_g, Kp, x_tilde, Kd)

        # s.t. Jbar'*tau = f_des
        #self.mp.AddLinearEqualityConstraint(Aeq=Jbar.T,
        #                                    beq=f_des,
        #                                    vars=tau)

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
        xdd_nom = result.GetSolution(xdd_nom)

        # Convert RoM rotational input from angular acceleration to RPYddt
        # for sending to the RoM
        wd_nom = xdd_nom[:3]
        pdd_nom = xdd_nom[3:]
        rpydd_nom = RPY_nom.CalcRpyDtFromAngularVelocityInParent(wd_nom)
        self.xdd_nom = np.hstack([rpydd_nom,pdd_nom])

        # Set arm torque outputs
        tau_applied = tau[:-2]   # the last two elements have to do with the gripper. We'll ignore those here
                                 # and set them in DoCalcGripperOutput
        output.SetFromVector(tau_applied)

        # Record stuff for plots
        self.V = 0.5*xd_tilde.T@Lambda@xd_tilde + 0.5*x_tilde.T@Kp@x_tilde
        self.x = x
        self.xd = xd
        self.err = x_tilde.T@x_tilde

        # Do some system ID for the grasped object
        qdd = (qd - self.qd_last)/self.dt
        if context.get_time() > 8.5:
            f = Jbar.T@self.tau_last    # spatial force applied at end-effector frame
            v = J@qd          # spatial velocity at end-effector frame
            a = J@qdd + Jdqd  # spatial acceleration at end-effector frame
            
            self.DoParameterEstimation(f,v,a)
        self.qd_last = qd
        self.tau_last = tau

        #Vdot = xd_tilde.T@(Jbar.T@tau - Jbar.T@tau_g + Lambda@Q@(Jbar@xd_tilde - qd) - Lambda@xdd_nom + Kp@x_tilde)
        #print(Vdot <= 0)

