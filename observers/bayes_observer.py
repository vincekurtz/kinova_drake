##
#
# An observer which uses Bayesian inference to estimate object inertial parameters.
#
##

from pydrake.all import *

class BayesObserver(LeafSystem):
    """
    An observer which estimates the inertial parameters of a grasped object using
    Bayesian inference. 

                            ---------------------------------
                            |                               |
                            |                               |
    joint_positions ------> |                               |
                            |         BayesObserver         |
    joint_velocities -----> |                               | ------> manipuland_parameter_estimate
                            |                               |
    joint_torques --------> |                               | ------> manipuland_parameter_covariance
                            |                               |
                            |                               |
                            |                               |
                            ---------------------------------
    
    """
    def __init__(self, time_step, gripper="hande"):
        LeafSystem.__init__(self)

        self.dt = time_step

        # Create an internal model of the plant (arm + gripper + object)
        # with symbolic values for unknown parameters
        self.plant, self.context, self.theta = self.CreateSymbolicPlant(gripper)

        # Declare input ports
        self.q_port = self.DeclareVectorInputPort(
                                    "joint_positions",
                                    BasicVector(7))
        self.qd_port = self.DeclareVectorInputPort(
                                    "joint_velocities",
                                    BasicVector(7))
        self.tau_port = self.DeclareVectorInputPort(
                                    "joint_torques",
                                    BasicVector(7))

        # Declare output ports
        self.DeclareVectorOutputPort(
                "manipuland_parameter_estimate",
                BasicVector(1),
                self.CalcParameterEstimate)
        self.DeclareVectorOutputPort(
                "manipuland_parameter_covariance",
                BasicVector(1),
                self.SendParameterCovariance)

        # Store regression coefficients (x_i = y_i*theta + epsilon)
        self.xs = []
        self.ys = []

        # Amount of data to store at any given time
        self.batch_size = 500

        # Store covariance
        self.cov = [0.0]

        # Store applied joint torques and measured joint velocities from the last timestep
        self.qd_last = np.zeros(7)
        self.tau_last = np.zeros(7)

    def CreateSymbolicPlant(self, gripper):
        """
        Creates a symbolic model of the plant with unknown variables theta
        parameterizing an object that we're grasping. 

        Returns the symbolic plant, a corresponding context with symbolic variables theta,
        and these variables. 
        """
        assert gripper=="hande", "2F-85 gripper not implemented yet"

        plant = MultibodyPlant(1.0)  # timestep not used

        # Add the arm
        arm_urdf = "./models/gen3_7dof/urdf/GEN3_URDF_V12.urdf"
        arm = Parser(plant=plant).AddModelFromFile(arm_urdf, "arm")

        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("base_link", arm))

        # Add the gripper
        gripper_urdf = "./models/hande_gripper/urdf/robotiq_hande_static.urdf"
        gripper = Parser(plant=plant).AddModelFromFile(gripper_urdf, "gripper")

        plant.WeldFrames(plant.GetFrameByName("end_effector_link", arm),
                         plant.GetFrameByName("hande_base_link", gripper))

        # Add the object we're holding
        peg_urdf = "./models/manipulands/peg.sdf"
        peg = Parser(plant=plant).AddModelFromFile(peg_urdf,"peg")

        X_peg = RigidTransform()
        X_peg.set_translation([0,0,0.13])
        X_peg.set_rotation(RotationMatrix(RollPitchYaw([0,0,np.pi/2])))
        plant.WeldFrames(plant.GetFrameByName("end_effector_link",arm),
                         plant.GetFrameByName("base_link", peg), X_peg)

        # Make unknown parameters symbolic
        plant.Finalize()
        plant_sym = plant.ToSymbolic()
        context_sym = plant_sym.CreateDefaultContext()

        m = Variable("m")
        #m = 0.028    # DEBUG: set to true mass
        peg_sym = plant_sym.GetBodyByName("base_link", peg)
        peg_sym.SetMass(context_sym, m)     # see also: SetSpatialInertiaInBodyFrame

        return plant_sym, context_sym, np.array([m])

    def DoBayesianInference(self, X, y, n):
        """
        Perform Bayesian linear regression to estimate theta, where

            y = X*theta + epsilon,
            epsilon ~ N(0, sigma^2)

        Assumes a uniform prior on (theta, log(sigma)).
        (See Gelman BDA3, chaper 14.2)
        """
        k = 1   # number of parameters

        XTX_inv = np.linalg.inv(X.T@X)
        theta_hat = XTX_inv@X.T@y      # Least squares estimate (mean)
        V_theta = XTX_inv              # Covariance excluding measurement noise

        # Posterior distribution of measurement noise variance (scaled inverse chi-squared)
        nu = n-k
        s_squared = 1/nu*(y-X@theta_hat).T@(y-X@theta_hat)

        # Maximum likelihood estimate of measurement noise variance
        sigma_hat_squared = nu*s_squared / (nu + 2)

        # Covariance inlcuding measurement noise
        self.cov = sigma_hat_squared * V_theta

        return theta_hat

    def DoLeastSquares(self, X, y):
        """
        Return a least-squares estimate of theta, where

            y = X*theta

        """
        theta_hat = np.linalg.inv(X.T@X)@X.T@y

        return theta_hat

    def SendParameterCovariance(self, context, output):
        """
        Send the current covariance of the parameter estimate as output.
        """
        output.SetFromVector(self.cov)

    def CalcParameterEstimate(self, context, output):
        """
        Compute the latest parameter estimates using Bayesian linear regression and
        the fact that the dynamics are linear in the interial parameters:

            Y(q, v, vd)*theta = tau

        """
        # Get data from input ports
        t = context.get_time()
        q = self.q_port.Eval(context)
        qd = self.qd_port.Eval(context)
        tau = self.tau_port.Eval(context)
        
        self.plant.SetPositions(self.context, q)
        self.plant.SetVelocities(self.context, qd)

        # Estimate acceleration numerically
        qdd = (qd - self.qd_last)/self.dt
        self.qd_last = qd

        # Compute generalized forces due to gravity
        tau_g = -self.plant.CalcGravityGeneralizedForces(self.context)

        # Get a symbolic expression for torques required to achieve the
        # current acceleration
        f_ext = MultibodyForces_[Expression](self.plant)
        f_ext.SetZero()
        tau_sym = self.plant.CalcInverseDynamics(self.context, qdd, f_ext) + tau_g

        # Write this expression for torques as linear in the parameters theta
        # TODO: this is slow. Any way to speed up?
        A, b = DecomposeAffineExpressions(tau_sym, self.theta)

        # Store linear regression data
        #
        #   tau_sym = A*theta + b = tau_last
        #
        # (i.e. torques computed with inverse dynamics over our symbolic model, tau_sym,
        #  should match the torques that were actually applied, tau_last)
        self.xs.append(A)                  # y_i = x_i'*theta + epsilon
        self.ys.append(self.tau_last - b)

        if context.get_time() > 0.0:

            ## Simple point estimate
            #m_hat = np.linalg.inv(A.T@A)@A.T@(self.tau_last-b)

            # Least-squares estimate
            #m_hat = self.DoLeastSquares(np.vstack(self.xs), np.hstack(self.ys))

            # Bayesian estimate
            n = len(self.xs)  # number of data points
            m_hat = self.DoBayesianInference(np.vstack(self.xs), np.hstack(self.ys), n)
        else:
            # Ingore the first timestep, since we don't have tau_last for that step
            m_hat = 0

        # Get rid of old data
        if len(self.xs) > self.batch_size:
            self.xs.pop(0)
            self.ys.pop(0)

        # send output
        output.SetFromVector([m_hat])

        # Save torques applied at this timestep
        self.tau_last = tau
