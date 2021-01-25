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
    def __init__(self, time_step, gripper="hande", method="standard", estimator="bayes", batch_size=np.inf):
        """
        Creates an instance of this estimation block with the above input and output ports. 

        Parameters: 
            time_step  :   the timestep for the simulation, used for estimating derivatives
            grippper   :   "hande" or "2f_85", the gripper to use in our internal system model
            method     :   The method we'll use for setting up the regression. The 'standard' 
                           method is based on
                           
                                M*qdd + C*qd + tau_g = Y(q,qd,qdd)*theta = tau,
                           
                           (i.e., the system dynamics are linear in the inertial parameters),
                           while the 'energy' method is based on
                            
                                Hdot = A*theta + b = qd'*tau,
                            
                           (i.e., the system energy dot is linear in the inertial parameters.)
            estimator  :   The method we'll use for estimation. Must be one of:

                            - 'LSE' - standard least-squares

                            - 'bayes' - iterative Bayesian estimation based on NormalInverseGamma prior

                            - 'full_bayes' - full Bayesian linear regression.

            batch_size :   Number of data points to consider for regression. Only relevant to
                           the 'LSE'and 'full_bayes' methods. 
        """
        LeafSystem.__init__(self)
        self.dt = time_step

        assert method == "standard" or method == "energy", "Invalid method %s" % method
        assert estimator == "LSE" or estimator == "bayes" or estimator == "full_bayes", "Invalid estimator %s" % estimator
        assert gripper=="hande", "2F-85 gripper not implemented yet"

        self.method = method
        self.estimator = estimator
        self.batch_size = batch_size

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

        # Prior parameters for iterative Bayes
        self.mu0 = np.array([0.0])          # mean and precision corresponding to uniform
        self.Lambda0 = np.array([[0.0]])    # prior over parameters theta

        self.a0 = 1         # shape and scale corresponding to uniform prior over
        self.b0 = 0         # log(measurment noise std deviation) [ log(sigma) ]


        # Store covariance
        self.cov = [0.0]

        # Store applied joint torques and measured joint velocities from the last timestep
        self.qd_last = np.zeros(7)
        self.tau_last = np.zeros(7)

        self.H_last = 0

    def CreateSymbolicPlant(self, gripper):
        """
        Creates a symbolic model of the plant with unknown variables theta
        parameterizing an object that we're grasping. 

        Returns the symbolic plant, a corresponding context with symbolic variables theta,
        and these variables. 
        """
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

        peg_sym = plant_sym.GetBodyByName("base_link", peg)

        m = Variable("m")

        c = peg_sym.default_com()
        #c = MakeVectorVariable(3,"c")

        Ibar = peg_sym.default_unit_inertia()
        Ibar = UnitInertia_[Expression](1.17e-5,1.9e-5,1.9e-5)
        #Ixx = Variable("Ixx")
        #Iyy = Variable("Iyy")
        #Izz = Variable("Izz")
        #Ixy = Variable("Ixy")
        #Ixz = Variable("Ixz")
        #Iyz = Variable("Iyz")
        #Ibar = UnitInertia_[Expression](Ixx,Iyy,Izz,Ixy,Ixz,Iyz)

        I = SpatialInertia_[Expression](
                m, 
                c,
                Ibar )

        print(I)

        peg_sym.SetSpatialInertiaInBodyFrame(context_sym, I)

        #theta = np.hstack([m,c,Ixx,Iyy,Izz,Ixy,Ixz,Iyz])
        theta = np.hstack([m])

        return plant_sym, context_sym, theta

    def DoFullBayesianInference(self, X, y, n):
        """
        Perform Bayesian linear regression to estimate theta, where

            y = X*theta + epsilon,
            epsilon ~ N(0, sigma^2)

        and (y, X) include all the availible data.

        Assumes a uniform prior on (theta, log(sigma)).
        (See Gelman BDA3, chaper 14.2)
        """
        k = 1   # number of parameters

        XTX_inv = np.linalg.inv(X.T@X)
        theta_hat = XTX_inv@X.T@y      # Least squares estimate (mean)
        V_theta = XTX_inv              # Covariance excluding measurement noise

        # Posterior distribution of measurement noise variance (scaled inverse chi-squared)
        nu = n-k
        if nu == 0:
            s_squared = 1e12   # a very big number
        else:
            s_squared = (1/nu)*(y-X@theta_hat).T@(y-X@theta_hat)

        # Maximum likelihood estimate of measurement noise variance
        sigma_hat_squared = nu*s_squared / (nu + 2)

        # Covariance inlcuding measurement noise
        self.cov = sigma_hat_squared * V_theta

        return theta_hat

    def DoIterativeBayesianInference(self, X, y):
        """
        Perform Bayesian linear regression to estimate theta, where

            y = X*theta + epsilon,
            epsilon ~ N(0, sigma^2)

        and (y, X) are just the data from the latest timestep. 

        Updates a normal inverse-gamma prior as per
        https://en.wikipedia.org/wiki/Bayesian_linear_regression#Posterior_distribution
        """

        LambdaN = X.T@X + self.Lambda0                                  # precision
        muN = np.linalg.inv(LambdaN)@( self.Lambda0@self.mu0 + X.T@y )  # mean

        n = 1   # number of data points
        aN = self.a0 + n/2                                                                 # shape
        bN = self.b0 + 0.5*(y.T@y + self.mu0.T@self.Lambda0@self.mu0 - muN.T@LambdaN@muN)  # scale

        # MAP estiamte of overall covariance
        sigma_squared_map = bN / (aN + 1)
        self.cov = sigma_squared_map*np.linalg.inv(LambdaN)

        # Update the priors
        self.Lambda0 = LambdaN
        self.mu0 = muN
        self.a0 = aN
        self.b0 = bN

        return muN

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

        # Set up the linear regression y = X*theta
        if self.method == "energy":
            
            # Compute Hamiltonian (overall system energy)
            U = self.plant.CalcPotentialEnergy(self.context)
            K = self.plant.CalcKineticEnergy(self.context)
            H = K + U   

            # Compute time derivative of hamiltonian
            Hdot = (H - self.H_last)/self.dt

            # qd'*tau = Hdot  is linear in theta
            A, b = DecomposeAffineExpressions(np.array([Hdot]), self.theta)

            X = A
            y = qd.T@tau - b
        
            # Save Hamiltonian for computing Hdot
            self.H_last = H
        
        else:  # standard method

            # Estimate acceleration numerically
            qdd = (qd - self.qd_last)/self.dt

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

            X = A
            y = self.tau_last - b

        # Ingore first timestep since derivative estimates will be way off
        if context.get_time() > 0:

            # Store linear regression data (not needed for iterative Bayes)
            self.xs.append(X)
            self.ys.append(y)

            # Least-squares estimate
            if self.estimator == "LSE":
                m_hat = self.DoLeastSquares(np.vstack(self.xs), np.hstack(self.ys))

            # Full Bayesian estimate
            elif self.estimator == "full_bayes":
                n = len(self.xs)  # number of data points
                m_hat = self.DoFullBayesianInference(np.vstack(self.xs), np.hstack(self.ys), n)

            # Iterative Bayesian estimate
            elif self.estimator == "bayes":
                m_hat = self.DoIterativeBayesianInference(X, y)

        else:
            m_hat = 0
    
        # Save data for computing derivatives
        self.tau_last = tau
        self.qd_last = qd

        # Get rid of old data
        if len(self.xs) > self.batch_size:
            self.xs.pop(0)
            self.ys.pop(0)
        
        # send output
        output.SetFromVector([m_hat])

