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
    def __init__(self, plant, time_step):
        LeafSystem.__init__(self)

        self.dt = time_step

        # Store an internal model of the full arm + gripper mass
        self.plant = plant
        self.context = self.plant.CreateDefaultContext()
        self.ee_frame = self.plant.GetFrameByName("end_effector_link")

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

        # Store last joint velocities for computing accelerations
        self.qd_last = np.zeros(7)

        # Store regression coefficients
        self.As = []
        self.bs = []

        # Amount of data to store at any given time
        self.batch_size = np.inf

        # Store covariance
        self.cov = 0.0

    def DoBayesianInference(self, X, y, n):
        """
        Perform Bayesian linear regression to estimate theta, where

            y = X*theta + epsilon

        """
        k = 1   # number of parameters

        theta_hat = 1/(X.T@X) * X.T@y  # Least squares estimate (mean)
        SigmaN = 1/(X.T@X)               # Covariance excluding measurement noise

        # Posterior distribution of measurement noise variance (scaled inverse chi-squared)
        nu = n-k
        s_squared = 1/nu*(y-X*theta_hat).T@(y-X*theta_hat)

        # Maximum likelihood estimate of measurement noise variance
        sigma_hat_squared = nu*s_squared / (nu + 2)

        # Covariance inlcuding measurement noise
        self.cov = sigma_hat_squared * SigmaN


        return theta_hat

    def SendParameterCovariance(self, context, output):
        """
        Send the current covariance of the parameter estimate as output.
        """
        output.SetFromVector([self.cov])

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

        # Estimate acceleration numerically
        qdd = (qd - self.qd_last)/self.dt
        self.qd_last = qd

        # Compute end-effector jacobian
        self.plant.SetPositions(self.context, q)
        self.plant.SetVelocities(self.context, qd)

        object_position_in_ee_frame = np.array([0,0.0,0.0])
        J_ee = self.plant.CalcJacobianSpatialVelocity(self.context,
                                                      JacobianWrtVariable.kV,
                                                      self.ee_frame,
                                                      object_position_in_ee_frame,
                                                      self.plant.world_frame(),
                                                      self.plant.world_frame())
        Jdqd_ee = self.plant.CalcBiasSpatialAcceleration(self.context,
                                                         JacobianWrtVariable.kV,
                                                         self.ee_frame,
                                                         object_position_in_ee_frame,
                                                         self.plant.world_frame(),
                                                         self.plant.world_frame()).get_coeffs()


        ee_position = self.plant.CalcPointsPositions(self.context,
                                                     self.ee_frame,
                                                     np.zeros(3),
                                                     self.plant.world_frame())
        object_position = self.plant.CalcPointsPositions(self.context,
                                                         self.ee_frame,
                                                         object_position_in_ee_frame,
                                                         self.plant.world_frame())

        J_ee = J_ee[5,:]  # just consider velocity in z direction
        Jdqd_ee = Jdqd_ee[5]

        # Compute torque due to gravity
        tau_g = -self.plant.CalcGravityGeneralizedForces(self.context)

        # Default estimate 
        m_hat = 0
        
        # Wait until we have a hold on the object to do any estimation
        if t >= 9:
            # Compute regression matrix y(q,qd,qdd)
            a_ee = J_ee@qdd + Jdqd_ee   # task-space acceleration
            y = a_ee + 9.81

            # Compute and store regression coefficients
            A = J_ee.T*y     # A*theta = b
            b = tau - tau_g

            # Simple point estimate based on this singlular data point
            m_hat = 1/(A.T@A) * A.T@b


            ## Full Bayesian inference (TODO: once point estimate is reasonable
            ##                                while end-effector is moving)
            #self.As.append(A)
            #self.bs.append(b)

            #if len(self.As) > 2:  # avoid singularity
            #    m_hat = self.DoBayesianInference(np.hstack(self.As), 
            #                                     np.hstack(self.bs),
            #                                     len(self.As))

        # Get rid of old data
        if len(self.As) > self.batch_size:
            self.As.pop(0)
            self.bs.pop(0)

        output.SetFromVector([m_hat])
