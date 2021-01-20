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
    ee_pose ------> |                               |
                    |         BayesObserver         |
    ee_twist -----> |                               | ------> manipuland_parameter_estimate
                    |                               |
    ee_wrench ----> |                               |
                    |                               |
                    |                               |
                    |                               |
                    ---------------------------------
    
    """
    def __init__(self, time_step):
        LeafSystem.__init__(self)

        self.dt = time_step

        # Declare input ports
        self.ee_pose_port = self.DeclareVectorInputPort(
                                    "ee_pose",
                                    BasicVector(6))
        self.ee_twist_port = self.DeclareVectorInputPort(
                                    "ee_twist",
                                    BasicVector(6))
        self.ee_wrench_port = self.DeclareVectorInputPort(
                                    "ee_wrench",
                                    BasicVector(6))

        # Declare output port 
        #example_estimate = 10   # TODO: create custom parameter estimate class
        #self.DeclareAbstractOutputPort(
        #        "manipuland_parameter_estimate",
        #        lambda: AbstractValue.Make(example_estimate),
        #        self.CalcParameterEstimate)
        self.DeclareVectorOutputPort(    # need a vector valued port to log
                "manipuland_parameter_estimate",
                BasicVector(1),
                self.CalcParameterEstimate)

        # Store last end-effector velocity for computing accelerations
        self.v_last = np.zeros(6)

        # Store applied torques (really end-effector wrenches) and regression matrices (Y)
        self.Ys = []
        self.taus = []

    def CalcRegressionMatrix(self, q, v, vd):
        """
        Construct the regression matrix Y(q, v, vd) for a single rigid body,
        where the rigid-body dynamics are given by 

            M*vd + C*qd + tau_g = tau 
            Y(q, v, vd)*theta = theta

        where theta = [mass                     ]
                      [center-of-mass position  ]
                      [inertia matrix           ]

        are the (lumped) inertial parameters.
        """
        pass

    def CalcLSE(self, feasibility_constrained=True):
        """
        Perform a least squares estimate of the inertial parameters theta, i.e.,

            min sum( Y_i*theta - tau_i ),

        where regression matrices Y_i and applied torques tau_i are stored in
        self.Ys and self.taus.

        Optionally, include a LMI feasibility constraint.
        """
        Y = np.vstack(self.Ys)
        tau = np.hstack(self.taus)

        # Set up the optimization problem
        mp = MathematicalProgram()
        theta = mp.NewContinuousVariables(10,1,"theta")

        m = theta[0]      # mass
        h = theta[1:4]    # mass * center-of-mass position
        I = np.array([[theta[4,0], theta[7,0], theta[8,0]],   # inertia 
                      [theta[7,0], theta[5,0], theta[9,0]],
                      [theta[8,0], theta[9,0], theta[6,0]]])

        # min || Y*theta - tau ||^2
        Q = Y.T@Y
        b = -tau.T@Y
        mp.AddQuadraticCost(Q=Q,b=b,vars=theta)

        if feasibility_constrained:
            # s.t. Pat's LMI realizability conditions
            Sigma = 0.5*np.trace(I)*np.eye(3) - I
            J = np.block([[ Sigma, h],
                          [ h.T,   m]])
            mp.AddPositiveSemidefiniteConstraint(J)

        res = Solve(mp)

        if res.is_success():
            theta_hat = res.GetSolution(theta)
            print(theta_hat[0])


    def CalcParameterEstimate(self, context, output):
        """
        Compute the latest parameter estimates using Bayesian linear regression and
        the fact that the dynamics are linear in the interial parameters:

            Y(q, v, vd)*theta = tau

        """
        # Get data from input ports
        t = context.get_time()
        q = self.ee_pose_port.Eval(context)
        v = self.ee_twist_port.Eval(context)
        tau = self.ee_wrench_port.Eval(context)

        # Estimate acceleration numerically
        vd = (v - self.v_last)/self.dt
        self.v_last = v

        m_hat = 0

        #if t >= 9:
        #    # Wait until we have a hold on the object to do any estimation
        #    print("End-effector torque: %s" % tau)

        #    # Construct the regression matrix Y
        #    Y, b = single_body_regression_matrix(vd, v)

        #    # Record regression matrix and applied wrenches
        #    self.Ys.append(Y)
        #    self.taus.append(tau)

        #    # Get a least-squares estimate of the parameters
        #    #self.CalcLSE(feasibility_constrained=False)

        # Do a simple estimation of the mass of the held object
        if t >= 10:
            # Wait until we have a hold on the object to do any estimation
            # Also need to wait until object isn't moving, since we're not
            # considering any sort of inertia effects

            # We'll do regression with 
            #
            #       f = ma
            #     f_z+mg = ma
            #       (a-g)*m = f_z
            #       
            f_z = tau[5]  # force applied in z-direction on end-effector
            g = -9.81     # acceleration due to gravity
            a = vd[5]     # acceleration of end-effector in z direction

            # Get a point estimate of the mass based only on the current data 
            # (basically least-squares)
            m_hat = f_z/(a-g)

            # Perform a Bayesian estimate
            #m_hat = self.DoBayesianUpdate(Y=
            
            print(m_hat)

        output.SetFromVector([m_hat])


def single_body_regression_matrix(a,v):
    """
    Given a spatial acceleration (a) and a spatial velocity (v) of
    a rigid object expressed in the end-effector frame, compute the 
    regression matrix Y(a,v) which relates (lumped) inertial parameters
    theta to spatial forces (f) applied in the end-effector frame. 
        f = Y(a,v)*theta. 
    We assume that theta = [ m  ]    (mass of object)
                           [ mc ]    (mass times position of CoM in end-effector frame
                           [ Ixx ]   (rotational inertia about end-effector frame)
                           [ Iyy ]
                           [ Izz ]
                           [ Ixy ]
                           [ Ixz ]
                           [ Iyz ]
    Note that the position of the CoM in the end-effector frame can be computed as
        c = mc/m.
    Similarly, the rotational inertia about the CoM can be computed using
        Ibar = Ibar_com + m*S(c)*S(c)'.
    """      

    # Create symbolic version of parameters theta
    m = Variable("m")                # mass
    #m = 1.0
    mc = MakeVectorVariable(3,"mc")  # mass times position of CoM in end-effector frame

    Ixx = Variable("Ixx")            # rotational inerta *about end-effector frame*
    Iyy = Variable("Iyy")            # (not CoM frame)
    Izz = Variable("Izz")
    Ixy = Variable("Ixy")
    Ixz = Variable("Ixz")
    Iyz = Variable("Iyz")
    Ibar = np.array([[Ixx, Ixy, Ixz],   
                     [Ixy, Iyy, Iyz],
                     [Ixz, Iyz, Izz]])

    I = np.block([[ Ibar   , S(mc)       ],   # Spatial inertia of the body in the 
                  [ S(mc).T, m*np.eye(3) ]])  # end-effector frame
    
    theta = np.hstack([m, mc, Ixx, Iyy, Izz, Ixy, Ixz, Iyz])
    #theta = np.hstack([mc, Ixx, Iyy, Izz, Ixy, Ixz, Iyz])

    # Create symbolic expression for spatial force,
    #
    #  f = Ia + v x* Iv
    #
    g = np.array([0,0,0,0,0,-9.81])
    f = I@a + spatial_force_cross_product(v, I@v)# - m*g

    # Write this expression as linear in the parameters,
    #
    # f = Y(a,v)*theta + b
    #
    Y, b = DecomposeAffineExpressions(f,vars=theta)

    return (Y,b)

def mbp_version(a,v):
    
    # Define variables theta
    m = 1.0                          # assume mass is 1
    m_com = MakeVectorVariable(3,"m_com")
    Ixx = Variable("Ixx")            # rotational inerta *about end-effector frame*
    Iyy = Variable("Iyy")            # (not CoM frame)
    Izz = Variable("Izz")
    Ixy = Variable("Ixy")
    Ixz = Variable("Ixz")
    Iyz = Variable("Iyz")
    theta = np.hstack([m_com, Ixx, Iyy, Izz, Ixy, Ixz, Iyz])

    # Spatial inertia of the object
    I = SpatialInertia_[Expression](
            m,
            m_com,
            UnitInertia_[Expression](Ixx,Iyy,Izz,Ixy,Ixz,Iyz))

    # Create plant which consists of a single rigid body
    plant = MultibodyPlant(1.0)  # timestep is irrelevant
    block = plant.AddRigidBody("block", SpatialInertia())

    # gravity off for now
    #plant.mutable_gravity_field().set_gravity_vector([0,0,0])

    plant.Finalize()

    # Convert the plant to symbolic form 
    sym_plant = plant.ToSymbolic()
    sym_context = sym_plant.CreateDefaultContext()

    # Set velocities from input
    sym_plant.SetVelocities(sym_context,v)   # from v

    # Set the spatial inertia of the symbolic plant to the symbolic version
    # (based on variables theta)
    sym_block = sym_plant.GetBodyByName("block")
    sym_block.SetSpatialInertiaInBodyFrame(sym_context, I)

    # Run inverse dynamics to get applied spatial forces consistent with acceleration a
    f_ext = MultibodyForces_[Expression](sym_plant)  # zero
    f = sym_plant.CalcInverseDynamics(sym_context, a, f_ext)

    # Write f as affine in theta: f = Y*theta + b
    Y, b = DecomposeAffineExpressions(f,theta)

    return (Y,b)

def S(p):
    """
    Given a vector p in R^3, return the skew-symmetric cross product matrix
    S = [ 0  -p2  p1]
        [ p2  0  -p0]
        [-p1  p0  0 ]
    such that S(p)*a = p x a
    """
    return np.array([[    0, -p[2], p[1]],
                     [ p[2],   0  ,-p[0]],
                     [-p[1],  p[0],  0  ]])

def spatial_force_cross_product(v_sp, f_sp):
    """
    Given a spatial velocity
        v_sp = [w;v]
    and a spatial force

        f_sp = [n;f],
    compute the spatial (force) cross product
        f_sp x* f_sp = [wxn + vxf]
                       [   wxf   ].
    """
    w = v_sp[:3]   # decompose linear and angular ocmponents
    v = v_sp[3:]

    n = f_sp[:3]
    f = f_sp[3:]

    return np.hstack([ np.cross(w,n) + np.cross(v,f),
                            np.cross(w,f)           ])
