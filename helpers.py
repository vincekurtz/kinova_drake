# Miscellaneous helper functions
from pydrake.all import *
import numpy as np

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

    # Create symbolic expression for spatial force,
    #
    #  f = Ia + v x* Iv
    #
    g = np.array([0,0,0,0,0,-9.81])
    f = I@a + spatial_force_cross_product(v, I@v)# - m*g

    # Write this expression as linear in the parameters,
    #
    # f = Y(a,v)*theta
    #
    Y = DecomposeLinearExpressions(f,vars=theta)

    return Y

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

def jacobian2(function, x):
    """
    This is a rewritting of the jacobian function from drake which addresses
    a strange bug that prevents computations of Jdot.

    Compute the jacobian of the function evaluated at the vector input x
    using Eigen's automatic differentiation. The dimension of the jacobian will
    be one more than the output of ``function``.

    ``function`` should be vector-input, and can be any dimension output, and
    must return an array with AutoDiffXd elements.
    """
    x = np.asarray(x)
    assert x.ndim == 1, "x must be a vector"
    x_ad = np.empty(x.shape, dtype=np.object)
    for i in range(x.size):
        der = np.zeros(x.size)
        der[i] = 1
        x_ad.flat[i] = AutoDiffXd(x.flat[i], der)
    y_ad = np.asarray(function(x_ad))

    yds = []
    for y in y_ad.flat:
        yd = y.derivatives()
        if yd.shape == (0,):
            yd = np.zeros(x_ad.shape)
        yds.append(yd)

    return np.vstack(yds).reshape(y_ad.shape + (-1,))

