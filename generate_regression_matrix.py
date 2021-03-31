#!/usr/bin/env python3

##
#
# This script generates an import-able function Y(q,v,vd) which 
# allows us to write the dynamics of a single rigid body as linear
# in the lumped inertial parameters, i.e.,
#
#   f = Y(q,v,vd)*theta
#
# where
#   - f \in R^6 are the spatial forces applied to the object
#   - q \in R^7 is the object's position (quaternion + position)
#   - v \in R^6 is the object's spatial velocity
#   - vd \in R^6 is the object's spatial acceleration (derivative of v)
#   - theta are the object's lumped inertial parameters, which include
#       - m, the object's mass
#       - h \in R^3, the mass times the position of the center-of-mass
#         relative to the point where f is applied
#       - Ibar \in R^(3x3), the object's rotational inertia about the 
#         point where f is applied
#
##

from pydrake.all import *
import sympy as sp
from helpers import drake_to_sympy
import cloudpickle  # allows us to pickle lambda functions

# Create a symbolic plant consisting of a single rigid body
print("Creating symbolic plant")
plant = MultibodyPlant(time_step=1.0)
Parser(plant=plant).AddModelFromFile("./models/manipulands/peg.sdf","peg")
plant.Finalize()
plant = plant.ToSymbolic()
peg = plant.GetBodyByName("base_link")
context = plant.CreateDefaultContext()

# Set the lumped parameters (theta) as Drake symbolic variables
# (this overwrites the parameters from the urdf file above)
print("Setting Drake Symbolic Variables")
m = Variable("m")
h = np.array([Variable("hx"), Variable("hy"), Variable("hz")])
Ixx = Variable("Ixx")
Iyy = Variable("Iyy")
Izz = Variable("Izz")
Ixy = Variable("Ixy")
Ixz = Variable("Ixz")
Iyz = Variable("Iyz")
Ibar = RotationalInertia_[Expression](Ixx, Iyy, Izz, Ixy, Ixz, Iyz)  # note: see ReExpress for changing frames
Ibar_unit = UnitInertia_[Expression]().SetFromRotationalInertia(Ibar,m)
I = SpatialInertia_[Expression](m, h/m, Ibar_unit)
peg.SetSpatialInertiaInBodyFrame(context, I)

# Set the position, velocity, and acceleration of the object as
# Drake symbolic variables.
q = np.array([Variable("q%s" % i) for i in range(7)])
v = np.array([Variable("v%s" % i) for i in range(6)])
vd = np.array([Variable("vd%s" % i) for i in range(6)])

plant.SetPositions(context, q)
plant.SetVelocities(context, v)

# Create sympy versions of the lumped dynamics parameters
# This is needed because sympy lets us do more powerful
# simplifications, as well as create lambda functions from 
# symbolic expressions. 
print("Defining sympy Symbolic Variables")
m_sp, hx_sp, hy_sp, hz_sp = sp.symbols("m, hx, hy, hz")
Ixx_sp, Iyy_sp, Izz_sp, Ixy_sp, Ixz_sp, Iyz_sp = \
        sp.symbols("Ixx, Iyy, Izz, Ixy, Ixz, Iyz")
sp_vars = {"m":m_sp, "hx":hx_sp, "hy":hy_sp, "hz":hz_sp,
        "Ixx":Ixx_sp, "Iyy":Iyy_sp, "Izz":Izz_sp, 
        "Ixy":Ixy_sp, "Ixz":Ixz_sp, "Iyz":Iyz_sp}
theta = np.asarray([*sp_vars.values()])

# Create sympy versions of position, velocity, acceleration
q_sp = np.array([sp.symbols("q%s" % i) for i in range(7)])
v_sp = np.array([sp.symbols("v%s" % i) for i in range(6)])
vd_sp = np.array([sp.symbols("vd%s" % i) for i in range(6)])

for i in range(7):
    sp_vars["q%s" % i] = q_sp[i]
for i in range(6):
    sp_vars["v%s" % i] = v_sp[i]
    sp_vars["vd%s" % i] = vd_sp[i]

# Use Drake's inverse dynamics to compute a symbolic expression
# for the forces f needed to consistent with accelerations vd. 
print("Computing Inverse Dynamics")
f_ext = MultibodyForces_[Expression](plant)
f = plant.CalcInverseDynamics(context, vd, f_ext)

# Convert this expression to a sympy expression, and write as
# f = Y(q,v,vd)*theta
print("Converting to Sympy")
f_sp = drake_to_sympy(f, sp_vars)
print("Writing as Linear Expression")
Y_sp, _ = sp.linear_eq_to_matrix(f_sp, theta)
print("Simplifying")
Y_sp = sp.simplify(Y_sp)

# Create a corresponding lambda function
print("Creating Lambda Function")
Y_fcn = sp.lambdify([q_sp, v_sp, vd_sp], Y_sp)

# Save this lambda function so we can load it in other scripts
print("Saving Lambda Function")
with open("single_body_regression_matrix.pkl", 'wb') as f:
    cloudpickle.dump(Y_fcn, f)
