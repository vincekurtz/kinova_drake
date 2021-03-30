#!/usr/bin/env python

# Simulate a single rigid body actuated by a spatial force, as a simple
# example of state estimation

from pydrake.all import *
from helpers import *

# Parameters
sim_time = 5
dt = 5e-3
realtime_rate = 10

q0 = np.zeros(7)
q0[3] = 1
v0 = np.zeros(6)

# Controller which applies spatial forces
class SpatialForceCtrl(LeafSystem):
    def __init__(self, plant, body_name, p_BBq):
        """
        Create the force applying controller. 

        @param plant        the MultiBodyPlant we'll apply forces to
        @param body_name    the name of the rigid body we'll apply forces to
        @param p_BBq        the position of the applied forces (Bq) relative to 
                            the body's base (CoM) frame B.
        """
        LeafSystem.__init__(self)

        self.body_index = plant.GetBodyByName(body_name).index()
        self.p_BBq = p_BBq

        self.DeclareAbstractOutputPort(   # for passing to the simulator
                "spatial_force",
                lambda: AbstractValue.Make(
                    [ExternallyAppliedSpatialForce()]),
                self.CalcOutput)

        self.DeclareVectorOutputPort(     # for easier logging
                "spatial_force_vector",
                BasicVector(6),
                self.CalcVectorOutput)

        self.input_port = self.DeclareVectorInputPort(
                "state",
                BasicVector(plant.num_multibody_states()))
       
        # Store applied spatial forces for easier logging
        self.f = np.zeros(6)

    def CalcVectorOutput(self, context, output):
        output.SetFromVector(self.f)

    def CalcOutput(self, context, output):
        t = context.get_time()
        tau = np.array([0.0,
                        0.0,
                        0.0])
        f = np.array([0.0,
                      0.0,
                      0.02])

        # Get current state of the object
        x = self.input_port.Eval(context)
        q = x[:7]; v = x[7:]
       
        # Store applied spatial forces
        self.f = np.hstack([tau,f])

        # Send as output
        spatial_force = ExternallyAppliedSpatialForce()
        spatial_force.body_index = self.body_index
        spatial_force.p_BoBq_B = self.p_BBq
        spatial_force.F_Bq_W = SpatialForce(tau=self.f[0:3],f=self.f[3:])
        output.set_value([spatial_force])

# Plant setup
builder = DiagramBuilder()
scene_graph = builder.AddSystem(SceneGraph())
plant = builder.AddSystem(MultibodyPlant(time_step=dt))
plant.RegisterAsSourceForSceneGraph(scene_graph)
peg = Parser(plant=plant).AddModelFromFile("./models/manipulands/peg.sdf","peg")
plant.mutable_gravity_field().set_gravity_vector([0,0,0])
plant.Finalize()

# Ground truth inertial parameters
body = plant.GetBodyByName("base_link")
m = body.default_mass()
Ibar_com = body.default_rotational_inertia().CopyToFullMatrix3()
p_com = np.array([-0.02, 0.0, 0.0])
I = np.block([[Ibar_com + m*S(p_com)@S(p_com).T, m*S(p_com) ],
              [ m*S(p_com).T                   , m*np.eye(3)]])

# Diagram setup
builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port())
builder.Connect(
        plant.get_geometry_poses_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id()))

controller = builder.AddSystem(SpatialForceCtrl(plant, "base_link", -p_com))
builder.Connect(
        controller.GetOutputPort("spatial_force"),
        plant.get_applied_spatial_force_input_port())
builder.Connect(
        plant.get_state_output_port(),
        controller.get_input_port())

DrakeVisualizer().AddToBuilder(builder=builder,scene_graph=scene_graph)

state_logger = LogOutput(plant.get_state_output_port(),builder)
ctrl_logger = LogOutput(controller.GetOutputPort("spatial_force_vector"),builder)

diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()

# Simulator setup
simulator = Simulator(diagram, diagram_context)
simulator.set_target_realtime_rate(realtime_rate)
simulator.set_publish_every_time_step(True)

plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
plant.SetPositions(plant_context, peg, q0)
plant.SetVelocities(plant_context, peg, v0)

# Run the simulation
simulator.Initialize()
simulator.AdvanceTo(sim_time)

# Process logged data
x = state_logger.data()
q = x[:plant.num_positions(),:]
v = x[plant.num_positions():,:]
vd = (v[:,1:] - v[:,:-1])/dt
f = ctrl_logger.data()

N = vd.shape[1]
#Ys = []
#Fs = []

RHS = []
err = []
for i in range(N):
    # Double check our spatial dynamics computations with ground-truth values
    f_i = f[:,i]
    q_i = q[:,i]
    a_i = vd[:,i]
    v_i = v[:,i]

    # Transformation from world frame to B (body frame)
    quat = q_i[:4] / np.linalg.norm(q_i[:4])   # normalize quaternion
    r = -q_i[4:]
    R = RotationMatrix(Quaternion(quat)).inverse().matrix()
    X_BW = np.block([[R,                -R@S(r)],
                     [np.zeros((3,3)),    R    ]])

    # Transformation from B (body frame) to Bq (force applied frame)
    X_BqB = np.block([[ np.eye(3),        -S(-p_com)],
                      [ np.zeros((3,3,)), np.eye(3)]])


    # Applied spatial force expressed in Bq
    f_Bq = X_BW@f_i

    rhs1 = I@a_i + spatial_force_cross_product(v_i, I@v_i)  # should equal f_i
    rhs = I@a_i + x_star(v_i)@I@v_i
    RHS.append(rhs)
   
    err.append(np.linalg.norm(f_i - rhs))

    print(f_i)
    print(rhs)
    print(rhs1)
    print("")

    #Y, b = single_body_regression_matrix(vd[:,i],v[:,i])
    #Ys.append(Y)
    #Fs.append(f[:,i] - b)

rhs = np.asarray(RHS)

#plt.plot(f.T)
#plt.plot(v.T)
#plt.plot(RHS)
plt.plot(err)
plt.show()

#Y = np.vstack(Ys)
#F = np.hstack(Fs)
#
#prog = MathematicalProgram()
#theta = prog.NewContinuousVariables(10,1,"theta")
#
#m = theta[0]
#h = theta[1:4]
#I = np.array([[theta[4,0], theta[7,0], theta[8,0]],
#              [theta[7,0], theta[5,0], theta[9,0]],
#              [theta[8,0], theta[9,0], theta[6,0]]])
#
## min \| Y*theta - F\|^2
#Q = Y.T@Y
#b = -F.T@Y
#prog.AddQuadraticCost(Q=Q,b=b,vars=theta)
#
## s.t. I > 0
##prog.AddPositiveSemidefiniteConstraint(I)
#
## s.t. Pat's LMI realizability conditions
#Sigma = 0.5*np.trace(I)*np.eye(3) - I
#J = np.block([[ Sigma, h],
#              [ h.T,   m]])
#prog.AddPositiveSemidefiniteConstraint(J)
#
## s.t. Cheater constraints related to true values
##prog.AddConstraint(m[0] == 0.1)
##prog.AddConstraint(I[1,0] == 0)
##prog.AddConstraint(I[2,0] == 0)
##prog.AddConstraint(I[2,1] == 0)
##prog.AddConstraint(I[0,0] <= 5e-4)
##prog.AddConstraint(I[1,1] <= 5e-4)
##prog.AddConstraint(I[2,2] <= 5e-4)
#
#res = Solve(prog)
#theta_hat = res.GetSolution(theta)
#
#m_hat = theta_hat[0]     # mass
#h_hat = theta_hat[1:4]  # mass*(position of CoM in end-effector frame)
#I_hat = np.array([[theta_hat[4], theta_hat[7], theta_hat[8]],
#                  [theta_hat[7], theta_hat[5], theta_hat[9]],
#                  [theta_hat[8], theta_hat[9], theta_hat[6]]])
#
#print(m_hat)
#print(h_hat)
#print(I_hat)
