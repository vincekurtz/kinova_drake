#!/usr/bin/env python

# Simulate a single rigid body actuated by a spatial force, as a simple
# example of state estimation

from pydrake.all import *
from helpers import *

# Parameters
sim_time = 5
dt = 5e-3
realtime_rate = 1.0

q0 = np.zeros(7)
q0[3] = 1
v0 = np.zeros(6)

# Controller which applies spatial forces
class SpatialForceCtrl(LeafSystem):
    def __init__(self, plant, body_name):
        LeafSystem.__init__(self)

        self.DeclareAbstractOutputPort(   # for passing to the simulator
                "spatial_force",
                lambda: AbstractValue.Make(
                    [ExternallyAppliedSpatialForce()]),
                self.CalcOutput)

        self.DeclareVectorOutputPort(     # for easier logging
                "spatial_force_vector",
                BasicVector(6),
                self.CalcVectorOutput)

        self.body_index = plant.GetBodyByName(body_name).index()
        
        self.f = np.zeros(6)

    def CalcVectorOutput(self, context, output):
        output.SetFromVector(self.f)

    def CalcOutput(self, context, output):
        t = context.get_time()
        tau = np.array([0.0,
                        0.001*np.sin(10*t),
                        0.001*np.cos(10*t)])
        f = np.array([0.0,
                      0.0,
                      0.981 + 0.1*np.sin(10*t)])

        self.f = np.hstack([tau,f])

        spatial_force = ExternallyAppliedSpatialForce()
        spatial_force.body_index = self.body_index
        spatial_force.F_Bq_W = SpatialForce(tau=self.f[0:3],f=self.f[3:])
        output.set_value([spatial_force])

# Plant setup
builder = DiagramBuilder()
scene_graph = builder.AddSystem(SceneGraph())
plant = builder.AddSystem(MultibodyPlant(time_step=dt))
plant.RegisterAsSourceForSceneGraph(scene_graph)
peg = Parser(plant=plant).AddModelFromFile("./models/manipulands/peg.sdf","peg")
plant.Finalize()

# Diagram setup
builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port())
builder.Connect(
        plant.get_geometry_poses_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id()))

controller = builder.AddSystem(SpatialForceCtrl(plant, "base_link"))
builder.Connect(
        controller.GetOutputPort("spatial_force"),
        plant.get_applied_spatial_force_input_port())

DrakeVisualizer().AddToBuilder(builder=builder,scene_graph=scene_graph)

state_logger = LogOutput(plant.get_state_output_port(),builder)
ctrl_logger = LogOutput(controller.GetOutputPort("spatial_force_vector"),builder)

diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()

# Simulator setup
simulator = Simulator(diagram, diagram_context)
simulator.set_target_realtime_rate(realtime_rate)

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
Ys = []
Fs = []
for i in range(N):
    Y, b = single_body_regression_matrix(vd[:,i],v[:,i])
    Ys.append(Y)
    Fs.append(f[:,i] - b)

Y = np.vstack(Ys)
F = np.hstack(Fs)

prog = MathematicalProgram()
theta = prog.NewContinuousVariables(10,1,"theta")

m = theta[0]
h = theta[1:4]
I = np.array([[theta[4,0], theta[7,0], theta[8,0]],
              [theta[7,0], theta[5,0], theta[9,0]],
              [theta[8,0], theta[9,0], theta[6,0]]])

# min \| Y*theta - F\|^2
Q = Y.T@Y
b = -F.T@Y
prog.AddQuadraticCost(Q=Q,b=b,vars=theta)

# s.t. I > 0
#prog.AddPositiveSemidefiniteConstraint(I)

# s.t. Pat's LMI realizability conditions
Sigma = 0.5*np.trace(I)*np.eye(3) - I
J = np.block([[ Sigma, h],
              [ h.T,   m]])
prog.AddPositiveSemidefiniteConstraint(J)

# s.t. m = 0.1 (true value)
#prog.AddConstraint(m[0] == 0.1)

res = Solve(prog)
theta_hat = res.GetSolution(theta)

m_hat = theta_hat[0]     # mass
h_hat = theta_hat[1:4]  # mass*(position of CoM in end-effector frame)
I_hat = np.array([[theta_hat[4], theta_hat[7], theta_hat[8]],
                  [theta_hat[7], theta_hat[5], theta_hat[9]],
                  [theta_hat[8], theta_hat[9], theta_hat[6]]])

print(m_hat)
print(h_hat)
print(I_hat)
