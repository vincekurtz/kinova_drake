#!/usr/bin/env python

# Simulate a single rigid body actuated by a spatial force, as a simple
# example of state estimation

from pydrake.all import *
from helpers import *
import sys
import cloudpickle

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
        tau = np.array([0.1,
                        0.0,
                        0.2])
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
        spatial_force.p_BoBq_B = np.zeros(3)  # DEBUG
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
Ibar = body.default_rotational_inertia().CopyToFullMatrix3()
p_com = body.default_com()
h_com = m*p_com

theta_gt = np.hstack([m, h_com, Ibar[0,0], Ibar[1,1], Ibar[2,2], 
                        Ibar[0,1], Ibar[0,2], Ibar[1,2]])

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

# Extract logged data
x = state_logger.data()
q = x[:plant.num_positions(),:]
v = x[plant.num_positions():,:]
vd = (v[:,1:] - v[:,:-1])/dt
f = ctrl_logger.data()
N = vd.shape[1]

# Load regression matrix from file
# generate_regression_matrix.py must be run first
try:
    with open("single_body_regression_matrix.pkl","rb") as in_file:
        Y_fcn = cloudpickle.load(in_file)
except FileNotFoundError:
    print("Error: Need to run `generate_regression_matrix.py` first")
    sys.exit(1)

err = []
for i in range(1,N):
    # Double check our spatial dynamics computations with ground-truth values
    f_i = f[:,i]
    q_i = q[:,i]
    vd_i = vd[:,i]
    v_i = v[:,i]

    # fix acceleration computation
    if np.all(vd_i == np.zeros(6)):
        vd_i = vd[:,i+1]

    Y = Y_fcn(q_i, v_i, vd_i)

    print(Y@theta_gt - f_i)

