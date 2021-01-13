from pydrake.all import *

class KinovaStation(Diagram):
    """
    A template system diagram for controlling 7 DoF Kinova Gen3 robot, modeled 
    after Drake's ManipulationStation, but with the kinova instead of a kuka arm.
   
                               ---------------------------------
                               |                               |
                               |                               |
                               |                               |
                               |                               | --> measured_arm_position
                               |                               | --> measured_arm_velocity
    target_ee_pose ----------> |         KinovaStation         | --> measured_arm_torque
    target_ee_twist ---------> |                               |
    target_ee_wrench --------> |                               |
                               |                               | --> measured_ee_pose
                               |                               | --> measured_ee_twist
                               |                               | --> measured_ee_wrench
                               |                               |
                               |                               |
    target_gripper_position -> |                               | --> measured_gripper_position
    target_gripper_velocity -> |                               | --> measured_gripper_velocity
                               |                               | --> measured_gripper_force
                               |                               |
                               |                               | --> camera_rgb_image (TODO)
                               |                               | --> camera_depth_image (TODO)
                               |                               |
                               ---------------------------------
   
    """
    def __init__(self, time_step=0.002):
        Diagram.__init__(self)
        self.set_name("manipulation_station")

        self.builder = DiagramBuilder()

        self.scene_graph = self.builder.AddSystem(SceneGraph())
        self.scene_graph.set_name("scene_graph")

        self.plant = self.builder.AddSystem(MultibodyPlant(time_step=time_step))
        self.plant.RegisterAsSourceForSceneGraph(self.scene_graph)
        self.plant.set_name("plant")

    def Finalize(self):
        """
        Do some final setup stuff. Must be called after making modifications
        to the station (e.g. adding arm, gripper, manipulands) and before using
        this diagram as a system. 
        """
        self.plant.Finalize()
        
        # Set up the scene graph
        self.builder.Connect(
                self.scene_graph.get_query_output_port(),
                self.plant.get_geometry_query_input_port())
        self.builder.Connect(
                self.plant.get_geometry_poses_output_port(),
                self.scene_graph.get_source_pose_port(self.plant.get_source_id()))

        # Create controller
        ctrl = self.builder.AddSystem(CartesianController(
                                        self.plant,
                                        self.arm))
        ctrl.set_name("cartesian_controller")

        # Inputs of target end-effector pose, twist, wrench go to the controller
        self.builder.ExportInput(ctrl.target_ee_pose_port,
                                 "target_ee_pose")
        self.builder.ExportInput(ctrl.target_ee_twist_port,
                                 "target_ee_twist")
        self.builder.ExportInput(ctrl.target_ee_wrench_port,
                                 "target_ee_wrench")

        # Output measured arm position and velocity
        demux = self.builder.AddSystem(Demultiplexer(
                                        self.plant.num_multibody_states(self.arm),
                                        self.plant.num_positions(self.arm)))
        demux.set_name("demux")
        self.builder.Connect(
                self.plant.get_state_output_port(self.arm),
                demux.get_input_port(0))
        self.builder.ExportOutput(
                demux.get_output_port(0),
                "measured_arm_position")
        self.builder.ExportOutput(
                demux.get_output_port(1),
                "measured_arm_velocity")
        
        # Measured arm position and velocity are sent to the controller
        self.builder.Connect(
                demux.get_output_port(0),
                ctrl.arm_position_port)
        self.builder.Connect(
                demux.get_output_port(1),
                ctrl.arm_velocity_port)

        # Torques from controller go to the simulated plant
        self.builder.Connect(
                ctrl.GetOutputPort("applied_arm_torque"),
                self.plant.get_actuation_input_port(self.arm))

        # Applied torques are treated as an output
        self.builder.ExportOutput(
                ctrl.GetOutputPort("applied_arm_torque"),
                "measured_arm_torques")

        # Build the diagram
        self.builder.BuildInto(self)

    def AddGround(self):
        """
        Add a flat ground with friction
        """
        pass

    def SetupArmOnly(self):
        """
        Add the 7-dof gen3 arm to the system.
        """
        arm_urdf = "./models/gen3_7dof/urdf/GEN3_URDF_V12.urdf"
        self.arm = Parser(plant=self.plant).AddModelFromFile(arm_urdf, "arm")

        # Fix the base of the arm to the world
        self.plant.WeldFrames(self.plant.world_frame(),
                              self.plant.GetFrameByName("base_link",self.arm))

    def SetupArmWithHandeGripper(self):
        pass

    def AddManipulandFromFile(self, model_file, X_WObject):
        pass

    def ConnectToDrakeVisualizer(self):
        visualizer_params = DrakeVisualizerParams(role=Role.kIllustration)
        DrakeVisualizer().AddToBuilder(builder=self.builder,
                                       scene_graph=self.scene_graph,
                                       params=visualizer_params)

class CartesianController(LeafSystem):
    """
    A controller which imitates the cartesian control mode of the Kinova gen3 arm.
 
                         -------------------------
                         |                       |
    target_ee_pose ----> |                       |
    target_ee_twist ---> |  CartesianController  |
    target_ee_wrench --> |                       |
                         |                       | ----> applied_arm_torque
                         |                       |
    arm_position ------> |                       |
    arm_velocity ------> |                       |
                         |                       |
                         |                       |
                         -------------------------


    If a target twist is avialible, this takes precidence over a the given target pose.
    Similarly, if a target end-effector wrench is availible, this takes precidence over 
    a target twist and pose.
    """
    def __init__(self, plant, arm_model):
        LeafSystem.__init__(self)

        self.plant = plant
        self.arm = arm_model
        self.context = self.plant.CreateDefaultContext()

        # Define input ports
        self.target_ee_pose_port = self.DeclareVectorInputPort(
                                        "target_ee_pose",
                                        BasicVector(6))
        self.target_ee_twist_port = self.DeclareVectorInputPort(
                                        "target_ee_twist",
                                        BasicVector(6))
        self.target_ee_wrench_port = self.DeclareVectorInputPort(
                                        "target_ee_wrench",
                                        BasicVector(6))

        self.arm_position_port = self.DeclareVectorInputPort(
                                        "arm_position",
                                        BasicVector(self.plant.num_positions(self.arm)))
        self.arm_velocity_port = self.DeclareVectorInputPort(
                                        "arm_velocity",
                                        BasicVector(self.plant.num_velocities(self.arm)))

        # Define output ports
        self.DeclareVectorOutputPort(
                "applied_arm_torque",
                BasicVector(self.plant.num_actuators()),
                self.CalcArmTorques)
    
    def CalcArmTorques(self, context, output):
        tau = np.zeros(7)
        output.SetFromVector(tau)


if __name__=="__main__":
    KST = KinovaStationTemplate()

