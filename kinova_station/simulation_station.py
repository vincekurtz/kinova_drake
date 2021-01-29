from pydrake.all import *
from kinova_station.common import (EndEffectorTarget, 
                                   GripperTarget, 
                                   EndEffectorWrenchCalculator,
                                   CameraPosePublisher)
class KinovaStation(Diagram):
    """
    A template system diagram for controlling a 7 DoF Kinova Gen3 robot, modeled 
    after Drake's ManipulationStation, but with the kinova instead of a kuka arm.
   
                               ---------------------------------
                               |                               |
                               |                               |
                               |                               |
                               |                               | --> measured_arm_position
                               |                               | --> measured_arm_velocity
    ee_target ---------------> |         KinovaStation         | --> measured_arm_torque
    ee_target_type ----------> |                               |
                               |                               |
                               |                               | --> measured_ee_pose
                               |                               | --> measured_ee_twist
                               |                               | --> measured_ee_wrench
    gripper_target ----------> |                               |
    gripper_target_type -----> |                               |
                               |                               | --> measured_gripper_position
                               |                               | --> measured_gripper_velocity
                               |                               |
                               |                               | --> camera_rgb_image
                               |                               | --> camera_depth_image
                               |                               | --> camera_transform
                               |                               |
                               |                               |
                               ---------------------------------

    The input ee_target can be a desired end-effector pose, twist or wrench, as specified by
    ee_target_type. 

    Similarly, the gripper_target can be a desired gripper position or velocity, as specified
    by gripper_target_type. 
   
    """
    def __init__(self, time_step=0.002):
        Diagram.__init__(self)
        self.set_name("kinova_manipulation_station")

        self.builder = DiagramBuilder()

        self.scene_graph = self.builder.AddSystem(SceneGraph())
        self.scene_graph.set_name("scene_graph")

        self.plant = self.builder.AddSystem(MultibodyPlant(time_step=time_step))
        self.plant.RegisterAsSourceForSceneGraph(self.scene_graph)
        self.plant.set_name("plant")

        # A separate plant which only has access to the robot arm + gripper mass,
        # and not any other objects in the scene
        self.controller_plant = MultibodyPlant(time_step=time_step)

        # Body id's and poses for any extra objects in the scene
        self.object_ids = []
        self.object_poses = []

        # Which sort of gripper we're using. (Robotiq Hand-e, Robotiq 2F-85, or none)
        self.gripper_type = None   # None, hande, or 2f_85

        # Whether or not we have a camera in the simulation
        self.has_camera = False

    def Finalize(self):
        """
        Do some final setup stuff. Must be called after making modifications
        to the station (e.g. adding arm, gripper, manipulands) and before using
        this diagram as a system. 
        """

        if self.gripper_type == "2f_85":
            # Add bushings to model the kinematic loop in the 2F-85 gripper.
            # This needs to be done pre-Finalize
            add_2f_85_bushings(self.plant, self.gripper)

        self.plant.Finalize()
        self.controller_plant.Finalize()
        
        # Set up the scene graph
        self.builder.Connect(
                self.scene_graph.get_query_output_port(),
                self.plant.get_geometry_query_input_port())
        self.builder.Connect(
                self.plant.get_geometry_poses_output_port(),
                self.scene_graph.get_source_pose_port(self.plant.get_source_id()))

        # Create controller
        cartesian_controller = self.builder.AddSystem(CartesianController(
                                        self.controller_plant,
                                        self.controller_arm))
        cartesian_controller.set_name("cartesian_controller")

        # End effector target and target type go to the controller
        self.builder.ExportInput(cartesian_controller.ee_target_port,
                                 "ee_target")
        self.builder.ExportInput(cartesian_controller.ee_target_type_port,
                                 "ee_target_type")

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
                cartesian_controller.arm_position_port)
        self.builder.Connect(
                demux.get_output_port(1),
                cartesian_controller.arm_velocity_port)

        # Torques from controller go to the simulated plant
        self.builder.Connect(
                cartesian_controller.GetOutputPort("applied_arm_torque"),
                self.plant.get_actuation_input_port(self.arm))

        # Controller outputs measured arm torques, end-effector pose, end-effector twist
        self.builder.ExportOutput(
                cartesian_controller.GetOutputPort("applied_arm_torque"),
                "measured_arm_torque")
        self.builder.ExportOutput(
                cartesian_controller.GetOutputPort("measured_ee_pose"),
                "measured_ee_pose")
        self.builder.ExportOutput(
                cartesian_controller.GetOutputPort("measured_ee_twist"),
                "measured_ee_twist")
        
        # Create gripper controller
        gripper_controller = self.builder.AddSystem(GripperController(self.gripper_type))
        gripper_controller.set_name("gripper_controller")

        # Connect gripper controller to the diagram
        self.builder.ExportInput(
                gripper_controller.GetInputPort("gripper_target"),
                "gripper_target")
        self.builder.ExportInput(
                gripper_controller.GetInputPort("gripper_target_type"),
                "gripper_target_type")

        self.builder.Connect(
                self.plant.get_state_output_port(self.gripper),
                gripper_controller.GetInputPort("gripper_state"))
        self.builder.Connect(
                gripper_controller.GetOutputPort("applied_gripper_torque"),
                self.plant.get_actuation_input_port(self.gripper))
    
        # Send gripper position and velocity as an output
        self.builder.ExportOutput(
                gripper_controller.GetOutputPort("measured_gripper_position"),
                "measured_gripper_position")
        self.builder.ExportOutput(
                gripper_controller.GetOutputPort("measured_gripper_velocity"),
                "measured_gripper_velocity")
        
        # Compute and output end-effector wrenches based on measured joint torques
        wrench_calculator = self.builder.AddSystem(EndEffectorWrenchCalculator(
                self.controller_plant,
                self.controller_plant.GetFrameByName("end_effector_link")))
        wrench_calculator.set_name("wrench_calculator")

        self.builder.Connect(
                demux.get_output_port(0),
                wrench_calculator.GetInputPort("joint_positions"))
        self.builder.Connect(
                demux.get_output_port(1),
                wrench_calculator.GetInputPort("joint_velocities"))
        self.builder.Connect(
                cartesian_controller.GetOutputPort("applied_arm_torque"),
                wrench_calculator.GetInputPort("joint_torques"))

        self.builder.ExportOutput(
                wrench_calculator.get_output_port(),
                "measured_ee_wrench")

        # Configure camera
        if self.has_camera:
            
            # Create and add the camera system
            camera_parent_body_id = self.plant.GetBodyFrameIdIfExists(self.camera_parent_frame.body().index())
            camera = self.builder.AddSystem(RgbdSensor(camera_parent_body_id,
                                                        self.X_camera,
                                                        self.color_camera,
                                                        self.depth_camera))
            camera.set_name("camera")

            # Wire the camera to the scene graph
            self.builder.Connect(
                    self.scene_graph.get_query_output_port(),
                    camera.query_object_input_port())

            self.builder.ExportOutput(
                    camera.color_image_output_port(),
                    "camera_rgb_image")
            self.builder.ExportOutput(
                    camera.depth_image_32F_output_port(),
                    "camera_depth_image")

            # Send pose of camera in world frame as output
            camera_transform_pub = self.builder.AddSystem(CameraPosePublisher(self.X_camera))
            camera_transform_pub.set_name("camera_transform_publisher")

            self.builder.Connect(
                    cartesian_controller.GetOutputPort("measured_ee_pose"),
                    camera_transform_pub.GetInputPort("ee_pose"))
            self.builder.ExportOutput(
                    camera_transform_pub.GetOutputPort("camera_transform"),
                    "camera_transform")

        # Build the diagram
        self.builder.BuildInto(self)

    def SetupSinglePegScenario(self, gripper_type="hande", arm_damping=False):
        """
        Set up a scenario with the robot arm, a gripper, and a single peg. 
        And connect to the Drake visualizer while we're at it.
        """
        self.AddGround()
        if gripper_type == "hande":
            self.AddArmWithHandeGripper(arm_damping=arm_damping)
        elif gripper_type == "2f_85":
            self.AddArmWith2f85Gripper(arm_damping=arm_damping)
        else:
            raise RuntimeError("Invalid gripper type: %s" % gripper_type)

        X_peg = RigidTransform()
        X_peg.set_translation([0.8,0,0.1])
        X_peg.set_rotation(RotationMatrix(RollPitchYaw([0,np.pi/2,0])))
        self.AddManipulandFromFile("./models/manipulands/peg.sdf", X_peg)

        self.ConnectToDrakeVisualizer()

    def AddGround(self):
        """
        Add a flat ground with friction
        """
        X_BG = RigidTransform()
        surface_friction = CoulombFriction(
                static_friction = 0.7,
                dynamic_friction = 0.5)
        self.plant.RegisterCollisionGeometry(
                self.plant.world_body(),
                X_BG,
                HalfSpace(),
                "ground_collision",
                surface_friction)
        self.plant.RegisterVisualGeometry(
                self.plant.world_body(),
                X_BG,
                HalfSpace(),
                "ground_visual",
                np.array([0.5,0.5,0.5,0]))  # transparent

    def AddArm(self, include_damping=False):
        """
        Add the 7-dof gen3 arm to the system.
        """
        if include_damping:
            # The hardware system has lots of damping so this is more realistic,
            # but requires a simulation with small timesteps.
            arm_urdf = "./models/gen3_7dof/urdf/GEN3_URDF_V12_with_damping.urdf"
        else:
            arm_urdf = "./models/gen3_7dof/urdf/GEN3_URDF_V12.urdf"

        self.arm = Parser(plant=self.plant).AddModelFromFile(arm_urdf, "arm")
        self.controller_arm = Parser(plant=self.controller_plant).AddModelFromFile(arm_urdf, "arm")

        # Fix the base of the arm to the world
        self.plant.WeldFrames(self.plant.world_frame(),
                              self.plant.GetFrameByName("base_link",self.arm))

        self.controller_plant.WeldFrames(self.controller_plant.world_frame(),
                                         self.controller_plant.GetFrameByName("base_link", self.controller_arm))

    def AddHandeGripper(self):
        """
        Add the Hand-e gripper to the system. The arm must be added first. 
        """
        self.gripper_type = "hande"

        # Add a gripper with actuation to the full simulated plant
        gripper_urdf = "./models/hande_gripper/urdf/robotiq_hande.urdf"
        self.gripper = Parser(plant=self.plant).AddModelFromFile(gripper_urdf,"gripper")

        self.plant.WeldFrames(self.plant.GetFrameByName("end_effector_link",self.arm),
                              self.plant.GetFrameByName("hande_base_link", self.gripper))

        # Add a gripper without actuation to the controller plant
        gripper_static_urdf = "./models/hande_gripper/urdf/robotiq_hande_static.urdf"
        static_gripper = Parser(plant=self.controller_plant).AddModelFromFile(
                                                                gripper_static_urdf,
                                                                "gripper")

        self.controller_plant.WeldFrames(
                self.controller_plant.GetFrameByName("end_effector_link",self.controller_arm),
                self.controller_plant.GetFrameByName("hande_base_link", static_gripper))

    def Add2f85Gripper(self):
        """
        Add the Robotiq 2F-85 gripper to the system. The arm must be added first. 
        """
        self.gripper_type = "2f_85"

        # Add a gripper with actuation to the full simulated plant
        gripper_urdf = "./models/2f_85_gripper/urdf/robotiq_2f_85.urdf"
        self.gripper = Parser(plant=self.plant).AddModelFromFile(gripper_urdf,"gripper")

        X_grip = RigidTransform()
        X_grip.set_rotation(RotationMatrix(RollPitchYaw([0,0,np.pi/2])))
        self.plant.WeldFrames(self.plant.GetFrameByName("end_effector_link",self.arm),
                              self.plant.GetFrameByName("robotiq_arg2f_base_link", self.gripper),
                              X_grip)

        # Add a gripper without actuation to the controller plant
        gripper_static_urdf = "./models/2f_85_gripper/urdf/robotiq_2f_85_static.urdf"
        static_gripper = Parser(plant=self.controller_plant).AddModelFromFile(
                                                                gripper_static_urdf,
                                                                "gripper")

        self.controller_plant.WeldFrames(
                self.controller_plant.GetFrameByName("end_effector_link",self.controller_arm),
                self.controller_plant.GetFrameByName("robotiq_arg2f_base_link", static_gripper), 
                X_grip)

    def AddCamera(self, show_window=False):
        """
        Add a simulated camera which is mounted to the robot end-effector, in roughly
        the same position as it is on the real robot. 
        """
        # Add renderer to scene graph
        renderer_name = "kinova_camera_renderer"
        self.scene_graph.AddRenderer(renderer_name, 
                                     MakeRenderEngineVtk(RenderEngineVtkParams()))
  
        # Set camera properites. These roughly correspond
        # to the camera on the hardware (Intel Realsense D410).
        intrinsics = CameraInfo(width=270, height=480, fov_y=np.radians(40))
        clipping = ClippingRange(0.01,3.0)
        X_lens = RigidTransform()
        camera_core = RenderCameraCore(renderer_name, intrinsics, clipping, X_lens)
        depth_range = DepthRange(0.1, 2.0)

        # Create the camera model
        self.color_camera = ColorRenderCamera(camera_core, show_window=show_window)
        self.depth_camera = DepthRenderCamera(camera_core, depth_range)
        
        # Set the frame and position of the camera
        self.camera_parent_frame = self.plant.GetFrameByName("end_effector_link", self.arm)
        self.X_camera = RigidTransform()         # position of camera in parent frame
        self.X_camera.set_translation([0,0.1,0.0])

        self.has_camera = True

        
    def AddArmWithHandeGripper(self, arm_damping=False):
        """
        Add the 7-dof arm and a model of the hande gripper to the system.
        """
        self.AddArm(include_damping=arm_damping)
        self.AddHandeGripper()
    
    def AddArmWith2f85Gripper(self, arm_damping=False):
        """
        Add the 7-dof arm and a model of the 2F-85 gripper to the system.
        """
        self.AddArm(include_damping=arm_damping)
        self.Add2f85Gripper()

    def AddManipulandFromFile(self, model_file, X_WObject):
        """
        Add an object to the simulation and place it in the given pose in the world
        """
        manipuland = Parser(plant=self.plant).AddModelFromFile(model_file)
        body_indices = self.plant.GetBodyIndices(manipuland)

        assert len(body_indices) == 1, "Only single-body objects are supported for now"
        
        self.object_ids.append(body_indices[0])
        self.object_poses.append(X_WObject)

    def SetArmPositions(self, diagram, diagram_context, q):
        """
        Set arm positions to the given values. Must be called after the overall
        system diagram is built, and the associated diagram_context set. 
        """
        plant_context = diagram.GetMutableSubsystemContext(self.plant, diagram_context)
        self.plant.SetPositions(plant_context, self.arm, q)

    def SetManipulandStartPositions(self, diagram, diagram_context):
        """
        Set positions of any manipulands to their starting values. Must be called
        after the overall system diagram is built, and the associated diagram_context set.
        """
        assert len(self.object_ids) == len(self.object_poses), "Manipuland poses and ids don't match"

        plant_context = diagram.GetMutableSubsystemContext(self.plant, diagram_context)
        
        for i in range(len(self.object_ids)):
            self.plant.SetFreeBodyPose(plant_context, 
                                       self.plant.get_body(self.object_ids[i]),
                                       self.object_poses[i])


    def ConnectToDrakeVisualizer(self):
        visualizer_params = DrakeVisualizerParams(role=Role.kIllustration)
        DrakeVisualizer().AddToBuilder(builder=self.builder,
                                       scene_graph=self.scene_graph,
                                       params=visualizer_params)

    def ConnectToMeshcatVisualizer(self, start_server=True):
        if start_server:
            # Start meshcat server. This saves the step of opening meshcat separately,
            # but does mean you need to refresh the page each time you re-run something.
            from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
            proc, zmq_url, web_url = start_zmq_server_as_subprocess()

        # Defining self.meshcat in this way allows us to connect to 
        # things like a point-cloud visualizer later
        self.meshcat = ConnectMeshcatVisualizer(builder=self.builder,
                                 zmq_url = zmq_url,
                                 scene_graph=self.scene_graph,
                                 output_port=self.scene_graph.get_query_output_port())

    def go_home(self, diagram, diagram_context, name="Home"):
        """
        Move the arm to the specified home position. Must be one of
        'Home', 'Retract', or 'Zero'.
        """
        if name == "Home":
            q0 = np.array([0, np.pi/12, np.pi, 4.014-2*np.pi, 0, 0.9599, np.pi/2])
        elif name == "Retract":
            q0 = np.array([0, 5.93-2*np.pi, np.pi, 3.734-2*np.pi, 0, 5.408-2*np.pi, np.pi/2])
        elif name == "Zero":
            q0 = np.zeros(7)
        else:
            raise RuntimeError("Home position name must be one of ['Home', 'Retract', 'Zero']")

        self.SetArmPositions(diagram, diagram_context, q0)



class GripperController(LeafSystem):
    """
    A simple gripper controller with two modes: position and velocity. 
    Both modes are essentially simple PD controllers. 

                            -------------------------
                            |                       |
                            |                       |
    gripper_target -------> |   GripperController   | ---> applied_gripper_torques
    gripper_target_type --> |                       |
                            |                       | ---> measured_gripper_position
                            |                       | ---> measured_gripper_velocity
    gripper_state --------> |                       |
                            |                       |
                            |                       |
                            -------------------------
    """
    def __init__(self, gripper_type):
        """
        The type can be "hande" or "2f_85", depending on the type of gripper.
        """
        LeafSystem.__init__(self)
        self.type = gripper_type
        
        # State input port size depends on what type of gripper we're using.
        # (2F-85 has many more degrees of freedom. )
        if self.type == "2f_85":
            state_size = 12

            # We'll create a simple model of the gripper which is welded to the floor. 
            # This will allow us to compute the distance between fingers.
            self.plant = MultibodyPlant(1.0)  # timestep doesn't matter
            gripper_urdf = "./models/2f_85_gripper/urdf/robotiq_2f_85.urdf"
            self.gripper = Parser(plant=self.plant).AddModelFromFile(gripper_urdf, "gripper")
            self.plant.WeldFrames(self.plant.world_frame(),
                                  self.plant.GetFrameByName("robotiq_arg2f_base_link",self.gripper))
            self.plant.Finalize()
            self.context = self.plant.CreateDefaultContext()

        elif self.type == "hande":
            state_size = 4
        else:
            raise RuntimeError("Invalid gripper type: %s" % self.type)

        # Declare input ports
        self.target_port = self.DeclareVectorInputPort(
                                  "gripper_target",
                                  BasicVector(1))
        self.target_type_port = self.DeclareAbstractInputPort(
                                  "gripper_target_type",
                                  AbstractValue.Make(GripperTarget.kPosition))

        self.state_port = self.DeclareVectorInputPort(
                                    "gripper_state",
                                    BasicVector(state_size))

        # Declare output ports
        self.DeclareVectorOutputPort(
                "applied_gripper_torque",
                BasicVector(2),
                self.CalcGripperTorque)
        self.DeclareVectorOutputPort(
                "measured_gripper_position",
                BasicVector(1),
                self.CalcGripperPosition,
                {self.time_ticket()}   # indicate that this doesn't depend on any inputs,
                )                      # but should still be updated each timestep
        self.DeclareVectorOutputPort(
                "measured_gripper_velocity",
                BasicVector(1),
                self.CalcGripperVelocity,
                {self.time_ticket()})

    def ComputePosition(self, state):
        """
        Compute the gripper position from state data. 
        This is especially useful for the 2F-85 gripper, since the
        state does not map neatly to the finger positions. 
        """
        if self.type == "hande":
            # For the simple Hand-e gripper, prismatic joint positions map fairly
            # directly to gripper position.
            finger_position = 0.03 - state[:2]   # each finger travels roughly 30mm,
                                                 # and is zero in the open position
        else:
            # For the more complex 2F-85 gripper, we need to do some kinematics 
            # calculations to figure out the gripper position
            self.plant.SetPositionsAndVelocities(self.context, state)

            right_finger = self.plant.GetFrameByName("right_inner_finger_pad")
            left_finger = self.plant.GetFrameByName("left_inner_finger_pad")
            base = self.plant.GetFrameByName("robotiq_arg2f_base_link")
           
            X_lf = self.plant.CalcRelativeTransform(self.context, 
                                                    left_finger,
                                                    base)
            X_rf = self.plant.CalcRelativeTransform(self.context, 
                                                    right_finger,
                                                    base)
           
            lf_pos = -X_lf.translation()[1]
            rf_pos = -X_rf.translation()[1]

            finger_position = np.array([lf_pos,rf_pos])

        return finger_position

    def ComputeVelocity(self, state):
        """
        Compute the gripper velocity from state data.
        This is especially useful for the 2F-85 gripper, since the
        state does not map neatly to the finger positions. 
        """
        if self.type == "hande":
            finger_velocity = -state[2:]
        else:
            # For the more complex 2F-85 gripper, we need to do some kinematics 
            self.plant.SetPositionsAndVelocities(self.context, state)
            v = state[-self.plant.num_velocities():]

            right_finger = self.plant.GetFrameByName("right_inner_finger_pad")
            left_finger = self.plant.GetFrameByName("left_inner_finger_pad")
            base = self.plant.GetFrameByName("robotiq_arg2f_base_link")
           
            J_lf = self.plant.CalcJacobianTranslationalVelocity(self.context,
                                                                JacobianWrtVariable.kV,
                                                                left_finger,
                                                                np.zeros(3),
                                                                base,
                                                                base)
            J_rf = self.plant.CalcJacobianTranslationalVelocity(self.context,
                                                                JacobianWrtVariable.kV,
                                                                right_finger,
                                                                np.zeros(3),
                                                                base,
                                                                base)
            
            lf_vel = -(J_lf@v)[1]
            rf_vel = (J_rf@v)[1]

            finger_velocity = np.array([lf_vel,rf_vel])

        return finger_velocity

    def CalcGripperPosition(self, context, output):
        state = self.state_port.Eval(context)

        if self.type == "hande":
            width = 0.03
        else:  #2f_85
            width = 0.06

        # Send a single number to match the hardware
        both_finger_positions = self.ComputePosition(state)
        net_position = 1/width* np.mean(both_finger_positions)

        output.SetFromVector([net_position])
        
    def CalcGripperVelocity(self, context, output):
        state = self.state_port.Eval(context)
        
        if self.type == "hande":
            width = 0.03
        else:  #2f_85
            width = 0.06

        # Send a single number to match the hardware
        both_finger_velocity = self.ComputeVelocity(state)
        net_velocity = 1/width* np.mean(both_finger_velocity)

        output.SetFromVector([net_velocity])

    def CalcGripperTorque(self, context, output):
        state = self.state_port.Eval(context)
        target = self.target_port.Eval(context)
        target_type = self.target_type_port.Eval(context)

        finger_position = self.ComputePosition(state)
        finger_velocity = self.ComputeVelocity(state)

        # Set PD gains depending on gripper type
        if self.type == "hande":
            width = 0.03
            Kp = 100*np.eye(2)
            Kd = 2*np.sqrt(Kp)
        else:
            width = 0.06
            Kp = 10*np.eye(2)
            Kd = 2*np.sqrt(0.01*Kp)
        
        # Set target positions and velocities based on the current control mode
        if target_type == GripperTarget.kPosition:
            target = width - width*target*np.ones(2)
            target_finger_position = target
            target_finger_velocity = np.zeros(2)
        elif target_type == GripperTarget.kVelocity:
            target_finger_position = finger_position
            target_finger_velocity = -width*target
        else:
            raise RuntimeError("Invalid gripper target type: %s" % target_type)

        # Determine applied torques with PD controller
        position_err = target_finger_position - finger_position
        velocity_err = target_finger_velocity - finger_velocity
        tau = -Kp@(position_err) - Kd@(velocity_err)

        output.SetFromVector(tau)


class CartesianController(LeafSystem):
    """
    A controller which imitates the cartesian control mode of the Kinova gen3 arm.
 
                         -------------------------
                         |                       |
                         |                       |
    ee_target ---------> |  CartesianController  | ----> applied_arm_torque
    ee_target_type ----> |                       |
                         |                       |
                         |                       | ----> measured_ee_pose
    arm_position ------> |                       | ----> measured_ee_twist
    arm_velocity ------> |                       |
                         |                       |
                         |                       |
                         -------------------------

    The type of target is determined by ee_target_type, and can be
        EndEffectorTarget.kPose,
        EndEffectorTarget.kTwist,
        EndEffectorTarget.kWrench.

    """
    def __init__(self, plant, arm_model):
        LeafSystem.__init__(self)

        self.plant = plant
        self.arm = arm_model
        self.context = self.plant.CreateDefaultContext()

        # Define input ports
        self.ee_target_port = self.DeclareVectorInputPort(
                                        "ee_target",
                                        BasicVector(6))
        self.ee_target_type_port = self.DeclareAbstractInputPort(
                                            "ee_target_type",
                                            AbstractValue.Make(EndEffectorTarget.kPose))

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

        self.DeclareVectorOutputPort(
                "measured_ee_pose",
                BasicVector(6),
                self.CalcEndEffectorPose,
                {self.time_ticket()}   # indicate that this doesn't depend on any inputs,
                )                      # but should still be updated each timestep
        self.DeclareVectorOutputPort(
                "measured_ee_twist",
                BasicVector(6),
                self.CalcEndEffectorTwist,
                {self.time_ticket()})

        # Define some relevant frames
        self.world_frame = self.plant.world_frame()
        self.ee_frame = self.plant.GetFrameByName("end_effector_link")

        # Set joint limits (set self.{q,qd}_{min,max})
        self.GetJointLimits()
           
        # Store desired end-effector pose and corresponding joint
        # angles so we only run full IK when we need to
        self.last_ee_pose_target = None
        self.last_q_target = None
    
    def GetJointLimits(self):
        """
        Iterate through self.plant to establish joint angle
        and velocity limits. 

        Sets:

            self.q_min
            self.q_max
            self.qd_min
            self.qd_max

        """
        q_min = []
        q_max = []
        qd_min = []
        qd_max = []

        joint_indices = self.plant.GetJointIndices(self.arm)

        for idx in joint_indices:
            joint = self.plant.get_joint(idx)
            
            if joint.type_name() == "revolute":  # ignore the joint welded to the world
                q_min.append(joint.position_lower_limit())
                q_max.append(joint.position_upper_limit())
                qd_min.append(joint.velocity_lower_limit())  # note that higher limits
                qd_max.append(joint.velocity_upper_limit())  # are availible in cartesian mode

        self.q_min = np.array(q_min)
        self.q_max = np.array(q_max)
        self.qd_min = np.array(qd_min)
        self.qd_max = np.array(qd_max)

    def CalcEndEffectorPose(self, context, output):
        """
        This method is called each timestep to determine the end-effector pose
        """
        q = self.arm_position_port.Eval(context)
        qd = self.arm_velocity_port.Eval(context)
        self.plant.SetPositions(self.context,q)
        self.plant.SetVelocities(self.context,qd)

        # Compute the rigid transform between the world and end-effector frames
        X_ee = self.plant.CalcRelativeTransform(self.context,
                                                self.world_frame,
                                                self.ee_frame)

        ee_pose = np.hstack([RollPitchYaw(X_ee.rotation()).vector(), X_ee.translation()])

        output.SetFromVector(ee_pose)
    
    def CalcEndEffectorTwist(self, context, output):
        """
        This method is called each timestep to determine the end-effector twist
        """
        q = self.arm_position_port.Eval(context)
        qd = self.arm_velocity_port.Eval(context)
        self.plant.SetPositions(self.context,q)
        self.plant.SetVelocities(self.context,qd)

        # Compute end-effector Jacobian
        J = self.plant.CalcJacobianSpatialVelocity(self.context,
                                                   JacobianWrtVariable.kV,
                                                   self.ee_frame,
                                                   np.zeros(3),
                                                   self.world_frame,
                                                   self.world_frame)

        ee_twist = J@qd
        output.SetFromVector(ee_twist)
    
    def CalcArmTorques(self, context, output):
        """
        This method is called each timestep to determine output torques
        """
        q = self.arm_position_port.Eval(context)
        qd = self.arm_velocity_port.Eval(context)
        self.plant.SetPositions(self.context,q)
        self.plant.SetVelocities(self.context,qd)

        # Some dynamics computations
        tau_g = -self.plant.CalcGravityGeneralizedForces(self.context)

        # Indicate what type of command we're recieving
        target_type = self.ee_target_type_port.Eval(context)

        if target_type == EndEffectorTarget.kWrench:
            # Compute joint torques consistent with the desired wrench
            wrench_des = self.ee_target_port.Eval(context)

            # Compute end-effector jacobian
            J = self.plant.CalcJacobianSpatialVelocity(self.context,
                                                       JacobianWrtVariable.kV,
                                                       self.ee_frame,
                                                       np.zeros(3),
                                                       self.world_frame,
                                                       self.world_frame)

            tau = tau_g + J.T@wrench_des

        elif target_type == EndEffectorTarget.kTwist:
            # Compue joint torques consistent with the desired twist
            twist_des = self.ee_target_port.Eval(context)

            # Use DoDifferentialInverseKinematics to determine desired qd
            params = DifferentialInverseKinematicsParameters(self.plant.num_positions(),
                                                             self.plant.num_velocities())
            params.set_timestep(0.005)
            params.set_joint_velocity_limits((self.qd_min, self.qd_max))
            params.set_joint_position_limits((self.q_min, self.q_max))

            result = DoDifferentialInverseKinematics(self.plant,
                                                     self.context,
                                                     twist_des,
                                                     self.ee_frame,
                                                     params)

            if result.status == DifferentialInverseKinematicsStatus.kSolutionFound:
                qd_nom = result.joint_velocities
            else:
                print("Differential inverse kinematics failed!")
                qd_nom = np.zeros(7)

            # Select desired accelerations using a proportional controller
            Kp = 10*np.eye(7)
            qdd_nom = Kp@(qd_nom - qd)

            # Compute joint torques consistent with these desired accelerations
            f_ext = MultibodyForces(self.plant)
            tau = tau_g + self.plant.CalcInverseDynamics(self.context, qdd_nom, f_ext)

        elif target_type == EndEffectorTarget.kPose:
            # Compute joint torques which move the end effector to the desired pose
            rpy_xyz_des = self.ee_target_port.Eval(context)

            # Only do a full inverse kinematics solve if the target pose has changed
            if (rpy_xyz_des != self.last_ee_pose_target).any():
                
                X_WE_des = RigidTransform(RollPitchYaw(rpy_xyz_des[:3]),
                                          rpy_xyz_des[-3:])         
          
                # First compute joint angles consistent with the desired pose using Drake's IK.
                # This sets up a nonconvex optimization problem to find joint angles consistent
                # with the given constraints
                ik = InverseKinematics(self.plant,self.context)
                ik.AddPositionConstraint(self.ee_frame,
                                         [0,0,0],
                                         self.world_frame,
                                         X_WE_des.translation(), 
                                         X_WE_des.translation())
                ik.AddOrientationConstraint(self.ee_frame,
                                            RotationMatrix(),
                                            self.world_frame,
                                            X_WE_des.rotation(),
                                            0.001)

                prog = ik.get_mutable_prog()
                q_var = ik.q()
                prog.AddQuadraticErrorCost(np.eye(len(q_var)), q, q_var)
                prog.SetInitialGuess(q_var, q)
                result = Solve(ik.prog())

                if not result.is_success():
                    print("Inverse Kinematics Failed!")
                    q_nom = np.zeros(7)
                else:
                    q_nom = result.GetSolution(q_var)

                    # Save the results of this solve for later
                    self.last_ee_pose_target = rpy_xyz_des
                    self.last_q_target = q_nom

            else:
                q_nom = self.last_q_target

            qd_nom = np.zeros(7)

            # Use PD controller to map desired q, qd to desired qdd
            Kp = 1*np.eye(7)
            Kd = 2*np.sqrt(Kp)  # critical damping
            qdd_nom = Kp@(q_nom - q) + Kd@(qd_nom - qd)

            # Compute joint torques consistent with these desired qdd
            f_ext = MultibodyForces(self.plant)
            tau = tau_g + self.plant.CalcInverseDynamics(self.context, qdd_nom, f_ext)

        else:
            raise RuntimeError("Invalid target type %s" % target_type)

        output.SetFromVector(tau)


def add_2f_85_bushings(plant, gripper):
    """
    The Robotiq 2F-85 gripper has a complicated mechanical component which includes
    a kinematic loop. The original URDF deals with this by using a "mimic" tag,
    but Drake doesn't support this. So we'll try to close the loop using a bushing,
    as described in Drake's four-bar linkage example: 

        https://github.com/RobotLocomotion/drake/tree/master/examples/multibody/four_bar.

    This needs to be done before plant is finalized. 
    """
    left_inner_finger = plant.GetFrameByName("left_inner_finger", gripper)
    left_inner_knuckle = plant.GetFrameByName("left_inner_knuckle", gripper)
    right_inner_finger = plant.GetFrameByName("right_inner_finger", gripper)
    right_inner_knuckle = plant.GetFrameByName("right_inner_knuckle", gripper)

    # Add frames which are located at the desired linkage point
    X_finger = RigidTransform()
    X_finger.set_translation([0.0,-0.016,0.007])
    X_knuckle = RigidTransform()
    X_knuckle.set_translation([0.0,0.038,0.043])

    left_inner_finger_bushing = FixedOffsetFrame(
                                        "left_inner_finger_bushing",
                                        left_inner_finger,
                                        X_finger,
                                        gripper)
    left_inner_knuckle_bushing = FixedOffsetFrame(
                                        "left_inner_knuckle_bushing",
                                        left_inner_knuckle,
                                        X_knuckle,
                                        gripper)
    right_inner_finger_bushing = FixedOffsetFrame(
                                        "right_inner_finger_bushing",
                                        right_inner_finger,
                                        X_finger,
                                        gripper)
    right_inner_knuckle_bushing = FixedOffsetFrame(
                                        "right_inner_knuckle_bushing",
                                        right_inner_knuckle,
                                        X_knuckle,
                                        gripper)


    plant.AddFrame(left_inner_finger_bushing)
    plant.AddFrame(left_inner_knuckle_bushing)
    plant.AddFrame(right_inner_finger_bushing)
    plant.AddFrame(right_inner_knuckle_bushing)

    # Force and torque stiffness and damping describe a revolute joint on the z-axis
    k_xyz = 8000
    d_xyz = 10
    k_rpy = 15
    d_rpy = 3
    force_stiffness_constants =  np.array([k_xyz,k_xyz,k_xyz])
    force_damping_constants =    np.array([d_xyz,d_xyz,d_xyz])
    torque_stiffness_constants = np.array([0,k_rpy,k_rpy])
    torque_damping_constants =   np.array([0,d_rpy,k_rpy])

    left_finger_bushing = LinearBushingRollPitchYaw(
                left_inner_finger_bushing, left_inner_knuckle_bushing,
                torque_stiffness_constants, torque_damping_constants,
                force_stiffness_constants, force_damping_constants)
    right_finger_bushing = LinearBushingRollPitchYaw(
                right_inner_finger_bushing, right_inner_knuckle_bushing,
                torque_stiffness_constants, torque_damping_constants,
                force_stiffness_constants, force_damping_constants)
    plant.AddForceElement(left_finger_bushing)
    plant.AddForceElement(right_finger_bushing)


