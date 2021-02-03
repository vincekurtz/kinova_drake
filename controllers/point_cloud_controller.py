from controllers.command_sequence_controller import *
import open3d as o3d

class PointCloudController(CommandSequenceController):
    """
    A controller which uses point cloud data to plan
    and execute a grasp. 
    """
    def __init__(self, start_sequence=None, 
                       command_type=EndEffectorTarget.kTwist, 
                       Kp=10*np.eye(6), Kd=2*np.sqrt(10)*np.eye(6)):
        """
        Parameters:

            start_sequence: a CommandSequence object for moving around and building up
                            a point cloud. 

            command_type: the type of command that we'll send (kTwist or kWrench)

            Kp/Kd: PD gains
        """
        if start_sequence is None:
            # Create a default starting command sequence for moving around and
            # building up the point cloud
            start_sequence = CommandSequence([])
            start_sequence.append(Command(
                name="front_view",
                target_pose=np.array([0.7*np.pi, 0.0, 0.5*np.pi, 0.5, 0.0, 0.15]),
                duration=3,
                gripper_closed=False))
            start_sequence.append(Command(
                name="left_view",
                target_pose=np.array([0.7*np.pi, 0.0, 0.2*np.pi, 0.6, 0.3, 0.15]),
                duration=3,
                gripper_closed=False))
            start_sequence.append(Command(
                name="front_view",
                target_pose=np.array([0.7*np.pi, 0.0, 0.5*np.pi, 0.5, 0.0, 0.15]),
                duration=3,
                gripper_closed=False))
            start_sequence.append(Command(
                name="right_view",
                target_pose=np.array([0.7*np.pi, 0.0, 0.8*np.pi, 0.6, -0.3, 0.15]),
                duration=3,
                gripper_closed=False))
            start_sequence.append(Command(
                name="home",
                target_pose=np.array([0.5*np.pi, 0.0, 0.5*np.pi, 0.5, 0.0, 0.2]),
                duration=3,
                gripper_closed=False))

        # Initialize the underlying command sequence controller
        CommandSequenceController.__init__(self, start_sequence, 
                                            command_type=command_type, Kp=Kp, Kd=Kd)

        # Create an additional input port for the point cloud
        self.point_cloud_input_port = self.DeclareAbstractInputPort(
                "point_cloud",
                AbstractValue.Make(PointCloud()))

        # Recorded point clouds from multiple different views
        self.stored_point_clouds = []
        self.merged_point_cloud = None

        # Drake model with just a floating gripper, used to evaluate grasp candidates
        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

        gripper_urdf = "./models/hande_gripper/urdf/robotiq_hande_static.urdf"
        self.gripper = Parser(plant=self.plant).AddModelFromFile(gripper_urdf, "gripper")

        # DEBUG: show this floating gripper in meshcat
        self.meshcat = ConnectMeshcatVisualizer(builder=builder, 
                                           zmq_url="tcp://127.0.0.1:6000",
                                           scene_graph=self.scene_graph,
                                           output_port=self.scene_graph.get_query_output_port(),
                                           prefix="test_prefix")
        self.meshcat.load()
        
        self.plant.Finalize()
        self.diagram = builder.Build()
        self.diagram_context = self.diagram.CreateDefaultContext()
        self.plant_context = self.diagram.GetMutableSubsystemContext(self.plant, self.diagram_context)

    def StorePointCloud(self, point_cloud, camera_position):
        """
        Add the given Drake point cloud to our list of point clouds. 

        Converts to Open3D format, crops, and estimates normals before adding
        to self.stored_point_clouds.
        """
        # Convert to Open3D format
        indices = np.all(np.isfinite(point_cloud.xyzs()), axis=0)
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(point_cloud.xyzs()[:, indices].T)
        if point_cloud.has_rgbs():
            o3d_cloud.colors = o3d.utility.Vector3dVector(point_cloud.rgbs()[:, indices].T / 255.)

        # Crop to relevant area
        x_min = 0.5; x_max = 1.5
        y_min = -0.3; y_max = 0.3
        z_min = 0.0; z_max = 0.5
        o3d_cloud = o3d_cloud.crop(o3d.geometry.AxisAlignedBoundingBox(
                                                min_bound=[x_min, y_min, z_min],
                                                max_bound=[x_max, y_max, z_max]))

        # Estimate normals
        o3d_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        o3d_cloud.orient_normals_towards_camera_location(camera_position)

        # Save
        self.stored_point_clouds.append(o3d_cloud)

    def ScoreGraspCandidate(self, ee_pose, cloud=None):
        """
        For the given point cloud (merged, downsampled, with normals) and
        end-effector pose corresponding to a candidate grasp, return the
        score associated with this grasp. 
        """
        if cloud is None:
            cloud = self.merged_point_cloud

        # Set the pose of our internal gripper model
        gripper = self.plant.GetBodyByName("hande_base_link")
        X_WG = RigidTransform(
                RotationMatrix(RollPitchYaw(ee_pose[:3])),
                ee_pose[3:])
        self.plant.SetFreeBodyPose(self.plant_context, gripper, X_WG)

        # Transform the point cloud to the gripper frame
        X_GW = X_WG.inverse()
        pts = np.asarray(cloud.points).T
        p_GC = X_GW.multiply(pts)

        # DEBUG: Visualize the candidate grasp point with meshcat
        self.diagram.Publish(self.diagram_context)


    def CalcEndEffectorCommand(self, context, output):
        """
        Compute and send an end-effector twist command.
        """
        t = context.get_time()

        # DEBUG: just load the saved point cloud from a file to test grasp scoring
        self.merged_point_cloud = o3d.io.read_point_cloud("merged_point_cloud.pcd")
        
        grasp = np.array([0.5*np.pi, 0, 0.5*np.pi, 0.68, 0.0, 0.1])
        self.ScoreGraspCandidate(grasp)


        output.SetFromVector(np.zeros(6))

        #if t < self.cs.total_duration():
        #    # Just follow the default command sequence while we're building up the point cloud
        #    CommandSequenceController.CalcEndEffectorCommand(self, context, output)

        #    if t % 1 == 0 and t != 0:
        #        # Only fetch the point clouds about once per second, since this is slow
        #        point_cloud = self.point_cloud_input_port.Eval(context)

        #        # Convert to Open3D, crop, compute normals, and save
        #        ee_pose = self.ee_pose_port.Eval(context)  # use end-effector position as a rough
        #                                                   # approximation of camera position
        #        self.StorePointCloud(point_cloud, ee_pose[3:])

        #elif self.merged_point_cloud is None:
        #    # Merge stored point clouds
        #    self.merged_point_cloud = self.stored_point_clouds[0]    # Just adding together may not
        #    for i in range(1, len(self.stored_point_clouds)):        # work very well on hardware...
        #        self.merged_point_cloud += self.stored_point_clouds[i]

        #    # Downsample merged point cloud
        #    self.merged_point_cloud = self.merged_point_cloud.voxel_down_sample(voxel_size=0.005)
        #    
        #    # Find a collision-free grasp location
        #    grasp_initial_guess = np.array([0.5*np.pi, 0, 0.5*np.pi, 0.68, 0.0, 0.1])  # TODO: use heuristics to get a good guess

        #    #DEBUG: save point cloud
        #    #o3d.io.write_point_cloud("merged_point_cloud.pcd", self.merged_point_cloud)

        #    score = self.ScoreGraspCandidate(grasp_initial_guess)


        #else:
        #    # Go to the grasp location, close the gripper, and move the object
        #    # to a target location.
        #    pass
