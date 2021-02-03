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

    def StorePointCloud(self, point_cloud):
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

        # Estimate normals

        # Save
        self.stored_point_clouds.append(o3d_cloud)


    def CalcEndEffectorCommand(self, context, output):
        """
        Compute and send an end-effector twist command.
        """
        t = context.get_time()

        if t < self.cs.total_duration():
            # Just follow the default command sequence while we're building up the point cloud
            CommandSequenceController.CalcEndEffectorCommand(self, context, output)

            if t % 1 == 0:
                # Only fetch the point clouds about once per second, since this is slow
                point_cloud = self.point_cloud_input_port.Eval(context)

                # Convert to Open3D, crop, compute normals, and save
                self.StorePointCloud(point_cloud)

        else:
            # Once the intial scanning sequence is done, compute a feasible grasp, go 
            # there, and pick up the object
            pass
            
            # Merge stored point clouds

            # Find a collision-free grasp

            # Go to the grasp location, close the gripper, and move the object
            # to a target location.
