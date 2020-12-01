from pydrake.all import *

class OmniscientObserver(LeafSystem):
    """
    An observer which has full ground-truth state information for 
    everything in the simulation. 
    """
    def __init__(self, plant):
        LeafSystem.__init__(self)

        self.plant = plant
        self.context = self.plant.CreateDefaultContext()

        # Input port for full multi-body state
        self.state_port = self.DeclareVectorInputPort(
                                "state",
                                BasicVector(self.plant.num_positions() +
                                            self.plant.num_velocities()))

        # Output port for ground truth position of manipuland CoM
        self.DeclareVectorOutputPort(
                "manipuland_com_gt",
                BasicVector(3),
                self.CalcGTCoM)
        
        # Relevant models
        self.arm = self.plant.GetModelInstanceByName("gen3")
        self.gripper = self.plant.GetModelInstanceByName("gripper")
        self.manipuland = self.plant.GetModelInstanceByName("manipuland")
        
        # Relevant reference frames
        self.world_frame = self.plant.world_frame()
        self.ee_frame = self.plant.GetFrameByName("end_effector_link")

    def CalcGTCoM(self, context, output):
        """
        Compute the ground-truth position of the manipulated object's 
        center of mass in the world frame. 
        """
        # Use state info from input to update dynamics info
        state = self.state_port.Eval(context)
        q = state[:self.plant.num_positions()]
        v = state[self.plant.num_positions():]

        self.plant.SetPositions(self.context, q)
        self.plant.SetVelocities(self.context, v)

        # Current end-effector position in world frame
        p_ee_world = self.plant.CalcPointsPositions(self.context,
                                                    self.ee_frame,
                                                    np.zeros(3),
                                                    self.world_frame)

        # Manipuland CoM position in world frame
        p_com_world = self.plant.CalcCenterOfMassPosition(self.context,
                                                          [self.manipuland])

        # Manipuland CoM position in end-effector frame
        p_com_ee = self.plant.CalcPointsPositions(self.context,
                                                  self.world_frame,
                                                  p_com_world,
                                                  self.ee_frame)

        print("Ground Truth: %s" % p_com_ee.flatten())
        
        output.SetFromVector(p_com_world)
