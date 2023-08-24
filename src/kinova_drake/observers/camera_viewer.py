##
#
# Describes a simple camera viewer class that subscribes to image topics
# and shows the associated images. 
#
##

from pydrake.all import *
import cv2

class CameraViewer(LeafSystem):
    """
    An observer which makes visualizations of camera data

                        -------------------------
                        |                       |
    color_image ------> |                       |
                        |     CameraViewer      |
    depth_image ------> |                       |
                        |                       |
                        -------------------------

    """
    def __init__(self):
        LeafSystem.__init__(self)

        # Create example images which will be used to define the 
        # (abstract) import port types
        sample_color_image = Image[PixelType.kRgba8U](width=640,height=480)
        sample_depth_image = Image[PixelType.kDepth32F](width=640,height=480)

        # Declare input ports
        self.color_image_port = self.DeclareAbstractInputPort(
                "color_image",
                AbstractValue.Make(sample_color_image))
        self.depth_image_port = self.DeclareAbstractInputPort(
                "depth_image",
                AbstractValue.Make(sample_depth_image))

        # Dummy continuous variable so we update at each timestep
        self.DeclareContinuousState(1)

    def DoCalcTimeDerivatives(self, context, continuous_state):
        """
        This method gets called every timestep. Its nominal purpose is
        to update the (dummy) continuous variable for the simulator, but
        here we'll use it to read in camera images from the input ports
        and do some visualization.
        """

        color_image = self.color_image_port.Eval(context)
        depth_image = self.depth_image_port.Eval(context)

        # color_image.data and depth_image.data contain raw np arrays
        # with the image. So we can do things like load into opencv, etc
        
        # Example of displaying the depth image (then waiting for a keystroke
        # to move to the next timestep:
        #cv2.imshow("depth_image", depth_image.data)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

