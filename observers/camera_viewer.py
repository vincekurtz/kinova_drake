##
#
# Describes a simple camera viewer class that subscribes to image topics
# and shows the associated images. 
#
##

from pydrake.all import *

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

        # Declare input ports
        self.color_image_port = self.DeclareAbstractInputPort(
                "color_image",
                AbstractValue.Make(sample_color_image))
        # TODO: include depth image

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

        # TODO: do some sort of visualization, maybe with matplotlib
