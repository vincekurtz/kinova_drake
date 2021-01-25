#!/usr/bin/env python

##
#
# Figure out how to handle depth images aquired from the hardware interface. 
#
##

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pydrake.all import *

# load saved image
with open("depth_image_saved.npy", "rb") as f:
    frame = np.load(f)  # I feel like these should not be all ints...

# show in a pretty way with 
#frame = frame[10:-10, 50:-10]
#plt.imshow(frame)
#plt.show()

# Create a drake depth image
depth_image = Image[PixelType.kDepth16U](width=frame.shape[1],height=frame.shape[0])
depth_image.mutable_data[:,:] = frame.T[np.newaxis].T

plt.imshow(np.squeeze(depth_image.data))
plt.show()
