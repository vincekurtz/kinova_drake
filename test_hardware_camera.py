#!/usr/bin/env python

##
#
# Try to read camera data from the robot to opencv.
# (The ultimate goal being to create a Drake Image)
#
##

import cv2
import numpy as np
import matplotlib.pyplot as plt

#print(cv2.getBuildInformation())  # check that GStreamer is included

def show_color_image():
    """
    Capture and display a color image from the camera
    """
    cap_receive = cv2.VideoCapture('rtspsrc location=rtsp://192.168.1.10/color latency=30 ! rtph264depay ! avdec_h264 ! videoconvert ! appsink', cv2.CAP_GSTREAMER)

    if cap_receive.isOpened():
        # Once we've opened the video stream, we can make multiple calls to
        # the read() method to capture new frames
        ret,frame = cap_receive.read()

        if not ret:
            print("empty frame")
        else:
            # 'frame' is a numpy array containing the image 

            plt.imshow(frame)
            
            #cv2.imshow("color_image", frame)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            #cv2.imwrite("color_image.png", frame)

    else:
        print("Image stream not opened")

def show_depth_image():
    """
    Capture and display a depth image from the camera
    """
    cap_receive = cv2.VideoCapture('rtspsrc location=rtsp://192.168.1.10/depth latency=30 ! rtpgstdepay ! videoconvert ! appsink', cv2.CAP_GSTREAMER)

    if cap_receive.isOpened():
        # Once we've opened the video stream, we can make multiple calls to
        # the read() method to capture new frames
        ret,frame = cap_receive.read()

        if not ret:
            print("empty frame")
        else:
            # 'frame' is a numpy array containing the image 

            #with open("depth_image_saved.npy", 'wb') as f:
            #    np.save(f, frame)

            # To get a good visual of the depth image, we'll rescale so all
            # values are in [0,255]
            max_val = np.max(frame)
            frame = frame*(255/max_val)

            # TODO: figure out how this works calibration-wise.
            # So far these depth images don't look particularly convincing...

            plt.figure()
            plt.imshow(frame)
            plt.show()


            #cv2.imshow("depth_image", frame)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            #cv2.imwrite('depth_image.png', frame)
    else:
        print("Image stream not opened")


if __name__=="__main__":
    show_color_image()
    show_depth_image()
