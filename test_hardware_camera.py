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

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

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

            plt.figure()
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
    #cap_receive = cv2.VideoCapture('rtspsrc location=rtsp://192.168.1.10/depth latency=30 ! rtph264depay !  videoconvert ! appsink', cv2.CAP_GSTREAMER)

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

            print(type(frame[0,0]))
            print(frame[100:110, 200:210])

            # TODO: figure out how this works calibration-wise.
            # So far these depth images don't look particularly convincing...

            plt.figure()
            plt.imshow(frame)


            #cv2.imshow("depth_image", frame)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            #cv2.imwrite('depth_image.png', frame)
    else:
        print("Image stream not opened")

def gst_to_opencv(sample):
    """
    Convert a Gstreamer byte array into np array
    """
    buf = sample.get_buffer()
    caps = sample.get_caps()

    print(buf.get_size())
    
    print(caps.get_structure(0).get_value('height'))
    print(caps.get_structure(0).get_value('width'))



    array = np.ndarray(
        (
            caps.get_structure(0).get_value('height'),
            caps.get_structure(0).get_value('width'),
            1
        ),
        buffer=buf.extract_dup(0, buf.get_size()), dtype=np.uint16)
    return array

def show_depth_image_alt():
    """
    Capture and show a depth image a different way, roughly following
    https://gist.github.com/patrickelectric/443645bb0fd6e71b34c504d20d475d5a,
    https://stackoverflow.com/questions/43777428/capture-gstreamer-network-video-with-python
    """
    Gst.init(None)

    #command = 'rtspsrc location=rtsp://192.168.1.10/color latency=30 ! rtph264depay ! avdec_h264 ! videoconvert ! appsink'
    command = 'rtspsrc location=rtsp://192.168.1.10/depth latency=30 ! rtpgstdepay ! videoconvert ! appsink'

    video_pipe = Gst.parse_launch(command)
    video_pipe.set_state(Gst.State.PLAYING)

    video_sink = video_pipe.get_by_name('appsink0')

    def callback(sink):
        print("hello callback")

    video_sink.connect('new-sample', callback)
    sample = video_sink.emit("pull-sample")

    frame = gst_to_opencv(sample).reshape((270, 480))
    plt.imshow(frame)

    print("done")


if __name__=="__main__":
    #show_color_image()
    #show_depth_image()
    show_depth_image_alt()
    plt.show()
