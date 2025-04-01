#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torch

import csv
import random
import requests
import string


from random import randint
from std_msgs.msg import String
from PIL import Image as PILImage
from PIL import ImageFont, ImageDraw
from sklearn.cluster import KMeans
from collections import Counter

'''
Subscribes to: high-res camera, cb_detected nodes
Publishes to: /score_tracker

@brief: If sign_detected = true, then process image and publish clue string.
        else, do nothing.
        
        Keep track of number of clues read (i.e. number of times sign_detected switched from F to T, vice versa) 
        in here, and handle publishing. (This will run in parallel with driving processing.)
'''

NUM_FRAMES = 5
frame_clue_predictions = [] # will hold NN predictions for each frame
frame_value_predictions = [] # will hold NN predictions for each frame
class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

class ClueBoardDetector:
# this object simply detects the blue border around a clue board and projects the cb into a rectangle, priming it for the cb model to use.
    def __init__(self):
        rospy.init_node("clue_board_detector", anonymous=True) # initialize script as a ROS node
        self.bridge = CvBridge() # object to convert ROS img msgs to numpy for cv2 processing
        self.image_sub = rospy.Subscriber("/B1/rrbot/camera1/image_raw", Image, self.image_callback)
        # self.image_sub = None # placeholder for img subscriber. gets initialized when PID node sends "ok"
        # self.read_cb = rospy.Subscriber("/start_signal", String, self.ok_callback) # subscribe to the /start_signal topic from the PID node to get the "ok" to read.

        self.mask_pub = rospy.Publisher('/masked_feed', Image, queue_size=1)
        self.inverted_pub = rospy.Publisher('/inverted_feed', Image, queue_size=1)
        self.contour_pub = rospy.Publisher('/contoured_feed', Image, queue_size=1)
        self.projected_cb_pub = rospy.Publisher('/projected_cb_feed', Image, queue_size=1)

        # self.model = CharNNModel() # implement later
        self.model.load_state_dict(torch.load("/ros_ws/src/my_controller/models/charNNModel.pth", map_location=torch.device('cpu'))) #implement later, including adding model folder
        self.model.eval()

    # def ok_callback(self, msg):
    #     if msg.data.lower() == "ok" and self.image_sub is None: # begins if we get the "ok" and we aren't already subscribed
    #         rospy.loginfo("Starting image stream.") # ROS version of a print statement
    #         self.image_sub = rospy.Subscriber("/B1/rrbot/camera1/image_raw", Image, self.image_callback) # start intaking frames


    def image_callback(self, msg):
        width, height = 600, 400
        

        rospy.loginfo("Received Image")
        try:
            # Convert ROS Image to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return
        
        # if no error, then we have obtained an image and bridged to cv2 format.
        # now, we can process the image!
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([110,200,50])
        upper_blue = np.array([135,255,220])


        # create a mask with only blue colors from the frame
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # bw_masked = cv2.imread(mask, cv2.COLOR_BGR2GRAY)
        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(mask, encoding="mono8")) # masked (telemetry)

        inverted = cv2.bitwise_not(mask)

        # Find Canny edges
        edged = cv2.Canny(inverted, 30, 255)

        # Find Contours using a copy of the image 
        contours, hierarchy = cv2.findContours(edged.copy(),
            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = float('inf')
        inner_contour = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000: #skip tiny noise. this also means the sign itself must > 1000 (in range)
                continue

            # Approximate shape
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Keep only 4-corner shapes (rectangle-like)
            if len(approx) == 4 and area < min_area:
                min_area = area
                inner_contour = approx
        
        # Draw all contours
        contour_feed = cv2.cvtColor(edged.copy(), cv2.COLOR_GRAY2BGR)
        if inner_contour is not None:
            cv2.drawContours(contour_feed, [inner_contour], -1, (0,255,0), 3)
            self.contour_pub.publish(self.bridge.cv2_to_imgmsg(contour_feed, encoding="bgr8"))
        
        dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
        ], dtype='float32')

        # Sort corners of the contour into an array
        src = sort_corners(inner_contour)

        # Compute transformation matrix
        M = cv2.getPerspectiveTransform(src, dst)

        # Transform the frame
        projected = cv2.warpPerspective(frame, M, (width, height))

        # Mask the frame
        hsv_projected = cv2.cvtColor(projected, cv2.COLOR_BGR2HSV)
        lower_blue_projected = np.array([110,100,50]) # make it less
        
        mask_projected = cv2.inRange(hsv_projected, lower_blue_projected, upper_blue)

        inverted_projected = cv2.bitwise_not(mask_projected)

        self.inverted_pub.publish(self.bridge.cv2_to_imgmsg(inverted_projected, encoding="mono8"))

        # Crop the clueboard into characters:

        # Find contours
        contours, _ = cv2.findContours(inverted_projected, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

def sort_corners(pts):
        pts = pts.reshape(4, 2)
        sum_pts = pts.sum(axis=1)
        diff_pts = np.diff(pts, axis=1)

        top_left = pts[np.argmin(sum_pts)]
        bottom_right = pts[np.argmax(sum_pts)]
        top_right = pts[np.argmin(diff_pts)]
        bottom_left = pts[np.argmax(diff_pts)]

        return np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

def start_controller():
    cb_detector = ClueBoardDetector()
    rospy.spin()

if __name__ == "__main__":
    start_controller()