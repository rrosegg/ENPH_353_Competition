#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os

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
from tensorflow.keras.models import load_model

NUM_FRAMES = 5
frame_clue_predictions = []
frame_value_predictions = []
class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


class ClueBoardDetector:
    def __init__(self):
        rospy.init_node("clue_board_detector", anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/B1/rrbot/camera1/image_raw", Image, self.image_callback)  # comment out when ok_callback is used
        # self.image_sub = None  # Defer image sub until "ok"
        # self.start_sub = rospy.Subscriber("/start_signal", String, self.ok_callback)

        self.mask_pub = rospy.Publisher('/masked_feed', Image, queue_size=1)
        self.inverted_pub = rospy.Publisher('/inverted_feed', Image, queue_size=1)
        self.contour_pub = rospy.Publisher('/contoured_feed', Image, queue_size=1)
        self.projected_cb_pub = rospy.Publisher('/projected_cb_feed', Image, queue_size=1)

        self.inverted_projected_pub = rospy.Publisher('/inverted_projected_feed', Image, queue_size=1)
        self.charbox_pub = rospy.Publisher('/charbox_feed', Image, queue_size=1)

        self.model = load_model("/home/fizzer/ros_ws/src/my_controller/reference/CNNs/ClueboardCNN.h5")


        # self.model = load_model("/ros_ws/src/my_controller/reference/CNNs/ClueboardCNN.keras")

    # def ok_callback(self, msg):
    #     if msg.data.lower() == "ok" and self.image_sub is None:
    #         rospy.loginfo("Start signal received. Beginning clueboard processing.")
    #         self.image_sub = rospy.Subscriber("/B1/rrbot/camera1/image_raw", Image, self.image_callback)

    def image_callback(self, msg):
        width, height = 600, 400
        rospy.loginfo("Received Image")

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([110, 200, 50])
        upper_blue = np.array([135, 255, 220])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(mask, encoding="mono8"))

        inverted = cv2.bitwise_not(mask)
        self.inverted_pub.publish(self.bridge.cv2_to_imgmsg(inverted, encoding="mono8"))

        edged = cv2.Canny(inverted, 30, 255)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        inner_contour = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:
                continue
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4 and area > max_area:
                max_area = area
                inner_contour = approx

        contour_feed = cv2.cvtColor(edged.copy(), cv2.COLOR_GRAY2BGR)

        if inner_contour is not None:
            cv2.drawContours(contour_feed, [inner_contour], -1, (0, 255, 0), 3)
            for pt in inner_contour:
                cv2.circle(contour_feed, tuple(pt[0]), 5, (0, 0, 255), -1)
            self.contour_pub.publish(self.bridge.cv2_to_imgmsg(contour_feed, encoding="bgr8"))

        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')
        src = sort_corners(inner_contour)
        M = cv2.getPerspectiveTransform(src, dst)
        projected = cv2.warpPerspective(frame, M, (width, height))

        self.projected_cb_pub.publish(self.bridge.cv2_to_imgmsg(projected, encoding="bgr8"))

        hsv_projected = cv2.cvtColor(projected, cv2.COLOR_BGR2HSV)
        lower_blue_projected = np.array([110, 100, 50])
        mask_projected = cv2.inRange(hsv_projected, lower_blue_projected, upper_blue)
        inverted_projected = cv2.bitwise_not(mask_projected)
        self.inverted_projected_pub.publish(self.bridge.cv2_to_imgmsg(inverted_projected, encoding="mono8"))

        # --- Character segmentation ---
        contours, _ = cv2.findContours(inverted_projected, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # --- Initial filter based on size ---
        raw_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if 15 < w < 100 and 30 < h < 100:
                raw_boxes.append((x, y, w, h, area))

        # Sort boxes by area descending (bigger ones first)
        raw_boxes = sorted(raw_boxes, key=lambda b: b[4], reverse=True)

        final_boxes = []
        for candidate in raw_boxes:
            if all(not boxes_overlap(candidate, kept) for kept in final_boxes):
                final_boxes.append(candidate)

        # Strip area field after filtering
        word_boxes = [(x, y, w, h) for (x, y, w, h, _) in final_boxes]

        # --- Cluster boxes into top and bottom rows using KMeans ---
        y_centers = np.array([[y + h // 2] for (_, y, _, h) in word_boxes])
        kmeans = KMeans(n_clusters=2, random_state=0).fit(y_centers)
        labels = kmeans.labels_

        top_boxes = [box for box, label in zip(word_boxes, labels) if label == 0]
        bottom_boxes = [box for box, label in zip(word_boxes, labels) if label == 1]

        # Ensure top is actually above bottom
        if np.mean([y for (_, y, _, _) in top_boxes]) > np.mean([y for (_, y, _, _) in bottom_boxes]):
            top_boxes, bottom_boxes = bottom_boxes, top_boxes
        
        top_boxes = sorted(top_boxes, key=lambda b: b[0])
        bottom_boxes = sorted(bottom_boxes, key=lambda b: b[0])
        
        # clue_boxes = [box for box, label in zip(word_boxes, labels) if label == 0]
        # value_boxes = [box for box, label in zip(word_boxes, labels) if label == 1]

        vis_char = cv2.cvtColor(inverted_projected.copy(), cv2.COLOR_GRAY2BGR)

        for (x, y, w, h) in top_boxes:
            cv2.rectangle(vis_char, (x, y), (x + w, y + h), (0, 255, 0), 2)  # green for clue

        for (x, y, w, h) in bottom_boxes:
            cv2.rectangle(vis_char, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue for value

        self.charbox_pub.publish(self.bridge.cv2_to_imgmsg(vis_char, encoding="bgr8"))


        clue_crops = get_letter_crops(top_boxes, width, height, inverted_projected)
        value_crops = get_letter_crops(bottom_boxes, width, height, inverted_projected)

        clue_nn_input = prepare_for_nn(clue_crops)
        value_nn_input = prepare_for_nn(value_crops)

        clue_predictions = self.model.predict(clue_nn_input)
        value_predictions = self.model.predict(value_nn_input)

        clue_chars = decode_predictions(clue_predictions, class_names)
        value_chars = decode_predictions(value_predictions, class_names)

        frame_clue_predictions.append(clue_chars)
        frame_value_predictions.append(value_chars)

        if len(frame_clue_predictions) >= NUM_FRAMES:
            clue_result = vote_across_frames(frame_clue_predictions)
            value_result = vote_across_frames(frame_value_predictions)

            rospy.loginfo(f"Clue Type:  {clue_result}")
            rospy.loginfo(f"Clue Value: {value_result}")

            frame_clue_predictions.clear()
            frame_value_predictions.clear()


def sort_corners(pts):
    pts = pts.reshape(4, 2)
    sum_pts = pts.sum(axis=1)
    diff_pts = np.diff(pts, axis=1)

    top_left = pts[np.argmin(sum_pts)]
    bottom_right = pts[np.argmax(sum_pts)]
    top_right = pts[np.argmin(diff_pts)]
    bottom_left = pts[np.argmax(diff_pts)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')


def get_letter_crops(boxes, width, height, img):
    letter_crops = []
    for i, (x, y, w, h) in enumerate(boxes):
        if x < 0 or y < 0 or x + w > width or y + h > height:
            print(f"Skipping invalid box at index {i}: {(x, y, w, h)}")
            continue

        letter = img[y:y+h, x:x+w]
        if letter.size == 0:
            print(f"Skipping empty crop at index {i}")
            continue

        resized = cv2.resize(letter, (32, 32))
        letter_crops.append(resized)
    return letter_crops

# --- Remove nested/overlapping boxes ---
def boxes_overlap(box1, box2, threshold=0.5):
    x1, y1, w1, h1, _ = box1
    x2, y2, w2, h2, _ = box2

    # Calculate overlap
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    if inter_area == 0:
        return False

    # Overlap ratio with respect to smaller box
    smaller_area = min(w1 * h1, w2 * h2)
    return inter_area / smaller_area > threshold

def prepare_for_nn(crops):
    valid = []
    for i, crop in enumerate(crops):
        if crop.shape != (32, 32):
            rospy.logwarn(f"Skipping invalid crop at index {i} with shape {crop.shape}")
            continue
        valid.append(crop)

    crops = np.array(valid).astype("float32") / 255.0
    return crops.reshape(-1, 32, 32, 1)

def decode_predictions(predictions, class_names):
    predicted_indices = np.argmax(predictions, axis=1)
    return [class_names[i] for i in predicted_indices]


def vote_across_frames(pred_lists):
    final_chars = []
    for i in range(len(pred_lists[0])):
        chars = [frame[i] for frame in pred_lists if len(frame) > i]
        vote = Counter(chars).most_common(1)[0][0]
        final_chars.append(vote)
    return "".join(final_chars)


def start_controller():
    cb_detector = ClueBoardDetector()
    rospy.spin()


if __name__ == "__main__":
    start_controller()