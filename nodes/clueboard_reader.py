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

NUM_FRAMES = 5
frame_clue_predictions = []
frame_value_predictions = []
class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


class ClueBoardDetector:
    def __init__(self):
        rospy.init_node("clue_board_detector", anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/B1/rrbot/camera1/image_raw", Image, self.image_callback)

        self.mask_pub = rospy.Publisher('/masked_feed', Image, queue_size=1)
        self.inverted_pub = rospy.Publisher('/inverted_feed', Image, queue_size=1)
        self.contour_pub = rospy.Publisher('/contoured_feed', Image, queue_size=1)
        self.projected_cb_pub = rospy.Publisher('/projected_cb_feed', Image, queue_size=1)

        self.model = CharNNModel()  # implement separately
        self.model.load_state_dict(torch.load(
            "/ros_ws/src/my_controller/models/charNNModel.pth",
            map_location=torch.device('cpu')
        ))
        self.model.eval()

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
        edged = cv2.Canny(inverted, 30, 255)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        min_area = float('inf')
        inner_contour = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:
                continue
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4 and area < min_area:
                min_area = area
                inner_contour = approx

        contour_feed = cv2.cvtColor(edged.copy(), cv2.COLOR_GRAY2BGR)
        if inner_contour is not None:
            cv2.drawContours(contour_feed, [inner_contour], -1, (0, 255, 0), 3)
            self.contour_pub.publish(self.bridge.cv2_to_imgmsg(contour_feed, encoding="bgr8"))

        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')
        src = sort_corners(inner_contour)
        M = cv2.getPerspectiveTransform(src, dst)
        projected = cv2.warpPerspective(frame, M, (width, height))

        hsv_projected = cv2.cvtColor(projected, cv2.COLOR_BGR2HSV)
        lower_blue_projected = np.array([110, 100, 50])
        mask_projected = cv2.inRange(hsv_projected, lower_blue_projected, upper_blue)
        inverted_projected = cv2.bitwise_not(mask_projected)
        self.inverted_pub.publish(self.bridge.cv2_to_imgmsg(inverted_projected, encoding="mono8"))

        # --- Character segmentation ---
        contours, _ = cv2.findContours(inverted_projected, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        word_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 15 < w < 100 and 30 < h < 100:
                word_boxes.append((x, y, w, h))

        if len(word_boxes) == 0:
            raise ValueError("No valid character boxes detected. Try adjusting contour filters.")

        y_centers = np.array([[y + h // 2] for (_, y, _, h) in word_boxes])
        kmeans = KMeans(n_clusters=2, random_state=0).fit(y_centers)
        labels = kmeans.labels_

        clue_boxes = [box for box, label in zip(word_boxes, labels) if label == 0]
        value_boxes = [box for box, label in zip(word_boxes, labels) if label == 1]

        clue_crops = get_letter_crops(clue_boxes, width, height, inverted_projected)
        value_crops = get_letter_crops(value_boxes, width, height, inverted_projected)

        clue_nn_input = prepare_for_nn(clue_crops)
        value_nn_input = prepare_for_nn(value_crops)

        clue_predictions = self.model(clue_nn_input)
        value_predictions = self.model(value_nn_input)

        clue_chars = decode_predictions(clue_predictions, class_names)
        value_chars = decode_predictions(value_predictions, class_names)

        frame_clue_predictions.append(clue_chars)
        frame_value_predictions.append(value_chars)

        value_boxes = sorted(value_boxes, key=lambda b: b[0])
        space_indices = []
        for i in range(1, len(value_boxes)):
            prev_x, _, prev_w, _ = value_boxes[i - 1]
            curr_x, _, _, _ = value_boxes[i]
            gap = curr_x - (prev_x + prev_w)
            if gap > 20:
                space_indices.append(i)

        if len(frame_clue_predictions) >= NUM_FRAMES:
            clue_result = vote_across_frames(frame_clue_predictions)
            value_result = vote_across_frames(frame_value_predictions, space_indices)

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

        resized = cv2.resize(letter, (54, 80))
        letter_crops.append(resized)
    return letter_crops


def prepare_for_nn(crops):
    tensor_batch = torch.stack([
        torch.tensor(crop, dtype=torch.float32).unsqueeze(0) / 255.0
        for crop in crops
    ])
    return tensor_batch


def decode_predictions(predictions, class_names):
    predicted_indices = torch.argmax(predictions, dim=1)
    return [class_names[i] for i in predicted_indices]


def vote_across_frames(pred_lists, space_indices=None):
    final_chars = []
    for i in range(len(pred_lists[0])):
        chars = [frame[i] for frame in pred_lists if len(frame) > i]
        vote = Counter(chars).most_common(1)[0][0]
        final_chars.append(vote)

    if space_indices:
        output = ""
        for i, char in enumerate(final_chars):
            if i in space_indices:
                output += " "
            output += char
        return output
    else:
        return "".join(final_chars)


def start_controller():
    cb_detector = ClueBoardDetector()
    rospy.spin()


if __name__ == "__main__":
    start_controller()
