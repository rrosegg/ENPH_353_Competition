#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import cv2
from sensor_msgs.msg import Image # Is this needed?
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import json  # for saving config from GUI


'''
TODO: Add state machine code s.t. when a pink line is crossed, we switch into a separate routine. 

States:     Paved_driving    Unpaved_driving     Offroad_driving     Mountain_driving

Road driving uses PID and object detection to avoid pedestrian and car while expecting to detect 3 signs.
Unpaved driving uses only PID with pit of despair in mind. Expects to read 3 signs.
Offroad driving is *either hard coded or following yoda. 
Mountain driving is purely well tuned, safe PID that is fast enough to get us up the mountain.

After reading the final sign, we publish the clue and stop driving.

*Use the red truck to align at the end, and consider preporatory alignment before hard coding section.

'''

class PID_control:

    # Define drive states
    STATE_PAVED    = 0
    STATE_UNPAVED  = 1
    STATE_OFFROAD  = 2
    STATE_MOUNTAIN = 3
    AVOID_YODA     = 444
    CP_ICE_SLICK   = 5
    CHILLIN        = 6

    INTEGRAL_CEILING = 100

    def __init__(self):
        rospy.init_node('topic_publisher', anonymous=True)

        # ROS Setup
        self.pub_cmd = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)
        self.pub_cb_detected = rospy.Publisher('/cb_detected', String, queue_size=1)
        self.pub_score = rospy.Publisher('/score_tracker', String, queue_size=1)
        ''' Remove the image subscriber initialization here, it's causing issues.'''
        # self.image_sub = rospy.Subscriber("/B1/rrbot/camera1/image_raw", Image, self.process_image) # Make this Low-Res
        # Is the correct path to the above actually this? "/B1/rrbot/camera1/image_raw" -- no hell nah
        # Old path: /robot/camera1/image_raw
        self.bridge = CvBridge()

        # PID Control Parameters
        self.Kp = 0.1     # Proportional gain  
        self.Ki = 0.0    # Integral gain  ---- 0.001
        self.Kd = 0.0     # Derivative gain  ---- 0.002
        self.prev_error = 0
        self.integral = 0

        # Speed Limits
        self.maxspeed = 2.0 # 1.0 works
        self.reducedspeed = 1.0 # 0.5 works

        # State Control (initialized for first paved section)
        self.state = 0 # Update as we cross pink lines
        self.obj_detection = True
        self.thresh_val = 240

        # Lane detection / Road stuff
        self.pinkpix = 0 # Number of pink pixels seen in the frame
        self.consec_pink_frames = 0
        self.consec_pinkless = 0

        # Misc
        self.lastpix = 0 # For use in road detection 
        # Wait for connections to publishers (necessary?)
        rospy.sleep(0.5) # 1.0?

        '''
        Some GUI stuff:
        '''
        ## Delete this:
        # self.window_name = "Tuner"
        # cv2.namedWindow(self.window_name)

        # GUI Trackbar Stuff
        cv2.namedWindow("Tuner")
        cv2.createTrackbar("Kp x1000", "Tuner", int(self.Kp * 1000), 1000, lambda x: None)
        cv2.createTrackbar("Ki x1000", "Tuner", int(self.Ki * 1000), 1000, lambda x: None)
        cv2.createTrackbar("Kd x1000", "Tuner", int(self.Kd * 1000), 1000, lambda x: None)
        cv2.createTrackbar("Max Speed x100", "Tuner", int(self.maxspeed * 100), 500, lambda x: None)
        cv2.createTrackbar("Reduced Speed x100", "Tuner", int(self.reducedspeed * 100), 500, lambda x: None)
        cv2.createTrackbar("Threshold", "Tuner", self.thresh_val, 255, lambda x: None)
        cv2.createTrackbar("Sim State", "Tuner", self.state, 3, lambda x: None)

        

    # Returns correction value: u(t) = Kp*err + Ki*integral(err) + Kp*d/dt(err) 
    def pid_control(self, error):
        self.integral += error # Sums error
        if abs(self.integral) > self.INTEGRAL_CEILING:
            self.integral = self.INTEGRAL_CEILING if self.integral > 0 else (-1*self.INTEGRAL_CEILING)
        derivative = error - self.prev_error # Difference term for sign
        self.prev_error = error # Update error
        return (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)


    ### Helper function counting colored pix for cd_detector() and scan_pink()
    def count_colorpix(self, img, color_name):
        """Return number of pixels matching the given color in the image."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        rgb = img

        if color_name == "blue":
            lower = np.array([100, 150, 50])
            upper = np.array([130, 255, 255]) ## Play with these
        elif color_name == "pink":
            # lower = np.array([140, 50, 50])
            # upper = np.array([170, 255, 255])
            # ## 
            lower = np.array([290, 90, 90])
            upper = np.array([310, 101, 101]) # 300,100,100 is the color!!

            ## Pink is 255,0,255
        elif color_name == "red":
            lower = np.array([250,0,0])
            upper = np.array([256,0,0])
        else:
            rospy.logwarn(f"Unknown color requested: {color_name}")
            return 0

        # mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.inRange(rgb, lower, upper)
        pixel_count = cv2.countNonZero(mask)

        # Optional: show mask for debugging
        # cv2.imshow(f"{color_name} mask", mask)

        return pixel_count
 

    # Scan for blue and publish to cb_detected
    def cb_detector(self, cv_image):
        min_blue = 200 # At least 200 bluepix to trigger "cb detected -> yes"
        bluepix = self.count_colorpix(cv_image, "blue")
        cb_detected = 'yes' if (bluepix > min_blue) else 'no'
        self.pub_cb_detected.publish(cb_detected)

    def scan_pink(self, roi):
        '''Existing code could potentially double update:'''
        current_pink = self.count_colorpix(roi, "pink")

        if current_pink > 0:
            self.consec_pink_frames += 1
            # self.consec_pinkless = 0
        else:
            if (self.consec_pink_frames > 10):
                self.state += 1
                self.update_state()    
                print("STATE UPDATED: NOW IN STATE ", self.state)

            self.consec_pink_frames = 0
            # self.consec_pinkless += 1

        self.pinkpix = current_pink # Update at the end.

    def update_state(self):
        # Update internal params based on current terrain state
        if self.state == self.STATE_PAVED:
            rospy.loginfo("Switched to: PAVED")
            self.Kp = 0.05
            self.Ki = 0.001
            self.Kd = 0.002
            self.thresh_val = 240
            self.maxspeed = 1.5
            self.reducedspeed = 0.5
            self.obj_detection = True

        elif self.state == self.STATE_UNPAVED:
            rospy.loginfo("Switched to: UNPAVED")
            # self.Kp = 0.04
            # self.Ki = 0.002
            # self.Kd = 0.003
            '''Troubleshooting:'''
            self.Kp = 1.0
            self.Ki = 1.0
            self.Kd = 1.0
            self.thresh_val = 170
            self.maxspeed = 1.2
            self.reducedspeed = 0.4
            self.obj_detection = False

        elif self.state == self.STATE_OFFROAD:
            rospy.loginfo("Switched to: OFFROAD")
            self.obj_detection = False
            # Insert logic for follow-yoda or hardcoded path
            self.handle_offroad()

        elif self.state == self.STATE_MOUNTAIN:
            rospy.loginfo("Switched to: MOUNTAIN")
            self.Kp = 0.03
            self.Ki = 0.001
            self.Kd = 0.004
            self.thresh_val = 235
            self.maxspeed = 1.0
            self.reducedspeed = 0.3
            self.obj_detection = False

        else:
            rospy.logwarn(f"Unknown state: {self.state}")

            

    def handle_offroad(self):
        pass
        ''' Avoid baby yoda at all costs. '''


    '''
    The next few functions are for telemetry purposes!
    '''
    def maketelemetry(self, thresh, y_coords):
        img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        thickness, blue = 2, (255,0,0)
        for y in y_coords:
            start, stop = (0, y), (img.shape[1], y)
            cv2.line(img, start, stop, blue, thickness)
        return img



    ### GUI UPDATER
    def update_tunables(self):
        self.Kp = cv2.getTrackbarPos("Kp x1000", "Tuner") / 1000.0
        self.Ki = cv2.getTrackbarPos("Ki x1000", "Tuner") / 1000.0
        self.Kd = cv2.getTrackbarPos("Kd x1000", "Tuner") / 1000.0
        self.maxspeed = cv2.getTrackbarPos("Max Speed x100", "Tuner") / 100.0
        self.reducedspeed = cv2.getTrackbarPos("Reduced Speed x100", "Tuner") / 100.0
        self.thresh_val = cv2.getTrackbarPos("Threshold", "Tuner")

        ## CHECK THAT THE LOGIC ON THIS IS RIGHT:
        new_state = cv2.getTrackbarPos("Sim State", "Tuner")
        if new_state != self.state:
            rospy.loginfo(f"Simulated state changed: {self.state} → {new_state}")
            self.state = new_state
            self.update_state()  # react to state change if needed

    def save_settings(self, filename="pid_config.json"):
        # Try not to call this anywhere yet. Only once PID is more thoroughly tuned.
        config = {
            "Kp": self.Kp,
            "Ki": self.Ki,
            "Kd": self.Kd,
            "maxspeed": self.maxspeed,
            "reducedspeed": self.reducedspeed,
            "thresh_val": self.thresh_val,
            "state": self.state
        }
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)
        rospy.loginfo(f"Settings saved to {filename}")





    # def detect_lane_blobs(self, thresh):
    #     """
    #     Detect two main white blobs (lane lines) and return their x-centers.
    #     """
    #     # Set up SimpleBlobDetector parameters.
    #     params = cv2.SimpleBlobDetector_Params()
    #     params.filterByArea = True
    #     params.minArea = 100    # Adjust based on resolution
    #     params.maxArea = 50000  # Prevent huge noise blobs
    #     params.filterByCircularity = False
    #     params.filterByConvexity = False
    #     params.filterByInertia = False
    #     params.blobColor = 255  # Looking for white blobs in binary image

    #     detector = cv2.SimpleBlobDetector_create(params)

    #     # Detect blobs
    #     keypoints = detector.detect(thresh)

    #     # Sort blobs by size (descending)
    #     keypoints = sorted(keypoints, key=lambda k: k.size, reverse=True)

    #     if len(keypoints) < 2:
    #         return None, thresh  # Not enough blobs found

    #     # Get the two largest blobs
    #     left_blob = keypoints[0].pt  # (x, y)
    #     right_blob = keypoints[1].pt

    #     # if (right_blob[0] - left_blob[0]) < 10:
    #     #     for n in keypoints:
    #     #         right_blob = keypoints[n]
    #     #         if (right_blob[0] - left_blob[0]) > 10:
    #     #             break
    #             ### If we never break then find out how to handle this.
                    

    #     # Optional: draw blobs
    #     out_img = cv2.drawKeypoints(thresh, keypoints[:2], None, (0, 0, 255),
    #                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #     cv2.imshow("blob debugger", out_img)

    #     return (int(left_blob[0]), int(right_blob[0]))





    '''
    Implement the main image processing control loop below and figure out how it fits in 
    with the high level running loop (and calling blue detector guy).
    '''

    def process_image(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        ### Preprocessing (grayscale, resize, blur, binary threshold)
        resized = cv2.resize(cv_image, (0,0), fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST) # INTER_AREA also an option
        roi = resized[int(resized.shape[0]*0.65):, :]  # bottom 35%
        height, width = roi.shape[:2]
        image_center = width // 2

        ## Preprocessing 
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) # Reduce bandwidth quickly
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, self.thresh_val, 255, cv2.THRESH_BINARY)


        # Check for colors (update state and publish cb_detector here)
        self.cb_detector(resized)
        self.scan_pink(roi)


        """ Set up SimpleBlobDetector parameters. """
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 100    # Adjust based on resolution
        params.maxArea = 50000  # Prevent huge noise blobs
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.blobColor = 255  # Looking for white blobs in binary image

        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs then sort by descending size
        keypoints = detector.detect(thresh)
        keypoints = sorted(keypoints, key=lambda k: k.size, reverse=True)

        twist = Twist()

        if len(keypoints) < 2:
             ### EITHER ONE OR NO BLOBS FOUND, START SPINNING LIKE CRAZY AFTER A FEW FRAMES
            if len(keypoints) == 1:
                '''Find out where the blob is and assume that its on the correct side. Create error and trigger pid'''
                single_blob = keypoints[0].pt
                single_x = int(single_blob[0])

                if single_x < image_center: 
                    right_x = width - 1     
                    left_x = single_x   # Our one blob is the left lane line
                else:
                    left_x = 0      
                    right_x = single_x  # Our one blob is the right lane line
                
                lane_center = (left_x + right_x) // 2
                error = image_center - lane_center
                twist.angular.z = self.pid_control(error)
                twist.linear.x = self.maxspeed if abs(error) < 10 else self.reducedspeed
                rospy.loginfo(f"Blob error: {error}, Z: {twist.angular.z:.2f}")

            else:
                rospy.logwarn("No blobs detected, PID abandoned; spinning")
                twist.angular.z = 1.0
                twist.linear.x = 0.0

        else:
            # Get the two largest blobs
            left_blob = keypoints[0].pt  # (x, y)
            right_blob = keypoints[1].pt

            if abs(right_blob[0] - left_blob[0]) < width * 0.3:
                for n in range(len(keypoints)):
                    right_blob = keypoints[n].pt
                    if abs(right_blob[0] - left_blob[0]) > width * 0.3:
                        break
                    else:
                        right_blob = None
                        ### If we never break then right blob is none, handle single case.
                        
            if right_blob is None:
                single_blob = keypoints[0].pt
                single_x = int(single_blob[0])

                if single_x < image_center: 
                    right_x = width - 1     
                    left_x = single_x   # Our one blob is the left lane line
                else:
                    left_x = 0      
                    right_x = single_x  # Our one blob is the right lane line
                
            else: 
                blobs = (int(left_blob[0]), int(right_blob[0]))
                left_x, right_x = sorted(blobs)

            lane_center = (left_x + right_x) // 2
            error = image_center - lane_center
            twist.angular.z = self.pid_control(error)
            twist.linear.x = self.maxspeed if abs(error) < 10 else self.reducedspeed
            rospy.loginfo(f"Blob error: {error}, Z: {twist.angular.z:.2f}")
                


            # Optional: draw blobs
            out_img = cv2.drawKeypoints(thresh, keypoints[:2], None, (0, 0, 255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("blob debugger", out_img)
            cv2.waitKey(1)

        


        




        # blobs = self.detect_lane_blobs(thresh)

        # twist = Twist()
        # if blobs is None:
        #     rospy.logwarn("Blob detection failed — fallback to white mean")
        #     white_pixels = np.where(thresh > 0)
        #     if white_pixels[1].size > 0:
        #         road_center = int(np.mean(white_pixels[1]))
        #         error = (thresh.shape[1] // 2) - road_center
        #         twist.angular.z = self.pid_control(error)
        #         twist.linear.x = self.reducedspeed
        #     else:
        #         twist.angular.z = 1.0
        #         twist.linear.x = 0.0
        # else: ### CHANGE TO ELIF
        #     left_x, right_x = sorted(blobs)
        #     lane_center = (left_x + right_x) // 2
        #     image_center = thresh.shape[1] // 2
        #     error = image_center - lane_center
        #     twist.angular.z = self.pid_control(error)
        #     twist.linear.x = self.maxspeed if abs(error) < 10 else self.reducedspeed
        #     rospy.loginfo(f"Blob error: {error}, Z: {twist.angular.z:.2f}")

        
        
        ### If one blob, check which half of screen it is on. Assume that this is right and set error to correct to the right.
        ### If no blobs, something fucked up. (Check intervals for expected behavior ?)


        '''

        twist = Twist()
        the_pocket = 5 # if our error < the_pocket=5 pixels then we drive fast

        # ## Attempt to FORCE fallback here:
        white_pixels = np.where(thresh > 0)
        if white_pixels[1].size > 0:
            road_center = int(np.mean(white_pixels[1]))
            error = image_center - road_center

            twist.angular.z = self.pid_control(error)
            twist.linear.x = self.maxspeed if abs(error) < the_pocket else self.reducedspeed
            rospy.loginfo(f"Error: {error}, Angular Z: {twist.angular.z:.2f}, Linear X: {twist.linear.x:.2f}")

        else:
            twist.angular.z = 1.0
            twist.linear.x = 0.05
            print("No road at all we're spinning")

        '''

        self.pub_cmd.publish(twist)
        
        
        ## Scan horizontal rows:
        y_coords = [int(height * w) for w in (0.8, 0.65, 0.4)]    # Edit weights to change row heights
        '''
        midpoints = []

        for y in y_coords:
            row = thresh[y, :]
            white = np.where(row == 255)[0]

            for i in range(len(white) - 1):
                gap = white[i + 1] - white[i]
                if gap > (width*0.2):  # Wide enough gap → possible road center
                    left, right = white[i], white[i + 1]
                    midpoints.append((left + right) // 2)
                    break  # Use first valid gap only --> Is this accurate?

        twist = Twist()

        if not midpoints:
            rospy.logwarn("No midpoints found, attempting fallback strategy")
            # Try fallback: if any white line exists, steer based on its avg position
            white_pixels = np.where(thresh > 0)
            if white_pixels[1].size > 0:
                center_est = int(np.mean(white_pixels[1]))
                error = image_center - center_est
                twist.angular.z = self.pid_control(error)
                twist.linear.x = self.reducedspeed
                print(error)
            else: 
                # No white anywhere, stop & spin
                error = 10000 # Garbage value -- will never get thrown into pid integral term
                twist.angular.z = 1.0
                twist.linear.x = 0.0
        else:
            road_center = sum(midpoints) // len(midpoints)

            # ## Attempt to FORCE fallback here:
            # white_pixels = np.where(thresh > 0)
            # if white_pixels[1].size > 0:
            #     road_center = int(np.mean(white_pixels[1]))

            error = image_center - road_center
            twist.angular.z = self.pid_control(error)
            twist.linear.x = self.maxspeed if abs(error) < 5 else self.reducedspeed
            rospy.loginfo(f"Error: {error}, Angular Z: {twist.angular.z:.2f}, Linear X: {twist.linear.x:.2f}")
        
        # self.pub_cmd.publish(twist)
        '''

        '''
        Figure out additional object detection and secondary state-based control flow algorithms here:
        '''
        # if (state == 0):
            # TODO: look for crosswalk and switch to motion-detection mode. (Stop until safe)
            # Also figure out this control algorithm for the  


        # === Visualized Feedback ===
        telemetry = self.maketelemetry(thresh, y_coords)
        # cv2.imshow("Telemetry", telemetry)
        # cv2.imshow("raw", cv_image)
        # cv2.imshow("gray", gray)
        # cv2.imshow("blurred", blurred)
        # cv2.imshow("thresh", thresh)    # This gives us road values -> doesn't 
        ''' Can also show blobs and color masks in respective helper functions. '''
        cv2.waitKey(1)

        '''Save PID coeffs here (but don't use this yet)'''
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('s'):
        #     self.save_settings()



if __name__ == '__main__':
    Drive = PID_control()
    rate = rospy.Rate(30)  # 30 Hz -- 30 fps but not necessarily

    while not rospy.is_shutdown():
        try:
            data = rospy.wait_for_message("/B1/rrbot/camera1/image_raw", Image, timeout=5) # Path should be correct
            Drive.process_image(data)
        except rospy.ROSException:
            # pass
            rospy.logwarn("No image received in 5s, retrying...")
        
        rate.sleep()
