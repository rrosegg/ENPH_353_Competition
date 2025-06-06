#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import cv2
from sensor_msgs.msg import Image # Is this needed?
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import json  # for saving config from GUI
import time # For troubleshooting

# For teleporting during TS
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState


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
    STATE_PAVED     = 0
    STATE_UNPAVED   = 1
    STATE_OFFROAD   = 2
    STATE_MOUNTAIN  = 3
    LAUNCH_STATE    = 4

    # State machine steps!
    PED_XING        = 1   # dont hit pedestrian.
    TRUCK_STOP      = 2 # wait for the truck then go.
    DONT_SWIM       = 3
    AVOID_YODA      = 4
    MOUNTAINEERING  = 5

    FCHECK_INTERVAL = 4

    

    def __init__(self):
        rospy.init_node('topic_publisher', anonymous=True)

        # ROS Setup
        self.pub_cmd = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)
        self.pub_cb_detected = rospy.Publisher('/clueboard_detected', String, queue_size=1)
        self.pub_score = rospy.Publisher('/score_tracker', String, queue_size=1)
        self.cb_read_sub = rospy.Subscriber('cb_read', String, queue_size=1)
        self.bridge = CvBridge()

        # Wait for connection to publishers
        rospy.sleep(1) # 0.5?
        
        # Quickstop
        
        # stop = Twist()
        # stop.linear.x = 0.1
        # stop.angular.z = 0.0
        # self.pub_cmd.publish(stop)
        # # self.pub_cb_detected.publish('yes')
        # rospy.sleep(1)

        self.pub_score.publish('Egg,pw,0,NA') #TODO send -1 when comp is done.

        # PID Control Parameters
        self.Kp = 0.13     # Proportional gain  
        self.Ki = 0.0    # Integral gain  ---- 0.001
        self.Kd = 0.02     # Derivative gain  ---- 0.002
        self.prev_error = 0
        self.integral = 0
        self.integral_cap = 0.5 # Down from 100

        # Speed Limits
        self.maxspeed = 0.1 # 2.0-1.0 works
        self.reducedspeed = 0.0 # 1.0-0.5 works

        # State Control (initialized for first paved section)
        self.state = self.STATE_PAVED # Update as we cross pink lines
        self.obj_detection = True
        self.thresh_val = 240
        

        """ Set up SimpleBlobDetector parameters. """
        self.params = cv2.SimpleBlobDetector_Params()
        self.params.filterByArea = True
        # self.params.minArea = 200    # Adjust based on resolution   (Used to be 100, then 50)
        # self.params.maxArea = 20000  # Prevent huge noise blobs (used to be 5000, then 15000, now 20000)
        self.params.minArea = 200
        self.params.maxArea = 200000  # Prevent huge noise blobs (used to be 5000, then 15000, now 20000)
        self.params.filterByCircularity = False
        self.params.filterByConvexity = False
        self.params.filterByInertia = False ## True??
        self.params.blobColor = 255  # Looking for white blobs in binary image

        # STATE MACHINE
        self.sm = self.PED_XING

        self.first_valid_frame_received = True


        # Lane detection / Road stuff
        self.consec_pink_frames = 0
        self.consec_pinkless = 0

        self.yoda_was_seen = False
        self.pause_for_truck = False
        self.truckpix = 100000 # Assume for 800x800 that we want max 400x400 blob

        self.fcount = 0 # Only look for colors every 5th frame

        self.min_cb_area = 800 #TODO: TUNE THIS

        self.cb_detected = False


        self.motion_threshold = 100 

        # Misc
        self.lastpix = 0 # For use in road detection 
        
        


        # Debugging params
        self.lasttime = time.time()
        self.show_time = True
        self.debug = True
        self.autopilot = 1  # 1 for on, 0 for off

        '''
        Some GUI stuff:
        '''
        ## Delete this:
        # self.window_name = "Tuner"
        # cv2.namedWindow(self.window_name)

        # GUI Trackbar Stuff 
        if self.debug:
            cv2.namedWindow("Tuner")
            cv2.createTrackbar("Kp x1000", "Tuner", int(self.Kp * 1000), 1000, lambda x: None)
            cv2.createTrackbar("Ki x1000", "Tuner", int(self.Ki * 1000), 1000, lambda x: None)
            cv2.createTrackbar("Kd x1000", "Tuner", int(self.Kd * 1000), 1000, lambda x: None)
            cv2.createTrackbar("Max Speed x100", "Tuner", int(self.maxspeed * 100), 500, lambda x: None)
            cv2.createTrackbar("Reduced Speed x100", "Tuner", int(self.reducedspeed * 100), 500, lambda x: None)
            cv2.createTrackbar("Threshold", "Tuner", self.thresh_val, 255, lambda x: None)
            cv2.createTrackbar("Sim State", "Tuner", self.state, 3, lambda x: None)
            cv2.createTrackbar("Autopilot", "Tuner", self.autopilot, 1, lambda x: None)
            cv2.moveWindow("Tuner", 1500, 50)  # (x=100px from left, y=200px from top)

        self.update_state() ## Call this to make sure everything aligns
        
        

    # Returns correction value: u(t) = Kp*err + Ki*integral(err) + Kp*d/dt(err) 
    def pid_control(self, error):
        self.integral += error # Sums error
        # if abs(self.integral) > self.integral_cap:
        #     self.integral = self.integral_cap if self.integral > 0 else (-1*self.integral_cap)
        ##  Chat recommendation:
        self.integral = max(min(self.integral, self.integral_cap), -self.integral_cap)
        derivative = error - self.prev_error # Difference term for sign
        self.prev_error = error # Update error
        return (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)


    ### Helper function counting colored pix for cd_detector() and scan_pink()
    def count_colorpix(self, img, color_name):
        """Return number of pixels matching the given color in the image."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # rgb = img

        if color_name == "blue":
            lower, upper = np.array([85, 150, 50]), np.array([130, 256, 256]) # HSV = (240,100,100) Play with these! -> Work for all but two
            # lower, upper = np.array([100, 70, 50]), np.array([250, 255, 220]) # HSV vals from Rose            
            # lower, upper = np.array([230, 0, 0]), np.array([256,20,20]) # BGR = (255,0,0)?????????
        elif color_name == "pink":
            lower, upper = np.array([150, 100, 100]), np.array([256, 256, 256]) # HSV = (300,100,100) -> 255,255,255    (150,100,100)
            # lower, upper = np.array([250, 0, 250]), np.array([256, 0, 256]) # BGR = (255,0,255)
        elif color_name == "red":
            lower, upper = np.array([0,200,200]), np.array([13,256,256]) # HSV = (0,100,100) on 300,100,100 scale
            # lower, upper = np.array([0,0,250]), np.array([0,0,256]) # BGR = (0,0,255)
        elif color_name == "silver":
            lower, upper = np.array([0,0,50]), np.array([0,0,75]) # HSV = (0,0,50-65)
        else:
            rospy.logwarn(f"Unknown color requested: {color_name}")
            return 0

        mask = cv2.inRange(hsv, lower, upper)
        # mask = cv2.inRange(rgb, lower, upper)
        pixel_count = cv2.countNonZero(mask)

        # # Optional: show mask for debugging
        # cv2.imshow(f"{color_name} mask", mask)
        # cv2.waitKey(1)

        return pixel_count
 

    # Scan for blue and publish to cb_detected
    def cb_detector(self, cv_image):
        # min_blue = 50 # At least 200 bluepix to trigger "cb detected -> yes"
        # bluepix = self.count_colorpix(cv_image, "blue")
        # cb_detected = 'yes' if (bluepix > min_blue) else 'no'
        # self.pub_cb_detected.publish(cb_detected)

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_blue, upper_blue = np.array([100, 70, 50]), np.array([250, 255, 220]) # Jackson's (empirical)
        # lower_blue, upper_blue = np.array([110, 200, 50]), np.array([135, 255, 255]) # original

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        inverted = cv2.bitwise_not(mask)
        edged = cv2.Canny(inverted, 30, 255)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        inner_contour = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 6000: continue 
            # 1000 too low, 10,000 too high.
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4 and area > max_area:
                max_area = area
                inner_contour = approx
            
        if inner_contour is not None and max_area > self.min_cb_area:
            rospy.loginfo("CB Visible, giving the ok to cb_reader node")
            cb_detected = "yes"
            self.cb_detected = True
        else: 
            cb_detected = "no"
            self.cb_detected = False
        self.pub_cb_detected.publish(cb_detected)

    def truckcheck(self, image):
        if self.count_colorpix(image, "silver") > 1000: #wow feels bad
            # 4k resolution has area of 12,000,000; shoot for 1/8 of that (100,000 for 800x800)
            self.pause_for_truck = True
            stop = Twist()
            stop.linear.x = 0.0
            stop.angular.z = 0.0
            self.pub_cmd.publish(stop)
            print("Truck nearby, pausing")
        else: self.pause_for_truck = False


    def scan_red(self, cv_image):
        # Make a box or something if detected?
        current_red = self.count_colorpix(cv_image, "red")

        if(current_red > 2000 and self.sm == self.PED_XING):
        # if(current_red > 2000): # Troubleshooting
            self.motion_detector()
            self.sm = self.TRUCK_STOP
        else:
            # We're looking for the second truck (yoda's) I presume.
            pass

    '''TODO: Patch this'''
    def motion_detector(self):
        stop = Twist()
        stop.linear.x = 0.0
        stop.angular.z = 0.0
        self.pub_cmd.publish(stop)
        time.sleep(0.1)

        try:
            prev_img = rospy.wait_for_message("/B1/rrbot/camera1/image_raw", Image, timeout=1.0)
            prev = self.bridge.imgmsg_to_cv2(prev_img, "bgr8")

        except CvBridgeError as e:
            print(e)

        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        waiting = True
        
        while waiting:
            try:
                img = rospy.wait_for_message("/B1/rrbot/camera1/image_raw", Image, timeout=1.0)
                now = self.bridge.imgmsg_to_cv2(img, "bgr8")
            except CvBridgeError as e:
                print(e)
            
            gray = cv2.cvtColor(now, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, gray) # Compute absolute difference between current and previous frame
            _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY) # Threshold the difference to get motion areas

            # Optional: Dilate to fill gaps
            # motion_mask = cv2.dilate(motion_mask, None, iterations=2)

            # Show motion mask
            if self.debug: 
                cv2.imshow("Motion", motion_mask)
                cv2.waitKey(1)

            # Update previous frame
            prev_gray = gray.copy()
            white_pixels = np.count_nonzero(motion_mask)

            if (self.sm == self.PED_XING):
                if white_pixels < 300: 
                    waiting = False
                    self.sm = self.TRUCK_STOP
                # Allow it to keep driving and go.
            elif (self.sm == self.TRUCK_STOP):
                continue # Just drive
            else:
                if self.yoda_was_seen:  
                    if white_pixels < 30: 
                        waiting = False 
                        self.handle_offroad()
                else: 
                    if white_pixels > 600:
                        self.yoda_was_seen = True

                ###Assume yoda


            #     ### ASSUME YODA AND CAR????
            #     if white_pixels > self.motion_threshold:
            #         waiting = False
            #         time.sleep(3)
            #     elif white_pixels < 10:
            #         waiting = False
            #         time.sleep(1)

                
            #     waiting = True
            #     print("Waiting!")

        # THEN EXECUTE WHATEVER CODE AFTER!


    def scan_pink(self, roi):
        '''Existing code could potentially double update:'''
        current_pink = self.count_colorpix(roi, "pink")

        if current_pink > 0:
            if current_pink > 100: 
                self.consec_pink_frames += 1
            self.consec_pinkless = 0 # Comment this out?
        else:
            if (self.consec_pink_frames > 3):
                self.state += 1
                self.update_state()    
                print(f"STATE UPDATED: NOW IN STATE {self.state}")

            self.consec_pink_frames = 0
            self.consec_pinkless += 1
            if self.consec_pinkless > 1: 
                self.big_pink_seen = False # Stop checking every second

    def update_state(self):
        # Update internal params based on current terrain state
        if self.state == self.STATE_PAVED:
            rospy.loginfo("Switched to: PAVED")
            self.Kp = 0.13
            self.Ki = 0.005 # 0.001
            self.Kd = 0.10 # 0.002 -> 0.02
            self.thresh_val = 240
            self.maxspeed = 2.2 # 1.5
            self.reducedspeed = 1.0 # 0.5
            self.obj_detection = True
            self.autopilot = 1

        elif self.state == self.STATE_UNPAVED:
            rospy.loginfo("Switched to: UNPAVED")
            self.Kp = 0.1
            self.Ki = 0.005
            self.Kd = 0.1
            self.thresh_val = 175
            self.maxspeed = 1.5
            self.reducedspeed = 1.0
            self.obj_detection = False

            self.sm = self.DONT_SWIM # Stop checking for the truck

        elif self.state == self.STATE_OFFROAD:
            rospy.loginfo("Switched to: OFFROAD")
            self.obj_detection = False
            self.sm = self.AVOID_YODA # Just for consistency with micro-state machine
            
            # Insert logic for follow-yoda or hardcoded path
            if not self.obj_detection: self.tunnel_teleport()
            # Just teleport if we're not object detecting

        elif self.state == self.STATE_MOUNTAIN:
            rospy.loginfo("Switched to: MOUNTAIN")
            self.Kp = 0.1
            self.Ki = 0.0
            self.Kd = 0.02
            self.thresh_val = 235
            self.maxspeed = 1.0
            self.reducedspeed = 0.3
            self.obj_detection = False
        
        elif self.state == self.LAUNCH_STATE:
            self.autopilot = 0
            stopped = Twist()
            stopped.linear.x = 0.8
            stopped.angular.z = 0
            self.pub_cmd.publish(stopped)
            rospy.sleep(2)
            # self.state = self.STATE_PAVED
            # self.update_state()


        else:
            rospy.logwarn(f"Unknown state: {self.state}")


        stop = Twist()
        stop.linear.x = 0.0
        stop.angular.z = 0.0
        self.pub_cmd.publish(stop)


        if self.debug:
            cv2.setTrackbarPos("Kp x1000", "Tuner", int(self.Kp * 1000))
            cv2.setTrackbarPos("Ki x1000", "Tuner", int(self.Ki * 1000))
            cv2.setTrackbarPos("Kd x1000", "Tuner", int(self.Kd * 1000))
            cv2.setTrackbarPos("Max Speed x100", "Tuner", int(self.maxspeed * 100))
            cv2.setTrackbarPos("Reduced Speed x100", "Tuner", int(self.reducedspeed * 100))
            cv2.setTrackbarPos("Threshold", "Tuner", self.thresh_val)
            cv2.setTrackbarPos("Sim State", "Tuner", self.state)
            cv2.setTrackbarPos("Autopilot", "Tuner", self.autopilot)


    def tunnel_teleport(self):
        self.teleport(self.STATE_MOUNTAIN)

        s3 = Twist()
        s3.angular.z = 1.6

        self.pub_cmd.publish(s3)
        rospy.sleep(1)
        
        stop = Twist()
        stop.linear.x = 0.0
        stop.angular.z = 0.0
        self.pub_cmd.publish(stop) # compilation error at comp
        rospy.sleep(1)
        
        s3.angular.z = 0.0
        s3.linear.x = 0.5
        self.pub_cmd.publish(s3)
        rospy.sleep(1)

        self.pub_cmd.publish(stop)
        rospy.sleep(2)
    
        self.pub_score.publish('Egg,pw,-1,NA') # End run.

        while(True): self.autopilot = 0   #infinite loop
        
        

    def handle_offroad(self):
        '''Hard coded routine'''
        stop = Twist()
        stop.linear.x = 0.0
        stop.angular.z = 0.0
        self.pub_cmd.publish(stop)
            
        '''
        routine = Twist()
        routine.linear.x = 0.5
        routine.angular.z = 0.7

        self.motion_detector() # Wait and be in loop
        #If no motion detected, send it? Or if some then wait and then send after hardcoded # of secs.

        # Exited loop, good to go.
        self.pub_cmd.publish(routine)
        rospy.sleep(1)

        routine.linear.x = 3.0
        routine.angular.z = 0.0
        self.pub_cmd.publish(routine)
        rospy.sleep(1)

        routine.linear.x = 1.0
        routine.angular.z = -0.5
        self.pub_cmd.publish(routine)
        rospy.sleep(1)

        routine.linear.x = 3.0
        routine.angular.z = 0.1
        self.pub_cmd.publish(routine)
        rospy.sleep(1)
        '''

        rospy.loginfo("Starting OFFROAD tracking...")
    
        self.pub_cmd.publish(stop)

        tracking_thing = True
        prev_error = 0

        while tracking_thing:
            twist = Twist()

            try:
                img = rospy.wait_for_message("/B1/rrbot/camera1/image_raw", Image, timeout=1.0)
                frame = self.bridge.imgmsg_to_cv2(img, "bgr8")
            except CvBridgeError as e:
                print(e)
                continue
            
            self.cb_detector(frame)
            if self.cb_detected:
                rospy.loginfo("CB Detected! Ending offroad tracking.")
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            if self.state == self.STATE_OFFROAD:
                lower_color = np.array([80, 90, 50])
                upper_color = np.array([130, 120, 120])
            else:
                # Define silver if needed here
                lower_color = np.array([80, 90, 50])
                upper_color = np.array([130, 120, 120])

            mask = cv2.inRange(hsv, lower_color, upper_color)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)

                if area < 800:
                    rospy.loginfo("Object too small, stopping")
                    self.pub_cmd.publish(stop)
                    continue

                x, y, w, h = cv2.boundingRect(largest)
                obj_center_y = y + h // 2

                error = -obj_center_y
                derivative = error - prev_error
                prev_error = error

                twist.angular.z = self.pid_control(error)
                twist.linear.x = self.maxspeed if abs(error) < 10 else self.reducedspeed

                self.pub_cmd.publish(twist)

                # Visual feedback
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Area: {area:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                rospy.loginfo("No object found. Stopping.")
                self.pub_cmd.publish(stop)

            if self.debug:
                cv2.imshow("Tracking", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break


        





        ''' Avoid baby yoda at all costs. '''


    '''
    The next few functions are for telemetry purposes! TODO: Make sure maketelemetry is never called
    '''
    def maketelemetry(self, thresh, y_coords):
        img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        thickness, blue = 2, (255,0,0)
        for y in y_coords:
            start, stop = (0, y), (img.shape[1], y)
            cv2.line(img, start, stop, blue, thickness)
        return img

    def spawn_position(self, position):
        msg = ModelState()
        msg.model_name = 'B1'

        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        msg.pose.orientation.x = position[3]
        msg.pose.orientation.y = position[4]
        msg.pose.orientation.z = position[5]
        msg.pose.orientation.w = position[6]

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(msg)
        except rospy.ServiceException:
            rospy.logerr("Service call failed")

    def teleport(self, state):
        pink_lines = [  
                        [ 0, 0, 0, 0, 0, 0, 0],                                 # Spawn pos (unknown, drop to origin)
                        [ 0.5,  -0.05,  0.05,   0.0, 0.0, 0.7071, 0.7071],      # First line (spawn not included)
                        [-3.9,   0.465, 0.05,   0.0, 0.0, 1.0,    0.0],         # Second line, skip this one
                        [-4.05, -2.3,   0.05,   0.0, 0.0, 0.0,    0.0]          # Third line
                ]

        if state > 3: state = 0

        position = pink_lines[state]
        self.spawn_position(position)



    ### GUI UPDATER
    def update_tunables(self):
        self.Kp = cv2.getTrackbarPos("Kp x1000", "Tuner") / 1000.0
        self.Ki = cv2.getTrackbarPos("Ki x1000", "Tuner") / 1000.0
        self.Kd = cv2.getTrackbarPos("Kd x1000", "Tuner") / 1000.0
        self.maxspeed = cv2.getTrackbarPos("Max Speed x100", "Tuner") / 100.0
        self.reducedspeed = cv2.getTrackbarPos("Reduced Speed x100", "Tuner") / 100.0
        self.thresh_val = cv2.getTrackbarPos("Threshold", "Tuner")
        self.autopilot = cv2.getTrackbarPos("Autopilot", "Tuner")

        ## CHECK THAT THE LOGIC ON THIS IS RIGHT:
        new_state = cv2.getTrackbarPos("Sim State", "Tuner")
        if new_state != self.state:
            rospy.loginfo(f"Simulated state changed: {self.state} → {new_state}")
            self.state = new_state
            self.update_state()  # react to state change if needed
            self.teleport(new_state) # teleport to new state if there was a change

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




    '''
    Implement the main image processing control loop below and figure out how it fits in 
    with the high level running loop (and calling blue detector guy).
    '''

    def process_image(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # if self.state == self.LAUNCH_STATE:
        #     '''Remove if needed'''
        #     # Wait until we see a CB or any valid blobs before starting
        #     self.cb_detector(self.bridge.imgmsg_to_cv2(data, "bgr8"))
        #     if self.cb_detected:
        #         rospy.loginfo("CB detected during launch. Switching to PAVED mode.")
        #         self.state = self.STATE_PAVED
        #         self.autopilot = 1
        #         self.update_state()
        #     else:
        #         # Optional: add a timeout fallback
        #         rospy.loginfo_throttle(2, "Waiting for CB to appear...")
        #         return
            

        ### Preprocessing (grayscale, resize, blur, binary threshold)
        resized = cv2.resize(cv_image, (0,0), fx=0.4, fy=0.4, interpolation=cv2.INTER_NEAREST) # INTER_AREA also an option
        roi = resized[int(resized.shape[0]*0.6):, :]  # bottom 40%
        height, width = roi.shape[:2]
        image_center = width // 2

        ## Preprocessing 
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) # Reduce bandwidth quickly
        blurred = cv2.GaussianBlur(gray, (3, 3), 0) # Formerly (5,5)
        _, thresh = cv2.threshold(blurred, self.thresh_val, 255, cv2.THRESH_BINARY)


        # Check for colors (update state and publish cb_detector here)

        self.cb_detector(cv_image)
        self.scan_pink(roi) # '''dont remove unless really needed for overhead'''
        if (self.sm == self.PED_XING or self.sm == self.AVOID_YODA): 
            self.scan_red(roi)
        # self.count_colorpix(roi, "red")
        if(self.sm == self.TRUCK_STOP):
            self.truckcheck(cv_image)
            # Listen for 3, if 3 published then once no blue visible
            # self.motion_detector()
            
            # self.sm +=1


            # Check is motion is on the right (and less than x number of pixels)
            # If that is true, rip it and go.



        # if self.big_pink_seen: scan_pink() # Start spamming it if we see it lots.
        '''
        If cb_detector finds a sign, it just sends a signal to thing that it's okay to look. If it doesnt find a sign, it will send "no"
        '''


        detector = cv2.SimpleBlobDetector_create(self.params)

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
                if self.fcount == 0: rospy.logwarn("No blobs detected, PID abandoned; spinning")
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
            if self.fcount == 0: rospy.loginfo(f"Blob error: {error}, Z: {twist.angular.z:.2f}")
        
        
        ### If one blob, check which half of screen it is on. Assume that this is right and set error to correct to the right.
        ### If no blobs, something fucked up. (Check intervals for expected behavior ?)


        # Only enable movement after first valid detection
        # if not self.first_valid_frame_received:
        #     if len(keypoints) >= 1:  # Set stricter condition if needed
        #         rospy.loginfo("First valid frame processed. Robot is now allowed to move.")
        #         self.first_valid_frame_received = True
        #     else:
        #         # Don't publish twist yet
        #         return

        if self.pause_for_truck: twist.linear.x, twist.angular.z = 0.0, 0.0

        # if self.autopilot == 1 and self.first_valid_frame_received: 
        #     self.pub_cmd.publish(twist)

        if self.autopilot == 1: 
            self.pub_cmd.publish(twist)
        
        
        ## Scan horizontal rows:
        # y_coords = [int(height * w) for w in (0.8, 0.65, 0.4)]    # Edit weights to change row heights

        '''
        Figure out additional object detection and secondary state-based control flow algorithms here:
        '''
        # if (state == 0):
            # TODO: look for crosswalk and switch to motion-detection mode. (Stop until safe)
            # Also figure out this control algorithm for the  



        if self.debug:
            if(self.fcount ==0): self.update_tunables() # Empirically speaking this is necessary
            
            # === Visualized Feedback ===
            # telemetry = self.maketelemetry(thresh, y_coords)
            # cv2.imshow("Telemetry", telemetry)
            # cv2.imshow("raw", cv_image)
            # cv2.imshow("gray", gray)
            # cv2.imshow("blurred", blurred)
            # cv2.imshow("thresh", thresh)    # This gives us road values -> doesn't 
            ''' Can also show blobs and color masks in respective helper functions. '''

            # blob_img, one_blob = None, None

            # Optional: draw blobs
            if len(keypoints) > 1:
                blob_img = cv2.drawKeypoints(thresh, keypoints[:2], None, (0, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.imshow("Telemetry", blob_img)
            elif len(keypoints) == 1:
                one_blob = cv2.drawKeypoints(thresh, keypoints, None, (0, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.imshow("Telemetry", one_blob)
            else:
                cv2.imshow("Telemetry", thresh)
            
            if(self.fcount==0): cv2.moveWindow("Telemetry", 1500, 380)  # (x=1500px from left, y=150px from top)
            cv2.waitKey(1)

        if self.show_time:
            print(f"Execution time: {(time.time()- self.lasttime)*1000} ms") # Display the time in ms
            self.lasttime = time.time()


        if(self.fcount == 0): 
            del cv_image, resized, roi, gray, blurred, thresh # Will this do anything we'll see
            # if blob_img is not None: del blob_img
            # if one_blob is not None: del one_blob


        

        '''Save PID coeffs here (but don't use this yet)'''
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('s'):
        #     self.save_settings()



if __name__ == '__main__':
    Drive = PID_control()
    rate = rospy.Rate(30)  # 30 Hz -- 30 fps but not necessarily

    boottwist = Twist()
    boottwist.linear.x = 0.0
    boottwist.angular.z = 0.0
    Drive.pub_cmd.publish(boottwist) #Just do this once

    while not rospy.is_shutdown():
        try:
            data = rospy.wait_for_message("/B1/rrbot/camera1/image_raw", Image, timeout=1.0)
            Drive.process_image(data)

            if Drive.debug:
                cv2.waitKey(1)
        except rospy.ROSException:
            rospy.logwarn("No image received")
        
        rate.sleep()

    # rospy.Subscriber("/B1/rrbot/camera1/image_raw", Image, Drive.process_image)
    # rospy.spin()  # replaces the while loop

