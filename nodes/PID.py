#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import cv2
from sensor_msgs.msg import Image # Is this needed?
from cv_bridge import CvBridge, CvBridgeError

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

    def __init__(self):
        rospy.init_node('topic_publisher', anonymous=True)

        # ROS Setup
        self.pub_cmd = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)
        self.pub_cb_detected = rospy.Publisher('/cb_detected', String, queue_size=1)
        self.pub_score = rospy.Publisher('/score_tracker', String, queue_size=1)
        self.image_sub = rospy.Subscriber("/robot/camera1/image_raw", Image, self.callback) # Make this Low-Res
        self.bridge = CvBridge()

        # PID Control Parameters
        self.Kp = 0.05     # Proportional gain  
        self.Ki = 0.001    # Integral gain 
        self.Kd = 0.002     # Derivative gain 
        self.prev_error = 0
        self.integral = 0

        # Speed Limits
        self.maxspeed = 1.50
        self.reducedspeed = 0.50

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
        derivative = error - self.prev_error # Difference term for sign
        self.prev_error = error # Update error
        return (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)


    ### Helper function counting colored pix for cd_detector() and scan_pink()
    def count_colorpix(self, img, color_name):
        """Return number of pixels matching the given color in the image."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if color_name == "blue":
            lower = np.array([100, 150, 50])
            upper = np.array([130, 255, 255]) ## Play with these
        elif color_name == "pink":
            lower = np.array([140, 50, 50])
            upper = np.array([170, 255, 255])
        else:
            rospy.logwarn(f"Unknown color requested: {color_name}")
            return 0

        mask = cv2.inRange(hsv, lower, upper)
        pixel_count = cv2.countNonZero(mask)

        # Optional: show mask for debugging
        # cv2.imshow(f"{color_name} mask", mask)

        return pixel_count
 

    # Scan for blue and publish to cb_detected
    def cb_detector(self, cv_image):
        min_blue = 100 # At least 100 bluepix to trigger "cb detected -> yes"
        bluepix = self.count_colorpix(cv_image, "blue")
        cb_detected = 'yes' if (bluepix > N) else 'no'
        self.pub_cb_detected.publish(cb_detected)

    def scan_pink(self, cv_image):
        current_pink = self.count_colorpix(cv_image, "pink")
        '''
        Existing code could potentially double update:
        '''
        if current_pink > 0:
            self.consec_pink_frames += 1
            # self.consec_pinkless = 0
        else:
            self.consec_pink_frames = 0
            # self.consec_pinkless += 1

        if (current_pink = 0 and self.consec_pink_frames > 10):
            state += 1
            self.update_state()

        self.pinkpix = current_pink # Update at the end.

    def update_state(self):
        # Update internal params based on current terrain state
        if self.state == STATE_PAVED:
            rospy.loginfo("Switched to: PAVED")
            self.Kp = 0.05
            self.Ki = 0.001
            self.Kd = 0.002
            self.thresh_val = 240
            self.maxspeed = 1.5
            self.reducedspeed = 0.5
            self.obj_detection = True

        elif self.state == STATE_UNPAVED:
            rospy.loginfo("Switched to: UNPAVED")
            self.Kp = 0.04
            self.Ki = 0.002
            self.Kd = 0.003
            self.thresh_val = 230
            self.maxspeed = 1.2
            self.reducedspeed = 0.4
            self.obj_detection = False

        elif self.state == STATE_OFFROAD:
            rospy.loginfo("Switched to: OFFROAD")
            self.obj_detection = False
            # Insert logic for follow-yoda or hardcoded path
            self.handle_offroad()

        elif self.state == STATE_MOUNTAIN:
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
            start, stop = (0, y), (colored.shape[1], y)
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




    '''
    Implement the main image processing control loop below and figure out how it fits in 
    with the high level running loop (and calling blue detector guy).
    '''

    def process_image(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        ### Resizing:
        resized = cv2.resize(cv_image, (400, 400))
        roi = resized[(resized.shape[0]*0.5):, :]  # bottom half
        # roi = cv_image[(cv_image.shape[0]*0.5):, :]  # bottom half -> Use after making second camera
        height, width = roi.shape[:2]
        image_center = width // 2

        ## Preprocessing (grayscale, blur, binary threshold)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, self.thresh_val, 255, cv2.THRESH_BINARY)


        # Check for colors (update state and publish cb_detector here)
        self.cb_detector(cv_image)
        self.scan_pink(cv_image)

        ## Scan horizontal rows:
        y_coords = [int(height * w) for w in (0.8, 0.65, 0.4)]    # Edit weights to change row heights
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
            white_pixels = np.where(binary > 0)
            if white_pixels[1].size > 0:
                center_est = int(np.mean(white_pixels[1]))
                error = image_center - center_est
                twist.angular.z = self.pid_control(error)
                twist.linear.x = self.reducedspeed
            else: 
                # No white anywhere, stop & spin
                twist.angular.z = 1.0
                twist.linear.x = 0.0
        else:
            # Midpoints is not empty:
            road_center = sum(midpoints) // len(midpoints)

            ## Attempt to do fallback here:
            white_pixels = np.where(binary > 0)
            if white_pixels[1].size > 0:
                road_center = int(np.mean(white_pixels[1]))
            else:
                road_center = image_center ### DRIVE STRAIGHT FORWARD
                ## Idk fix this later

            error = image_center - road_center
            twist.angular.z = self.pid_control(error)
            twist.linear.x = self.maxspeed if abs(error) < 10 else self.reducedspeed
            rospy.loginfo(f"Error: {error}, Angular Z: {twist.angular.z:.2f}, Linear X: {twist.linear.x:.2f}")

        self.pub_cmd.publish(twist)

        # === Visual feedback ===
        telemetry = self.maketelemetry(thresh, y_coords)
        cv2.imshow("Telemetry", telemetry)
        # cv2.imshow("Tuner", Tuner)
        cv2.waitKey(1)

        # cv2.imshow("blurred", blurred)
        # cv2.imshow("thresh", thresh)
        # cv2.imshow("gray", gray)
        # cv2.imshow("raw", cv_image)
        # cv2.waitKey(1)


        ##############################################################


        max_error = 10 # (arbitrary)

        resized = cv2.resize(cv_image, (400, 400))  # Shrink incoming image size before processing

        height = resized.shape[0]
        width = resized.shape[1]

        y_coord = int(height * 0.8)  # y-coord for scanning, bottom row
        

        # Resize and crop region of interest (optional)
        roi = resized[int(height * 0.5):, :]  # Bottom half of the frame

        # Update dimensions (clean this up later):
        height = roi.shape[0]
        width = roi.shape[1]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(blurred, self.thresh_val, 255, cv2.THRESH_BINARY)




        '''
        Check for road between lines. (do subtraction and check for )
        
        Telemetry is going to be pivotal here.

        binary threshold everything with only very bright whites getting through and just look for a jump between indices to indicate a jump.

        '''


        ### For 3 y locations, find midpoints and average them. Weight the middle highest, bottom second highest, and top third highest.

        y_coords = [(int(height*0.8)), (int(height*0.65)), (int(height*0.4))]

        # mp_weights = [0.35, 0.45, 0.20]        # Weights sum to 1.0 jk dont do this
        midpoints = []

        for n in range(len(y_coords)):
            lanelines = []
            scan_row = thresh[y_coords[n], :]
            for i in range(len(scan_row)):
                if (scan_row[i] == 255):
                    lanelines.append(i)

            min_check = 50 # Num pixels between white lines detected (minimum 500?)
            for i in range(len(lanelines)-1):
                if (lanelines[i+1] - lanelines[i] > min_check):     # Confirm that this only occurs once?
                    left = lanelines[i] + 1
                    right = lanelines[i+1] - 1
                    # TODO: Fix this it is so wrong -- this is just for initial git push.
                    midpoints.append((left + right // 2))

                    # Edge cases,   we read that the gap is sufficient but it's misreading (i.e. crosswalk)
                    #               left white line is off the page, 
                    #               the above case but we detect a fleck and it gets confused. (Contour detection and filter that way?)
                    #               It sees more than 2 lines and gets overwritten. (Integral term gets bad?)
                    #               Any of these could get really bad in the second section.
                    #               Signs would confuse it.
                    # Arguments for leaving this:
                    #   If it misreads only one frame, it may just correct next frame.
                    # Arguments against: Most of these edge cases aren't just for one frame.
   
        twist = Twist()
        image_center = width // 2
        
        # IDEALLY THIS IF STATEMENT IS NEVER TRUE
        if (len(midpoints) == 0):
            # print("Midpoints not found, spinning CCW")
            # twist.linear.x = 0.0
            # twist.angular.z = 3.0  # Spin until road is found
            # TODO: Change this so we check derivative error and spin the right way.

            # if((lanelines[-1] - lanelines[0]) > image_center):
            #     error = ((lanelines[0] + lanelines[-1]) // 2) - imagecenter
            if (len(lanelines) > 0):
                print("Midpoints not found, choosing line")
                error = image_center - ((lanelines[0] + lanelines[-1]) // 2)
                # Check that the sign is right on this
                    
                # Apply PID
                twist.angular.z = self.pid_control(error)
                twist.linear.x = self.reducedspeed # Doubly reduced
            else:
                print("No lines detected anywhere, spinning")
                twist.angular.z = 1.0
                twist.linear.x = -0.05 # maybe make zero


            # read first and last vals in lanelines, if theyre x pix wide and 10 < x < 50 pix, 
            # treat it as a lane line and set road center left.

            '''
            Add a check for the case where we actually just lost the line. Drive backwards? or spin?
            '''
        else:
            road_center = sum(midpoints) // len(midpoints)
            error = image_center - road_center  # error value for PID

            # Apply PID
            twist.angular.z = self.pid_control(error)
            twist.linear.x = self.maxspeed if (error < max_error) else self.reducedspeed    # Max speed decreases if error exceeds some value.

            print(f"Error: {error}, Angular Z: {twist.angular.z}, Linear X: {twist.linear.x}")
        
        self.pub_cmd.publish(twist)

 

        # if (state == 0):
            # TODO: look for crosswalk and switch to motion-detection mode. (Stop until safe)
            # Also figure out this control algorithm for the  

        ### Telemetry and GUI stuff:
        telemetry = self.maketelemetry(thresh, y_coords)

        cv2.imshow("Telemetry", telemetry)
        cv2.waitKey(1)
        
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('s'):
        #     self.save_settings()



if __name__ == '__main__':
    Drive = PID_control()
    rate = rospy.Rate(30)  # 30 Hz -- 30 fps but not necessarily

    # Rospy spinning shenanigans? 
    #    - Only if we can end it once the mountain sign is read.

    while not rospy.is_shutdown():
        try:
            data = rospy.wait_for_message("/B1/rrbot/camera1/image_raw", Image, timeout=5) # Is path correct?
            Drive.process_image(data)
        except rospy.ROSException:
            # pass
            rospy.logwarn("No image received in 5s, retrying...")
        
        rate.sleep()
