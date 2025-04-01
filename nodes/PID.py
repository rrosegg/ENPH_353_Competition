#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import cv2
from sensor_msgs.msg import Image # Is this needed?
from cv_bridge import CvBridge, CvBridgeError


'''
TODO: Add state machine code s.t. when a pink line is crossed, we switch into a separate routine. 

States:     Road_driving    Unpaved_driving     Offroad_driving     Mountain_driving

Road driving uses PID and object detection to avoid pedestrian and car while expecting to detect 3 signs.
Unpaved driving uses only PID with pit of despair in mind. Expects to read 3 signs.
Offroad driving is *either hard coded or following yoda. 
Mountain driving is purely well tuned, safe PID that is fast enough to get us up the mountain.

After reading the final sign, we publish the clue and stop driving.

*Use the red truck to align at the end, and consider preporatory alignment before hard coding section.

'''

class PID_drive:
    def __init__(self):
        rospy.init_node('topic_publisher', anonymous=True)

        # Publishers
        self.pub_cmd = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)
        self.pub_cb_detected = rospy.Publisher('/cb_detected', String, queue_size=1)
        # self.pub_score = rospy.Publisher('/score_tracker', String, queue_size=1)

        # Subscribers
        self.image_sub = rospy.Subscriber("/robot/camera1/image_raw", Image, self.callback) # Make this Low-Res

        # Instatiate cv bridge
        self.bridge = CvBridge()

        # PID coefficients (MAKE THESE MORE DYNAMIC)
        self.Kp = 0.005     # Proportional gain  
        self.Ki = 0.0001    # Integral gain 
        self.Kd = 0.002     # Derivative gain 

        self.prev_error = 0
        self.integral = 0

        self.maxspeed = 2.50
        self.reducedspeed = 0.50

        # For the initial state at spawn:
        self.state = 0 # Initialize with 0 and update as we cross pink lines.
        self.obj_detection = True # for paved road, initialize with True.
        self.thresh_val = 240 # stay between white lines

        self.pinkpix = 0 # Number of pink pixels seen in the frame
        self.lastpix = 0 # For use in road detection

        # self.y1 = 
        # self.y2 = 
        # self.y3 = 

        # Wait for connections to publishers (needed??)
        rospy.sleep(0.5) # 1.0?
        

    # Returns correction value: u(t) = Kp*err + Ki*integral(err) + Kp*d/dt(err) 
    def pid_control(self, error):
        self.integral += error # Sums error
        derivative = error - self.prev_error # Difference term for sign
        self.prev_error = error # Update error
        return (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)



    # Mask input img and look for blue. If > x num pixels are blue then capture and read image.
    def cb_detector(self): # Image as well?
        # scan image cv for blue 
        # Use rosemary's colab masking code for this

        N = 100     # this is arbitrary right now
        bluepix = 0 # this should be the number of blue pixels in the incoming raw RBG image.

        cb_detected = 'yes' if (bluepix > N) else 'no'
        self.pub_cb_detected.publish(cb_detected)



    def update_state(self):
        # Paved road
        # if (self.state == 0):
            # Nothing, this is default. PID vals are preset.

        
        # Unpaved road
        if (self.state == 1):
            self.Kp == 999
            self.Ki == 999
            self.Kd == 999
            self.obj_detection = False
            self.thresh_val = 230 # Is this accurate

            self.maxspeed = 999
            self.reducedspeed = 999 
        
        # Offroad
        elif (self.state == 2):
            self.avoid_yoda()

        # Mountain
        elif (self.state == 3):
            self.Kp == 999
            self.Ki == 999
            self.Kd == 999
            self.obj_detection = False


    '''
    Implement the main image processing control loop below and figure out how it fits in 
    with the high level running loop (and calling blue detector guy).
    '''


    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") #Is this true?
        except CvBridgeError as e:
            print(e)
            return              ### low key dont go through this twice

# From white to road color to white then take the edge indexes of the white and of the road.

    def process_image(self, data):
        '''
        this code as is is wrong, need to implement it more similarly to callback above^^^^^
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # cv2.imshow("raw", cv_image)

        height = cv_image.shape[0]
        width = cv_image.shape[1]
      
        y_coord = height - 50  # y-coord for scanning, bottom row
        # road_center = 0

        max_error = 100 # (arbitrary)


        # Resize and crop region of interest (optional)
        roi = cv_image[int(height * 0.5):, :]  # Bottom half of the frame

        # Update dimensions (clean this up later):
        height = roi.shape[0]
        width = roi.shape[1]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(blurred, self.thresh_val, 255, cv2.THRESH_BINARY)

        # cv2.imshow("blurred", blurred)
        # cv2.imshow("thresh", thresh)
        # cv2.imshow("gray", gray)
        # cv2.waitKey(1)
        # cv2.imshow("raw", cv_image)


        '''
        Check for road between lines. (do subtraction and check for )
        
        Telemetry is going to be pivotal here.

        binary threshold everything with only very bright whites getting through and just look for a jump between indices to indicate a jump.

        '''


        ### For 3 y locations, find midpoints and average them. Weight the middle highest, bottom second highest, and top third highest.

        y_coords = [(height - 30), (height - 40), (height - 50)]
        # mp_weights = [0.35, 0.45, 0.20]        # Weights sum to 1.0 jk dont do this
        midpoints = []

        for n in range(len(y_coords)):
            lanelines = []
            scan_row = thresh[y_coords[n], :]
            for i in range(len(scan_row)):
                if (scan_row[i] == 255):
                    lanelines.append(i)

            min_check = 250 # Num pixels between white lines detected (minimum 500?)
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
        
        # IDEALLY THIS IF STATEMENT IS NEVER TRUE
        if (len(midpoints) == 0):
            print("Midpoints not found, spinning CCW")
            twist.linear.x = 0.0
            twist.angular.z = 3.0  # Spin until road is found
            # TODO: Change this so we check derivative error and spin the right way.
        else:
            road_center = sum(midpoints) // len(midpoints)
            image_center = width // 2
            error = image_center - road_center  # error value for PID

            # Apply PID
            twist.angular.z = self.pid_control(error)
            twist.linear.x = self.maxspeed if (error < max_error) else self.reducedspeed    # Max speed decreases if error exceeds some value.

            print(f"Error: {error}, Angular Z: {twist.angular.z}, Linear X: {twist.linear.x}")
        
        self.pub_cmd.publish(twist)


        # Check for pink and blue in here.

        ### Checking for blue:
        N = 100     # this is arbitrary right now
        bluepix = 0 # this should be the number of blue pixels in the incoming raw RBG image.

        cb_detected = 'yes' if (bluepix > N) else 'no'
        self.pub_cb_detected.publish(cb_detected)

        ### Checking for pink line:
        M = 200
        last_pinkpix = self.pinkpix
        new_pinkpix = 99999 # Write some code to check the number of pink pixels in this screen.

        if (last_pinkpix - new_pinkpix > M):
            state += 1 # Update the 



        # if (state == 0):
            # TODO: look for crosswalk and switch to motion-detection mode. (Stop until safe)
            # Also figure out this control algorithm for the  


        cv2.imshow("thresh", thresh)
        cv2.waitKey(1)
        




    def run(self):
        # Read the first sign on-spawn.

        pass
        # self.pub_score.publish('Egg,unknown,-1,END')



if __name__ == '__main__':
    Drive = PID_drive()
    Drive.run()  # optional: can initialize or handle sign reading

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
