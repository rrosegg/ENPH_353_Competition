#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
# Add any other necessary imports

'''
Subscribes to: high-res camera, cb_detected nodes
Publishes to: /score_tracker

@brief: If sign_detected = true, then process image and publish clue string.
        else, do nothing.
        
        Keep track of number of clues read (i.e. number of times sign_detected switched from F to T, vice versa) 
        in here, and handle publishing. (This will run in parallel with driving processing.)
'''