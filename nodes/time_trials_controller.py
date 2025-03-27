#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String # do all of these need to be imported when doing catkin_make_pkg?

rospy.init_node('topic_publisher')
pub_cmd = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)
pub_score = rospy.Publisher('/score_tracker', String, queue_size=1)

# rate = rospy.Rate(2) # Using this and sleep is redundant
move_start = Twist()
move_start.linear.x = 0.5
move_start.angular.z = 0

move_stop = Twist()
move_stop.linear.x = 0.0

rospy.sleep(1)

pub_score.publish('Egg,unknown,0,BEGIN') #what to write for 0 and -1?
rospy.sleep(2) # Buffer for publishing and reading
pub_cmd.publish(move_start)
rospy.sleep(6)
pub_cmd.publish(move_stop)
pub_score.publish('Egg,unknown,-1,END')

# while not rospy.is_shutdown():
#    pub_cmd.publish(move_stop)
#    rate.sleep()