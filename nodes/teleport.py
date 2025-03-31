#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String

# Imports recommended by ChatGPT during debugging:
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

class Teleport:
    def __init__(self):
        rospy.init_node('topic_publisher', anonymous=True)

        # Publishers
        self.pub_cmd = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)
        self.pub_score = rospy.Publisher('/score_tracker', String, queue_size=1)

        # Wait for connections to publishers
        rospy.sleep(1.0)

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

        
        # TODO: Implement read_sign()

    def get_sign(self, line_num):
        ### Read the sign

        stop = Twist()
        stop.linear.x = 0.0
        stop.linear.y = 0.0

        if (line_num == 0):
                # clue = read_sign()
                # self.pub_score.publish('Egg,unknown,0,', clue)
                self.pub_score.publish('Egg,unknown,0,CLUE')
                rospy.sleep(1) # 2?

        elif (line_num == 1):
                s1 = Twist()
                s1.linear.x = -3.2
                # self.pub_cmd.publish.linear.x = -2.0
                self.pub_cmd.publish(s1)
                rospy.sleep(1)
                self.pub_cmd.publish(stop)
                # clue = read_sign()
                # self.pub_score.publish('Egg,unknown,123424,', clue)
                rospy.sleep(1)

        # elif (line_num == 2):
        #         # Chill out idk what's going on here yet.
        #         rospy.sleep(1)
        
        # elif (line_num == 3):
        ## Skip second pink line and go to third, final line.
        else:
                s3 = Twist()
                s3.angular.z = 1.6

                self.pub_cmd.publish(s3)
                rospy.sleep(1)
                self.pub_cmd.publish(stop)
                # clue = read_sign()
                # self.pub_score.publish('Egg,unknown,123424,', clue)
                rospy.sleep(1)

                # Consider ending the run here.
                

    def run(self):
        # Read the first sign on-spawn.
        self.get_sign(0)

        # Coordinates: x, y, z, ox, oy, oz, ow (x,y,z, quaternion)
        pink_lines = [  
                        [ 0.5,  -0.05,  0.05,   0.0, 0.0, 0.7071, 0.7071],      # First line (spawn not included)
                        # [-3.9,   0.465, 0.05,   0.0, 0.0, 1.0,    0.0],       # Second line, skip this one
                        [-4.05, -2.3,   0.05,   0.0, 0.0, 0.0,    0.0]          # Third line
                ]
        
        line_num = 0
        
        for pos in pink_lines:
                self.spawn_position(pos)
                line_num += 1
                self.get_sign(line_num)

        self.pub_score.publish('Egg,unknown,-1,END')

if __name__ == '__main__':
    teleport = Teleport()
    teleport.run()
