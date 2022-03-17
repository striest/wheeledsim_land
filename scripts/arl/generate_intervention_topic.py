#! /usr/bin/python3

import rospy

from common_msgs.msg import BoolStamped
from geometry_msgs.msg import Twist

class InterventionGen:
    """
    Quick implementation to generate intervention data
    Low-passes the topic to handle communication latency issues
    """
    def __init__(self, buf_size=3):
        self.buf_size = buf_size
        self.intervention = [False] * self.buf_size
        self.last_msg = False

    def handle_teleop(self, msg):
        self.last_msg = True

    def get_intervention_from_buf(self):
        n = sum(self.intervention)
        return n > (self.buf_size/2)

    def get_intervention_msg(self):
        msg = BoolStamped()
        msg.header.stamp = rospy.Time.now()
        msg.data = self.get_intervention_from_buf()
        self.intervention = self.intervention[1:] + [self.last_msg]
        self.last_msg = False
        return msg

if __name__ == '__main__':
    rospy.init_node('intervention_pub')

    gen = InterventionGen()
    rate = rospy.Rate(10)

    pub = rospy.Publisher('/mux/intervention', BoolStamped, queue_size=1)
    sub = rospy.Subscriber('/wanda/rc_teleop/cmd_vel', Twist, gen.handle_teleop)

    while not rospy.is_shutdown():
        msg = gen.get_intervention_msg()
        pub.publish(msg)
        rate.sleep()
