#! /usr/bin/env python

import rospy

from common_msgs.msg import BoolStamped
from geometry_msgs.msg import Twist

class InterventionGen:
    """
    Quick implementation to generate intervention data
    """
    def __init__(self):
        self.intervention = False

    def handle_teleop(self, msg):
        self.intervention = True

    def get_intervention_msg(self):
        msg = BoolStamped()
        msg.header.stamp = rospy.Time.now()
        msg.data = self.intervention
        self.intervention = False
        return msg

if __name__ == '__main__':
    rospy.init_node('intervention_pub')

    gen = InterventionGen()
    rate = rospy.Rate(10)

    pub = rospy.Publisher('/mux/intervention', BoolStamped, queue_size=1)
    sub = rospy.Subscriber('/warthog2/rc_teleop/cmd_vel', Twist, gen.handle_teleop)

    while not rospy.is_shutdown():
        msg = gen.get_intervention_msg()
        pub.publish(msg)
        rate.sleep()
