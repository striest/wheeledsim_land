#!/usr/bin/env python3

import rospy

from sensor_msgs.msg import Joy
from common_msgs.msg import BoolStamped

class CmdMux:
    """
    Mux between a teleop and autonomous command safely.
    Publish to a mux'ed topic and intervention indicator
    General behavior is to publish the teleop command unless teleop is 0, and auto button is pressed.
    """
    def __init__(self, auto_button=5, skip_axes=[]):
        self.auto_button = auto_button
        self.skip_axes = skip_axes

        self.auto_msg = Joy()
        self.auto_msg.axes = [0.] * 6
        self.auto_msg.buttons = [0] * 12
        self.teleop_msg = Joy()
        self.teleop_msg.axes = [0.] * 6
        self.teleop_msg.buttons = [0] * 12

    def handle_teleop(self, msg):
        axes = list(msg.axes)
        for i in self.skip_axes:
            axes[i] = 0.
        msg.axes = axes

        self.teleop_msg = msg

    def handle_autonomous(self, msg):
        axes = list(msg.axes)
        for i in self.skip_axes:
            axes[i] = 0.
        msg.axes = axes

        self.auto_msg = msg

    def get_mux_cmd(self):
        out_msg = self.auto_msg if self.check_auto() else self.teleop_msg
        out_msg.header.stamp = rospy.Time.now()
        return out_msg

    def get_intervention(self):
        msg = BoolStamped()
        msg.header.stamp = rospy.Time.now()
        msg.data = not self.check_auto()
        return msg

    def check_auto(self):
        """
        Only allow auto if:
            1. The teleop auto button is held down
            2. Every other teleop button/axis is 0
        """
        auto = all([abs(x) < 1e-2 for x in self.teleop_msg.axes]) and (sum(self.teleop_msg.buttons)==1) and (self.teleop_msg.buttons[self.auto_button]==1)
        
        return auto

if __name__ == '__main__':
    rospy.init_node('cmd_mux')
    muxer = CmdMux(auto_button=4, skip_axes=[])
    teleop_sub = rospy.Subscriber('/joy', Joy, muxer.handle_teleop)
    auto_sub = rospy.Subscriber('/joy_auto', Joy, muxer.handle_autonomous)
    mux_pub = rospy.Publisher('/mux/joy', Joy, queue_size=10)
    intervention_pub = rospy.Publisher('/mux/intervention', BoolStamped, queue_size=10)

    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        mux_pub.publish(muxer.get_mux_cmd())
        intervention_pub.publish(muxer.get_intervention())
        print(muxer.get_mux_cmd())
        rate.sleep()
