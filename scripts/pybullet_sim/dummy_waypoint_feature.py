#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Vector3Stamped

"""
To keep parity with ATV, publish dummy waypoint features
"""

if __name__ == '__main__':
    rospy.init_node('dummy_waypoint_feature_publisher')
    rate = rospy.Rate(10)
    pub = rospy.Publisher('/waypoint_feature', Vector3Stamped, queue_size=1)

    while not rospy.is_shutdown():
        msg = Vector3Stamped()
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        rate.sleep()
