#!/usr/bin/python3

import rospy
import torch
import numpy as np
import matplotlib.pyplot as plt
import pybullet
import time
import argparse

from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from common_msgs.msg import AckermannDriveArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped

from wheeledSim.rosSimController import rosSimController

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Simulator config file')
    args = parser.parse_known_args()[0]

    env = rosSimController(args.config, render=True)

    rospy.init_node("pybullet_simulator")
    rate = rospy.Rate(10)

    cmd_sub = rospy.Subscriber("/cmd", AckermannDriveStamped, env.handle_cmd, queue_size=1)
    odom_pub = rospy.Publisher("/odom", Odometry, queue_size=1)
    cam_pub = rospy.Publisher("/front_camera", Image, queue_size=1)

    while not rospy.is_shutdown():
        env.step()
        state, sensing = env.get_sensing()
        img = sensing['front_camera']
        odom_pub.publish(state)
        cam_pub.publish(img)
        rate.sleep()
