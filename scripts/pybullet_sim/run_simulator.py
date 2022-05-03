import rospy
import torch
import numpy as np
import matplotlib.pyplot as plt
import pybullet
import time
import tf2_ros

from std_msgs.msg import Int32
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from common_msgs.msg import AckermannDriveArray, Float32Stamped
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TransformStamped, Pose, Point, Quaternion
from grid_map_msgs.msg import GridMap
from rosgraph_msgs.msg import Clock

from wheeledSim.rosSimController import rosSimController

def racetrack_gridmap_to_costmap(msg):
    """
    Simple debug script that turns the gridmap from sim into a costmap
    Since I'm assuming it's the racetrack env, I'm just converting the message and saying green=high cost
    """
    cmap_msg = OccupancyGrid()
    cmap_msg.header.stamp = msg.info.header.stamp
    cmap_msg.header.frame_id = msg.info.header.frame_id
    cmap_msg.info.map_load_time = msg.info.header.stamp
    cmap_msg.info.resolution = msg.info.resolution

    nx = int(msg.info.length_x/msg.info.resolution)
    ny = int(msg.info.length_y/msg.info.resolution)
    idx = msg.layers.index('green')

    cmap_msg.info.width = nx
    cmap_msg.info.height = ny
    cmap_msg.info.origin = Pose(
        position = Point(
            x=msg.info.pose.position.x - msg.info.length_x/2.,
            y=msg.info.pose.position.y - msg.info.length_y/2.,
            z=msg.info.pose.position.z
        ),
        orientation = Quaternion(w=1.0)
    )

    data = np.array(msg.data[idx].data).astype(np.uint8).reshape(nx, ny) * 100
    cmap_msg.data = data[::-1, ::-1].flatten() #gridmap indexing is different
    return cmap_msg

if __name__ == '__main__':
    from wheeledRobots.clifford.cliffordRobot import Clifford

    from wheeledSim.sensors.shock_travel_sensor import ShockTravelSensor
    from wheeledSim.sensors.local_heightmap_sensor import LocalHeightmapSensor
    from wheeledSim.sensors.front_camera_sensor import FrontCameraSensor
    from wheeledSim.sensors.lidar_sensor import LidarSensor

    rate = 10
    dt = 1./rate
    curr_time = 0.

    rospy.init_node("pybullet_simulator")
    rate = rospy.Rate(10)

    config_path = rospy.get_param("~simulator_config_file")

    env = rosSimController(config_path, render=True)

    cmd_sub = rospy.Subscriber("/cmd", AckermannDriveStamped, env.handle_cmd, queue_size=1)

    clock_pub = rospy.Publisher("/clock", Clock, queue_size=1)
    odom_pub = rospy.Publisher("/odom", Odometry, queue_size=1)
    cam_pub = rospy.Publisher("/front_camera", Image, queue_size=1)
    steer_angle_pub = rospy.Publisher('/steering_angle', Float32Stamped, queue_size=1)
    globalmap_pub = rospy.Publisher('/global_map', GridMap, queue_size=1)
    costmap_pub = rospy.Publisher("/local_cost_map_final_occupancy_grid", OccupancyGrid, queue_size=1)
    tf_br = tf2_ros.TransformBroadcaster()

    while not rospy.is_shutdown():
        t1 = time.time()
        env.step()
        t2 = time.time()
        state, sensing = env.get_sensing()
        img = sensing['front_camera']
        ang = sensing['steering_angle']
        ang.data *= (180./np.pi) * (-415./30.) #Convert to degrees steering wheel for consistency with ATV
        gmap = sensing['global_map']

        cmap = racetrack_gridmap_to_costmap(gmap)

        t3 = time.time()

        odom_pub.publish(state)
        cam_pub.publish(img)
        steer_angle_pub.publish(ang)
        globalmap_pub.publish(gmap)
        costmap_pub.publish(cmap)

        #Tf
        tf_msg = TransformStamped()
        tf_msg.header.stamp = state.header.stamp
        tf_msg.header.frame_id = state.header.frame_id
        tf_msg.child_frame_id = state.child_frame_id
        tf_msg.transform.translation.x = state.pose.pose.position.x
        tf_msg.transform.translation.y = state.pose.pose.position.y
        tf_msg.transform.translation.z = state.pose.pose.position.z
        tf_msg.transform.rotation.x = state.pose.pose.orientation.x
        tf_msg.transform.rotation.y = state.pose.pose.orientation.y
        tf_msg.transform.rotation.z = state.pose.pose.orientation.z
        tf_msg.transform.rotation.w = state.pose.pose.orientation.w
        tf_br.sendTransform(tf_msg)

#        print('STEP: {:.6f}, SENSE: {:.6f}'.format(t2-t1, t3-t2))

        rate.sleep()
