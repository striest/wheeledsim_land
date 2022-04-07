#!/usr/bin/python3

import rospy
import yaml

from wheeledsim_land.visualization.eil_viz import EilVisualization

if __name__ == '__main__':
    rospy.init_node('eil_viz')

    actlib_fp = rospy.get_param('~action_library_path')
    action_topic = rospy.get_param("~discrete_action_topic")
    q_values_topic = rospy.get_param("~q_values_topic")
    intervention_topic = rospy.get_param("~intervention_topic")
    base_frame = rospy.get_param("~base_frame")
    actlib = yaml.safe_load(open(actlib_fp, 'r'))

    viz = EilVisualization(actlib, action_topic, q_values_topic, intervention_topic, base_frame)

    viz.spin()
