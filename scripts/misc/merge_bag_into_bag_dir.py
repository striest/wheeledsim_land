import rospy
import rosbag
import argparse
import numpy as np
import os

from common_msgs.msg import BoolStamped
from geometry_msgs.msg import Twist

"""
Simple script that takes in a bag of features and adds them into a dir of bags per our yamaha format
"""

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_dir', type=str, required=True, help='Path to the bag dir to add topics to')
    parser.add_argument('--add_bag_fp', type=str, required=True, help='Path to bag with additional topics')
    args = parser.parse_args()

    bag_fps = os.listdir(args.bag_dir)
    print('Merging {} for bag dir {} (containing:)'.format(args.add_bag_fp, args.bag_dir))
    for endpt in bag_fps:
        print('\t{}'.format(endpt))

    for i, endpt in enumerate(bag_fps):
        print('Adding to {} ({}/{})'.format(endpt, i+1, len(bag_fps)))
        base_bag_fp = os.path.join(args.bag_dir, endpt)
        base_bag = rosbag.Bag(base_bag_fp, 'a')

        add_bag = rosbag.Bag(args.add_bag_fp, 'r')

        start_time = base_bag.get_start_time()
        end_time = base_bag.get_end_time()

        for topic, msg, t in add_bag.read_messages():
            t_f = t.to_sec()
            if t_f >= start_time and t_f <= end_time:
                base_bag.write(topic, msg, t)
        
        add_bag.close()
        base_bag.close()

    print('done')
