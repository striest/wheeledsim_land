#!/usr/bin/python3

import rospy

from sensor_msgs.msg import Joy

from rosbag_to_dataset.converter.online_converter import OnlineConverter
from rosbag_to_dataset.config_parser.config_parser import ConfigParser
from rosbag_to_dataset.util.os_util import str2bool

from wheeledsim_land.policies.to_joy import ToJoy
from wheeledsim_land.policies.action_sequences import generate_action_sequences
from wheeledsim_land.policies.action_sequence_policy import RandomActionSequencePolicy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_spec', type=str, required=True, help='Path to the yaml file that contains the dataset spec.')
    parser.add_argument('--use_stamps', type=str2bool, required=False, default=True, help='Whether to use the time provided in the stamps or just the ros time')

    args = parser.parse_known_args()[0]

    config_parser = ConfigParser()
    spec, converters, remap, rates = config_parser.parse_from_fp(args.config_spec)
    converter = OnlineConverter(spec, converters, remap, rates, args.use_stamps)

    rospy.init_node('random_policy')

    seqs = generate_action_sequences(throttle=(1, 1), throttle_n=1, steer=(-1, 1), steer_n=5, t=10)
    policy = RandomActionSequencePolicy(env=None, action_sequences=seqs)
    joy_policy = ToJoy(policy, saxis=2, taxis=1)

    cmd_pub = rospy.Publisher("/joy_auto", Joy, queue_size=1)

    rate = rospy.Rate(10)

    rate.sleep()

    import pdb;pdb.set_trace()
    obs = converter.get_data()
    msg = joy_policy.action(obs)
    cmd_pub.publish(msg)
    rate.sleep()
