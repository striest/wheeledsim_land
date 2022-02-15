#!/usr/bin/python

import rospy
import torch
import argparse
import os
from tabulate import tabulate

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool

from rosbag_to_dataset.converter.online_converter import OnlineConverter
from rosbag_to_dataset.config_parser.config_parser import ConfigParser
from rosbag_to_dataset.util.os_util import str2bool

from wheeledsim_land.managers.eil_manager import EilManager
from wheeledsim_land.networks.image_waypoint_network import ResnetWaypointNet
from wheeledsim_land.policies.to_twist import ToTwist
from wheeledsim_land.policies.action_sequences import generate_action_sequences
from wheeledsim_land.policies.action_sequence_policy import InterventionMinimizePolicy
from wheeledsim_land.replay_buffers.intervention_replay_buffer import InterventionReplayBuffer
from wheeledsim_land.trainers.intervention_predictor import InterventionPredictionTrainer
from wheeledsim_land.util.util import dict_map, dict_to

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_spec', type=str, required=True, help='Path to the yaml file that contains the dataset spec.')
    parser.add_argument('--use_stamps', type=str2bool, required=False, default=True, help='Whether to use the time provided in the stamps or just the ros time')
    parser.add_argument('--n_steer', type=int, required=False, default=5, help='Number of steering angles to consider')
    parser.add_argument('--T', type=int, required=False, default=5, help="Number of timesteps (from yaml) to execute actions for")
    parser.add_argument('--pT', type=int, required=False, default=2, help="Multiplier on T for collision lookahead")
    parser.add_argument('--grad_rate', type=float, required=False, default=1., help='Numer of training steps to take per dt')
    parser.add_argument('--viz', type=str2bool, required=False, default=True, help='Whether to display a viz')

    args = parser.parse_known_args()[0]

    print("ARGS:")
    print(tabulate(vars(args).items(), headers = ['Arg', 'Value'], tablefmt='psql'))

    rospy.init_node('online_learning')

    steer_n = 5
    T = 5
    smax = 1.0
    frame_offset = 0

    config_parser = ConfigParser()
    spec, converters, remap, rates = config_parser.parse_from_fp(args.config_spec)
        
    buf = InterventionReplayBuffer(spec, capacity=5000).to('cpu')
    net = ResnetWaypointNet(insize=[3, 64, 64], outsize=steer_n, n_blocks=2, pool=4, mlp_hiddens=[32, ]).to('cpu')
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)

    seqs = generate_action_sequences(throttle=(1, 1), throttle_n=1, steer=(-smax, smax), steer_n=steer_n, t=T)
    policy = InterventionMinimizePolicy(env=None, action_sequences=seqs, net=net)
    joy_policy = ToTwist(policy).to('cpu')

    trainer = InterventionPredictionTrainer(policy, net, buf, opt, T=args.pT*T, tscale=3.0, sscale=1.0)

    cmd_pub = rospy.Publisher('/wanda/cmd_vel', Twist, queue_size=1)

    print("POLICY RATE: {:.2f}s, GRAD RATE: {:.2f}s".format(spec.dt, spec.dt*args.grad_rate))

    manager = EilManager(args.config_spec, joy_policy, trainer, seqs, spec.dt, spec.dt*args.grad_rate, cmd_pub).to('cpu')
    manager.spin()
