#!/usr/bin/python3

import rospy
import torch
import argparse
import os
import yaml
from tabulate import tabulate
import datetime

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool
from common_msgs.msg import Int32Stamped

from rosbag_to_dataset.converter.online_converter import OnlineConverter
from rosbag_to_dataset.config_parser.config_parser import ConfigParser
from rosbag_to_dataset.util.os_util import str2bool

from wheeledsim_land.managers.eil_manager import EilManager
from wheeledsim_land.networks.image_waypoint_network import ResnetWaypointNet
from wheeledsim_land.policies.to_int32 import ToInt32
from wheeledsim_land.policies.action_sequences import generate_action_sequences
from wheeledsim_land.policies.discrete_policy import DiscreteInterventionMinimizePolicy
from wheeledsim_land.replay_buffers.intervention_replay_buffer import InterventionReplayBuffer
from wheeledsim_land.trainers.q_learning import QLearningTrainer
from wheeledsim_land.data_augmentation.gaussian_observation_noise import GaussianObservationNoise
from wheeledsim_land.util.util import dict_map, dict_to

if __name__ == '__main__':
    rospy.init_node('online_learning')

    config_spec_fp = rospy.get_param("~config_spec_path")
    action_library_fp = rospy.get_param("~action_library_path")
    net_dir = rospy.get_param("~net_dir")
    net_fp = rospy.get_param("~net_fp")

    T = rospy.get_param("~T")
    grad_rate = rospy.get_param("~grad_rate")

    actlib = yaml.safe_load(open(action_library_fp, 'r'))
    n_acts = len(actlib['library'])

    config_parser = ConfigParser()
    spec, converters, remap, rates = config_parser.parse_from_fp(config_spec_fp)

    #Set up logging
    log_dir = datetime.datetime.utcfromtimestamp(rospy.Time.now().to_sec()).strftime("%Y-%m-%d-%H-%M-%S")
    log_fp = os.path.join(net_dir, log_dir)
    os.mkdir(log_fp)
    
    print("ARGS:")
    print(tabulate([["Timestep",spec.dt*T], ["Grad Rate",grad_rate], ["N Acts",n_acts], ["Save to",net_dir]], headers = ['Arg', 'Value'], tablefmt='psql'))
        
    buf = InterventionReplayBuffer(spec, capacity=5000).to('cpu')

    if '.pt' not in os.path.basename(net_fp):
        net = ResnetWaypointNet(insize=[3, 128, 128], outsize=n_acts, n_blocks=2, pool=4, mlp_hiddens=[32, ]).to('cpu')
        opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    else:
        rospy.loginfo("Loading net {}".format(net_fp))
        checkpoint = torch.load(net_fp)
        net = checkpoint['net']
        net.load_state_dict(checkpoint['net_state_dict'])
        opt = checkpoint['opt']
        opt.load_state_dict(checkpoint['opt_state_dict'])

    policy = DiscreteInterventionMinimizePolicy(env=None,  net=net)
    joy_policy = ToInt32(policy).to('cpu')

    aug = [GaussianObservationNoise({'image_rgb':0.1})]
    trainer = QLearningTrainer(policy, net, buf, opt, aug, T=T, tscale=1.0, sscale=1.0, discount=0.98, )

    cmd_pub = rospy.Publisher('/eil/discrete_action', Int32Stamped, queue_size=1)

    print("POLICY RATE: {:.2f}s, GRAD RATE: {:.2f}s".format(spec.dt, spec.dt*grad_rate))

    manager = EilManager(config_spec_fp, joy_policy, trainer, spec.dt, spec.dt*grad_rate, cmd_pub, log_fp, log_every=10.).to('cpu')
    manager.spin()
