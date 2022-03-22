#!/usr/bin/python3

import rospy
import torch
import argparse
import os
import matplotlib.pyplot as plt
from tabulate import tabulate

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool

from rosbag_to_dataset.converter.online_converter import OnlineConverter
from rosbag_to_dataset.config_parser.config_parser import ConfigParser
from rosbag_to_dataset.util.os_util import str2bool

from wheeledsim_land.networks.image_waypoint_network import ResnetWaypointNet
from wheeledsim_land.replay_buffers.dict_replay_buffer import NStepDictReplayBuffer
from wheeledsim_land.policies.action_sequences import generate_action_sequences
from wheeledsim_land.policies.action_sequence_policy import InterventionMinimizePolicy
from wheeledsim_land.trainers.bc_q_learning import BCQLearningTrainer
from wheeledsim_land.data_augmentation.gaussian_observation_noise import GaussianObservationNoise
from wheeledsim_land.util.util import dict_map, dict_to

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, required=True, help='Path to dir containing the BC data')
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

    smax = 1.0

    config_parser = ConfigParser()
    spec, converters, remap, rates = config_parser.parse_from_fp(args.config_spec)
        
    #Only sample human examples
    buf = NStepDictReplayBuffer(spec, capacity=25000).to('cpu')

    #Load samples into the buf
    fps = os.listdir(args.data_dir)
    for fp in fps:
        traj_fp = os.path.join(args.data_dir, fp)
        traj = dict_to(torch.load(traj_fp), 'cpu')
        buf.insert(traj)

    net = ResnetWaypointNet(insize=[3, 128, 128], outsize=args.n_steer, n_blocks=2, pool=4, mlp_hiddens=[32, ]).to('cpu')
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)

    seqs = generate_action_sequences(throttle=(1, 1), throttle_n=1, steer=(-smax, smax), steer_n=args.n_steer, t=args.T)
    policy = InterventionMinimizePolicy(env=None, action_sequences=seqs, net=net)

    aug = [GaussianObservationNoise({'image_rgb':0.1})]
    trainer = BCQLearningTrainer(policy, net, buf, opt, aug, T=args.T, tscale=1.0, sscale=1.0, discount=0.99, )

    #Actual training here
    N = 2000
    for i in range(N):
        info = trainer.update()
        
        if i % 100 == 0:
            print("{}/{}: {}".format(i+1, N, info))

    torch.save(net, 'bc_init_net.pt')

    #Evaluate (This part isn't generic)
    for i in range(100):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        batch = buf.sample(1, 1)
        s_curr = dict_map(batch['observation'], lambda x: x[:, 0])
        with torch.no_grad():
            qs = net.forward(s_curr).squeeze()

        axs[0].imshow(batch['observation']['image_rgb'].squeeze().permute(1, 2, 0))
        axs[1].plot(qs)
        plt.show()
