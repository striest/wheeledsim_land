#!/usr/bin/python3

import rospy
import torch
import argparse
import os
import matplotlib.pyplot as plt
from tabulate import tabulate

from sensor_msgs.msg import Joy
from std_msgs.msg import Bool
from std_msgs.msg import Float32

from wheeledsim_rl.networks.cnn_blocks.cnn_blocks import ResnetCNN
from wheeledsim_rl.replaybuffers.dict_replaybuffer import NStepDictReplayBuffer
from wheeledsim_rl.util.util import dict_map, dict_to

from rosbag_to_dataset.converter.online_converter import OnlineConverter
from rosbag_to_dataset.config_parser.config_parser import ConfigParser
from rosbag_to_dataset.util.os_util import str2bool

from wheeledsim_land.policies.to_joy import ToJoy
from wheeledsim_land.policies.action_sequences import generate_action_sequences
from wheeledsim_land.policies.action_sequence_policy import EnsembleInterventionMinimizePolicy
from wheeledsim_land.replay_buffers.intervention_replay_buffer import InterventionReplayBuffer
from wheeledsim_land.trainers.ensemble_intervention_predictor import EnsembleInterventionPredictionTrainer

from wheeledsim_land.visualization.replay_buffer import ReplayBufferViz
from wheeledsim_land.visualization.intervention_prediction import EnsembleInterventionPredictionViz

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_spec', type=str, required=True, help='Path to the yaml file that contains the dataset spec.')
    parser.add_argument('--use_stamps', type=str2bool, required=False, default=True, help='Whether to use the time provided in the stamps or just the ros time')
    parser.add_argument('--n_steer', type=int, required=False, default=5, help='Number of steering angles to consider')
    parser.add_argument('--T', type=int, required=False, default=5, help="Number of timesteps (from yaml) to execute actions for")
    parser.add_argument('--K', type=int, required=False, default=3, help="Number of nets in ensemble")
    parser.add_argument('--pT', type=int, required=False, default=2, help="Multiplier on T for collision lookahead")
    parser.add_argument('--grad_rate', type=float, required=False, default=1., help='Numer of training steps to take per dt')
    parser.add_argument('--viz', type=str2bool, required=False, default=True, help='Whether to display a viz')

    args = parser.parse_known_args()[0]

    print("ARGS:")
    print(tabulate(vars(args).items(), headers = ['Arg', 'Value'], tablefmt='psql'))

    rospy.init_node('online_learning')

    config_parser = ConfigParser()
    spec, converters, remap, rates = config_parser.parse_from_fp(args.config_spec)
    converter = OnlineConverter(spec, converters, remap, rates, args.use_stamps)

    steer_n = 5
    T = 5
    smax = 0.3
    frame_offset = 2

#    buf = NStepDictReplayBuffer(spec, capacity=2000).to('cuda')
    buf = InterventionReplayBuffer(spec, capacity=10000, frame_offset=frame_offset).to('cuda')

    nets = []
    opts = []

    for i in range(args.K):
        net = ResnetCNN(insize=[3, 128, 128], outsize=steer_n, n_blocks=3, pool=4, mlp_hiddens=[32, ]).to('cuda')
        opt = torch.optim.Adam(net.parameters(), lr=3e-4)
        nets.append(net)
        opts.append(opt)

    seqs = generate_action_sequences(throttle=(1, 1), throttle_n=1, steer=(-smax, smax), steer_n=steer_n, t=T)
    policy = EnsembleInterventionMinimizePolicy(env=None, action_sequences=seqs, nets=nets, lam=-5.0)
    joy_policy = ToJoy(policy, saxis=2, taxis=1).to('cuda')

    trainer = EnsembleInterventionPredictionTrainer(policy, nets, buf, opts, T=args.pT*T, sscale=-1.0)

    cmd_pub = rospy.Publisher("/joy_auto", Joy, queue_size=1)

#    replay_buffer_viz = ReplayBufferViz(buf)
    intervention_prediction_viz = EnsembleInterventionPredictionViz(policy)

    rate = rospy.Rate(1/spec.dt)

    print('waiting 1s for topics...')
    for i in range(int(1/spec.dt)):
        rate.sleep()

    prev_data = dict_to(converter.get_data(), buf.device)

    plt.show(block=False)

    if args.grad_rate > 1:
        gi = 1
        gii = int(args.grad_rate)
    else:
        gi = int(1/args.grad_rate)
        gii = 1

    i = 0

    #Ok, we're doing this the sketchy way (global var saying if should train)
    should_train = False
    def train_callback(msg):
        global should_train
        should_train = msg.data

    train_sub = rospy.Subscriber('/enable_training', Bool, train_callback)

    #Callback to update uncertainty coeff
    def unc_callback(msg):
        global policy
        policy.lam = msg.data

    unc_sub = rospy.Subscriber('/policy/lambda', Float32, unc_callback)

    while not rospy.is_shutdown():
        can_sample = buf.can_sample(trainer.T)
        print('TRAIN = {}'.format(should_train))
        print('CAN SAMPLE = {}'.format(can_sample))

        data = dict_to(converter.get_data(), buf.device)

        batch = {
                'observation': prev_data['observation'],
                'action':prev_data['action'],
                'reward':torch.tensor([0]).to(buf.device),
                'terminal':torch.tensor([False]).to(buf.device),
                'next_observation': data['observation']
                }

        batch = dict_map(batch, lambda x:x.unsqueeze(0))
        buf.insert(batch)

        msg = joy_policy.action(data['observation'])
        cmd_pub.publish(msg)

        prev_data = data

#        replay_buffer_viz.update_figs()
        intervention_prediction_viz.update(prev_data['observation']['image_rgb'])

        print(buf.n)

        plt.pause(1e-2)

        if (~buf.intervention).sum() > 0 and buf.n > 0 and (i % gi) == 0 and should_train and can_sample:
            for i in range(gii):
                trainer.update()

        i += 1
        rate.sleep()

    batch = {
            'observation': prev_data['observation'],
            'action':prev_data['action'],
            'reward':torch.tensor([0]).to(buf.device),
            'terminal':torch.tensor([True]).to(buf.device),
            'next_observation': data['observation']
            }

    batch = dict_to(dict_map(batch, lambda x:x.unsqueeze(0)), 'cuda')
    buf.insert(batch)
    print('saving buf to {}...'.format(os.path.join(os.getcwd(), 'buffer.pt')))
    torch.save(buf.to('cpu'), 'buffer.pt')
    torch.save(net.to('cpu'), 'net.pt')
