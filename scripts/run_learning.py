#!/usr/bin/python3

import rospy
import torch
import argparse
import os
import matplotlib.pyplot as plt

from sensor_msgs.msg import Joy

from wheeledsim_rl.networks.cnn_blocks.cnn_blocks import ResnetCNN
from wheeledsim_rl.replaybuffers.dict_replaybuffer import NStepDictReplayBuffer
from wheeledsim_rl.util.util import dict_map, dict_to

from rosbag_to_dataset.converter.online_converter import OnlineConverter
from rosbag_to_dataset.config_parser.config_parser import ConfigParser
from rosbag_to_dataset.util.os_util import str2bool

from wheeledsim_land.policies.to_joy import ToJoy
from wheeledsim_land.policies.action_sequences import generate_action_sequences
from wheeledsim_land.policies.action_sequence_policy import InterventionMinimizePolicy
from wheeledsim_land.replay_buffers.intervention_replay_buffer import InterventionReplayBuffer
from wheeledsim_land.trainers.intervention_predictor import InterventionPredictionTrainer

from wheeledsim_land.visualization.replay_buffer import ReplayBufferViz
from wheeledsim_land.visualization.intervention_prediction import InterventionPredictionViz

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_spec', type=str, required=True, help='Path to the yaml file that contains the dataset spec.')
    parser.add_argument('--use_stamps', type=str2bool, required=False, default=True, help='Whether to use the time provided in the stamps or just the ros time')

    args = parser.parse_known_args()[0]

    rospy.init_node('online_learning')

    config_parser = ConfigParser()
    spec, converters, remap, rates = config_parser.parse_from_fp(args.config_spec)
    converter = OnlineConverter(spec, converters, remap, rates, args.use_stamps)

    steer_n = 5
    T = 5

#    buf = NStepDictReplayBuffer(spec, capacity=2000).to('cuda')
    buf = InterventionReplayBuffer(spec, capacity=50000).to('cuda')
    net = ResnetCNN(insize=[3, 64, 64], outsize=steer_n, n_blocks=2, pool=4, mlp_hiddens=[32, ]).to('cuda')
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)

    seqs = generate_action_sequences(throttle=(1, 1), throttle_n=1, steer=(-1, 1), steer_n=steer_n, t=T)
    policy = InterventionMinimizePolicy(env=None, action_sequences=seqs, net=net)
    joy_policy = ToJoy(policy, saxis=2, taxis=1).to('cuda')

    trainer = InterventionPredictionTrainer(policy, net, buf, opt, T=2*T)

    cmd_pub = rospy.Publisher("/joy_auto", Joy, queue_size=1)

    replay_buffer_viz = ReplayBufferViz(buf)
    intervention_prediction_viz = InterventionPredictionViz(net)

    rate = rospy.Rate(1/spec.dt)

    print('waiting 1s for topics...')
    for i in range(int(1/spec.dt)):
        rate.sleep()

    prev_data = dict_to(converter.get_data(), buf.device)

    plt.show(block=False)

    i = 0
    while not rospy.is_shutdown():
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

        replay_buffer_viz.update_figs()
        intervention_prediction_viz.update(prev_data['observation']['image_rgb'])

        print(buf.n)

        plt.pause(1e-2)

        if (~buf.intervention).sum() > 0 and buf.n > 0:
            for i in range(3):
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
