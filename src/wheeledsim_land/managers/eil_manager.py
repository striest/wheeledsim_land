#! /usr/bin/python

import rospy
import torch
import os
import numpy as np
from tabulate import tabulate
from cv_bridge import CvBridge

from sensor_msgs.msg import Joy, Image
from std_msgs.msg import Bool, Float64
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose, PoseStamped

from rosbag_to_dataset.converter.online_converter import OnlineConverter
from rosbag_to_dataset.config_parser.config_parser import ConfigParser

from wheeledsim_land.util.logger import Logger
from wheeledsim_land.util.util import dict_map, dict_to


class EilManager:
    """
    General manager class that handles all the ROS->torch and torch->ROS stuff for EIL
    I'm not sure if this is a great way to handle this, but for now I will subscribe to
    a clock that handles the buffer/policy updates
    """
    def __init__(self, config_spec, policy, trainer, seqs, update_rate, train_rate, cmd_pub, use_stamps=True, robot_base_frame='/warty/base', device='cpu'):
        """
        Args:
            config_spec: Path to yaml file containing observation config
            policy: Policy that executes control on the vehicle
            trainer: The trainer to train from (expected to have a net and optimizer and buf)
            seqs: The set of action sequences to train over
            update_rate: The dt to update buffers at
            train_rate: The dt to take trainer updates at
            cmd_pub: The publisher to publish control messages to
            robot_base_frame: For viz, provide the robot's base frame
        """
        # Initialize torch objects
        config_parser = ConfigParser()
        spec, converters, remap, rates = config_parser.parse_from_fp(config_spec)
        self.converter = OnlineConverter(spec, converters, remap, rates, use_stamps)

        self.policy = policy
        self.trainer = trainer
        #For now, assume that trainers have a net, opt, buf and seqs
        self.network = trainer.network
        self.opt = trainer.opt
        self.buf = trainer.buf
        self.seqs = seqs
        self.device = device
        self.prev_data = None

        #Initialize ROS elements
        self.rate = rospy.Rate(1./train_rate)
        self.should_train = False

        self.cmd_pub = cmd_pub
        self.train_sub = rospy.Subscriber('/eil/enable_training', Bool, self.handle_train)
        
        print('wait 1s for topics...')
        for i in range(int(1/train_rate)):
            self.rate.sleep()

        self.converter.get_data()

        #logging/statistics
        self.logger = Logger()
        self.policy_time = -1.
        self.policy_info = {}
        self.trainer_info = {}
        self.itrs = 0

        #debug
        self.robot_base_frame = robot_base_frame
        self.loss_pub = rospy.Publisher('/eil/loss', Float64, queue_size=1)
        self.action_library_pub = rospy.Publisher('/eil/action_library', Path, queue_size=1)
        self.best_action_pub = rospy.Publisher('/eil/best_action', Path, queue_size=1)
        self.waypoint_feature_pub = rospy.Publisher('/eil/waypoint_feature', Path, queue_size=1)
        self.img_pub = rospy.Publisher('/eil/image', Image, queue_size=1)
        self.bridge = CvBridge()

        #Start updating
        rospy.Timer(rospy.Duration(update_rate), self.update)

    def handle_train(self, msg):
        self.should_train = msg.data

    def update(self, event):
        data = dict_to(self.converter.get_data(), self.device)

#        print('Got data:')
#        for k,v in data['observation'].items():
#            print('{}:{}'.format(k, v.shape))
        
        if self.prev_data is not None:
            batch = {
                    'observation': self.prev_data['observation'],
                    'action':self.prev_data['action'],
                    'reward':torch.tensor([0]).to(self.device),
                    'terminal':torch.tensor([False]).to(self.device),
                    'next_observation': data['observation']
                    }

            batch = dict_map(batch, lambda x:x.unsqueeze(0))
            self.buf.insert(batch)

        self.prev_data = data
        ts = rospy.Time.now()
        msg, self.policy_info = self.policy.action(data['observation'], return_info=True)
        self.policy_time = (rospy.Time.now() - ts).to_sec()
        self.cmd_pub.publish(msg)

    def can_train(self):
        can_sample = self.buf.can_sample(self.trainer.T)
        return can_sample and self.should_train

    def log(self):
        self.logger.record_item(field='Iters', data=self.itrs)
        self.logger.record_item(field='Train Msg', data=self.should_train, prefix='Training')
        self.logger.record_item(field='Can Train', data=self.can_train_b, prefix='Training')
        self.logger.record_item(field='Policy Inference Time', data=self.policy_time, prefix='Performance')

        for k,v in self.policy_info.items():
            self.logger.record_item(field=k, data=v, prefix='Policy')

        for k,v in self.trainer_info.items():
            self.logger.record_item(field=k, data=v, prefix='Trainer')

        isamp, nisamp = self.buf.get_intervention_samples(self.trainer.T)
        self.logger.record_item(field='Size', data=len(self.buf), prefix='Buffer')
        self.logger.record_item(field='Num Intervention Samples', data=len(isamp), prefix='Buffer')
        self.logger.record_item(field='Num Nonintervention Samples', data=len(nisamp), prefix='Buffer')

        self.logger.print_data()

    def publish_debug(self):
        if 'loss' in self.trainer_info.keys():
            self.loss_pub.publish(self.trainer_info['loss'])
        
        trajlib_msg = Path()
        trajlib_msg.header.stamp = rospy.Time.now()
        trajlib_msg.header.frame_id = self.robot_base_frame

        for seq in self.seqs:
            p1 = PoseStamped()
            p2 = PoseStamped()
            p1.pose.position.x = seq[-1, 0].item() * 3.
            p1.pose.position.y = seq[-1, -1].item() * 3.
            p1.header.frame_id = self.robot_base_frame
            p2.header.frame_id = self.robot_base_frame
            trajlib_msg.poses.append(p1)
            trajlib_msg.poses.append(p2)

        self.action_library_pub.publish(trajlib_msg)

        if 'act' in self.policy_info.keys():
            best_action_msg = Path()
            best_action_msg.header.stamp = rospy.Time.now()
            best_action_msg.header.frame_id = self.robot_base_frame
            p1 = PoseStamped()
            p2 = PoseStamped()
            p2.pose.position.x = self.policy_info['act'][0].item() * 5.
            p2.pose.position.y = self.policy_info['act'][1].item() * 5.
            p1.header.frame_id = self.robot_base_frame
            p2.header.frame_id = self.robot_base_frame
            best_action_msg.poses.append(p1)
            best_action_msg.poses.append(p2)
            self.best_action_pub.publish(best_action_msg)

        if len(self.buf) > 0 and 'waypoint' in self.buf.buffer['observation'].keys():
            wpt = self.buf.buffer['observation']['waypoint'][(self.buf.n-1) % self.buf.capacity]
            wpt_msg = Path()
            wpt_msg.header.stamp = rospy.Time.now()
            wpt_msg.header.frame_id = self.robot_base_frame
            p1 = PoseStamped()
            p2 = PoseStamped()
            p2.pose.position.x = wpt[0].item()
            p2.pose.position.y = wpt[1].item()
            p1.header.frame_id = self.robot_base_frame
            p2.header.frame_id = self.robot_base_frame
            wpt_msg.poses.append(p1)
            wpt_msg.poses.append(p2)
            self.waypoint_feature_pub.publish(wpt_msg)

        if len(self.buf) > 0 and 'image_rgb' in self.buf.buffer['observation'].keys():
            img = self.buf.buffer['observation']['image_rgb'][(self.buf.n-1) % self.buf.capacity]
            img = img.permute(1, 2, 0)
            print(img.shape)
            img_msg = self.bridge.cv2_to_imgmsg((img * 255.).byte().numpy(), encoding="rgb8")
            self.img_pub.publish(img_msg)

    def spin(self):
        while not rospy.is_shutdown():
            #First collect data.
            self.can_train_b = self.can_train()

            if self.can_train_b:
                self.trainer_info = self.trainer.update()

            self.itrs += 1
            self.log()
            self.publish_debug()

            self.rate.sleep()

#            if self.itrs > 1200:
#                print('SAVE TO: ', os.getcwd())
#                torch.save(self.buf, os.path.join(os.getcwd(), 'buffer.pt'))
#                return

    def to(self, device):
        self.buf = self.buf.to(device)
        self.network = self.network.to(device)
        self.policy = self.policy.to(device)
        return self
