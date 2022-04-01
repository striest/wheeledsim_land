import torch
import numpy as np
import copy
import time

from wheeledsim_land.replay_buffers.intervention_replay_buffer import InterventionReplayBuffer
from wheeledsim_land.util.util import dict_map

class QLearningTrainer:
    """
    Train a network using Q learning with intervention as a reward signal.
    I will implement this such that Q is cost.

    NOTE THAT T HERE SHOULD MATCH THE ACTION SELECTION RATE
    """
    def __init__(self, policy, network, buf, opt, augmentations, intervention_cost=5., discount=0.99, soft_update_tau=0.05, batchsize=16, T=10, tscale=1.0, sscale=-0.6):
        """
        Args:
            policy: The policy to train
            network: The policy's network (This is actually what updates)
            buf: Replay buffer to train on
            opt: The optimizer for the net
            augmentations: A list of data augmentations to apply to the dataset
            intervention_cost: The cost associated with an intervention
            discount: Discount factor for Q-learning
            batchsize: Number of examples/minibatch
            T: Number of steps ahead to look for intervention
            tscale: scaling on throttle cmds to map continuous back to discrete
            sscale: scaling on steer cmds to map continuous back to discrete
        """
        self.policy = policy
        self.network = network
        self.buf = buf
        self.opt = opt
        self.augmentations = augmentations

        self.intervention_cost = intervention_cost
        self.discount = discount
        self.soft_update_tau = soft_update_tau

        self.batchsize = batchsize
        self.T = T

        self.target_network = copy.deepcopy(self.network)

    def update(self):
        ts = time.time()
        batch = self.buf.sample(self.batchsize, self.T)
        aug = np.random.choice(self.augmentations)
        batch = aug.forward(batch)

        #_x = batch['observation']['image_rgb'][:, 0]

        s_curr = dict_map(batch['observation'], lambda x: x[:, 0])
        s_next = dict_map(batch['next_observation'], lambda x:x[:, -1])
        seq_idxs = dict_map(batch['action'], lambda x:x[:, 0, 0].long())

        print({k:v.shape for k,v in s_curr.items()})
        intervention = (batch['observation']['intervention'].abs() > 1e-2).any(dim=1).squeeze().float()

        """
        Steps:
            1. Calculate reward
            2. Calculate Q estimates
            3. Compute Bellman backup
            4. Optmize Bellman loss
        """
        costs = self.intervention_cost * intervention
        q_curr = self.network.forward(s_curr)[torch.arange(self.batchsize), seq_idxs]
        with torch.no_grad():
            #TODO: switch to target network once double DQN
            q_next = self.target_network.forward(s_next).min(dim=-1)[0]

        #Mux s.t. terminal states(interventions) have only cost
        #Note that Sanjiban says this is wrong. I'm still not sure why
        q_target = costs + (1.-intervention) * (self.discount * q_next)

        loss = (q_curr - q_target).pow(2).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.soft_update()

        te = time.time() - ts
        info = {
            'loss':loss.detach().cpu().item(),
            'train time': te
        }

        print("INTERVENTIONS: ", intervention)
        print("ACTIONS: ", seq_idxs)
        print("QS: ", q_curr.detach())
        print("COSTS: ", costs.detach())

        return info

    def soft_update(self):
#        print('SOFT UPDATE (params = {})'.format(list(self.target_network.parameters())[0]))
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(
                target_param.data * (1.-self.soft_update_tau) + param.data * (self.soft_update_tau)
            )
