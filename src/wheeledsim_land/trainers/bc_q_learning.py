import torch
import numpy as np
import copy
import time

from wheeledsim_land.replay_buffers.intervention_replay_buffer import InterventionReplayBuffer
from wheeledsim_land.util.util import dict_map

class BCQLearningTrainer:
    """
    Use behavioral cloning to supervise a Q function (using max margin)
    I will implement this such that Q is cost.
    Unlike most of the other trainers, only work with the intervention data.

    NOTE THAT T HERE SHOULD MATCH THE ACTION SELECTION RATE
    """
    def __init__(self, policy, network, buf, opt, augmentations, margin=1.0, discount=0.99, soft_update_tau=0.05, batchsize=16, T=10, tscale=1.0, sscale=-0.6):
        """
        Args:
            policy: The policy to train
            network: The policy's network (This is actually what updates)
            buf: Replay buffer to train on
            opt: The optimizer for the net
            augmentations: A list of data augmentations to apply to the dataset
            margin: The desired cost difference between the human action and the other actions.
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

        self.margin = margin
        self.discount = discount
        self.soft_update_tau = soft_update_tau

        self.batchsize = batchsize
        self.T = T
        self.scaling = torch.tensor([tscale, sscale]).to(self.buf.device)
        self.scaled_acts = self.policy.sequences[:, 0] * self.scaling.unsqueeze(0)

        self.target_network = copy.deepcopy(self.network)

    def update(self):
        ts = time.time()
        batch = self.buf.sample(self.batchsize, self.T)
        aug = np.random.choice(self.augmentations)
        batch = aug.forward(batch)

        #Consider assigning highest magnitude in the slice to get better label distribution
        max_steer_idx = batch['action'][:, :, 1].abs().argmax(dim=1)
        seq_idxs = torch.argmin(torch.norm(batch['action'][torch.arange(self.batchsize), max_steer_idx].unsqueeze(1) - self.scaled_acts.unsqueeze(0), dim=-1), dim=-1)

        #_x = batch['observation']['image_rgb'][:, 0]

        s_curr = dict_map(batch['observation'], lambda x: x[:, 0])
        s_next = dict_map(batch['next_observation'], lambda x:x[:, -1])

#        print({k:v.shape for k,v in s_curr.items()})
        intervention = (batch['observation']['intervention'].abs() > 1e-2).any(dim=1).squeeze().float()

#        print(self.scaling)
#        print('SACT:', self.scaled_acts)
#        print('ACTS + LABELS:', torch.cat([batch['action'][:, 0], intervention.unsqueeze(-1)], dim=-1))
#        print('SEQS:', seq_idxs)
#        print('RAW LABELS:', batch['observation']['intervention'].abs() )

        """
        Steps:
            1. Calculate reward (since expert, always 0 cost)
            2. Calculate Q estimates
            3. Compute Bellman backup
            4. Compute max-margin loss
            5. Minimize max-margin and DQN losses
        """
        q_curr = self.network.forward(s_curr)
        q_curr_acts = q_curr[torch.arange(self.batchsize), seq_idxs]
        with torch.no_grad():
            q_next = self.target_network.forward(s_next).min(dim=-1)[0]

        q_target = (self.discount * q_next)

        dqn_loss = (q_curr_acts - q_target).pow(2).mean()

        #Max-margin loss. L = Q(s, a_e) - min_a[Q(s, a) - m(a, a_e)]
        #i.e. loss is 0 if the lowest-cost action is expert. Loss is the margin if lowest-cost action is not expert
        margin = torch.ones_like(q_curr)
        margin[torch.arange(self.batchsize), seq_idxs] = 0.
        min_margin_cost = torch.min(q_curr - margin, dim=-1)[0]
        margin_loss = (q_curr_acts - min_margin_cost).mean()

        loss = dqn_loss + 1.0 * margin_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.soft_update()

        te = time.time() - ts
        info = {
            'loss':loss.detach().cpu().item(),
            'dqn_loss': dqn_loss.detach().cpu().item(),
            'margin_loss': margin_loss.detach().cpu().item(),
            'train time': te
        }

#        print("QS:", q_curr)

        return info

    def soft_update(self):
#        print('SOFT UPDATE (params = {})'.format(list(self.target_network.parameters())[0]))
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(
                target_param.data * (1.-self.soft_update_tau) + param.data * (self.soft_update_tau)
            )
