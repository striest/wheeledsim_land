import torch
import numpy as np
import time

from wheeledsim_land.replay_buffers.intervention_replay_buffer import InterventionReplayBuffer
from wheeledsim_land.util.util import dict_map

class InterventionPredictionTrainer:
    """
    Train a network to predict probabilities of intervention from images.
    """
    def __init__(self, policy, network, buf, opt, augmentations, batchsize=16, T=10, tscale=1.0, sscale=-0.6):
        """
        Args:
            policy: The policy to train
            network: The policy's network (This is actually what updates)
            buf: Replay buffer to train on
            opt: The optimizer for the net
            augmentations: A list of data augmentations to apply to the dataset
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

        self.batchsize = batchsize
        self.T = T
        self.scaling = torch.tensor([tscale, sscale]).to(self.buf.device)
        self.scaled_acts = self.policy.sequences[:, 0] * self.scaling.unsqueeze(0)

    def update(self):
        ts = time.time()
        batch = self.buf.sample(self.batchsize, self.T)
        aug = np.random.choice(self.augmentations)
        batch = aug.forward(batch)

        seq_idxs = torch.argmin(torch.norm(batch['action'][:, 0].unsqueeze(1) - self.scaled_acts.unsqueeze(0), dim=-1), dim=-1)

#        print(self.scaling)
#        print('SACT:', self.scaled_acts)
#        print('ACTS:', batch['action'][:, 0])
#        print('SEQS:', seq_idxs)

        #_x = batch['observation']['image_rgb'][:, 0]

        _x = dict_map(batch['observation'], lambda x: x[:, 0])
        print({k:v.shape for k,v in _x.items()})
        _y = (batch['observation']['intervention'].abs() > 1e-2).any(dim=1).squeeze().float()
        _ypred = self.network.forward(_x)
        _ypred_seq = _ypred[torch.arange(self.batchsize), seq_idxs]

        loss = torch.nn.functional.binary_cross_entropy_with_logits(_ypred_seq, _y)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        te = time.time() - ts
        info = {
            'loss':loss.detach().cpu().item(),
            'train time': te
        }

        return info


