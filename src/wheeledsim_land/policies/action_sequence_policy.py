import numpy as np
import torch
import argparse

from wheeledsim_land.util.util import dict_repeat, dict_stack, dict_to_torch, dict_map

class RandomActionSequencePolicy:
    """
    Policy that acts by selecting an action sequence to run for k timesteps.
    To keep consistency with other non-hierarchical methods, provide the switching rate.
    Store current action sequence and step once every time action is called.

    action_sequences expected to be a tensor of size: [choicedim x timedim x actdim]
    """
    def __init__(self, env, action_sequences, device='cpu'):
        self.act_dim = action_sequences.shape[-1]
        self.sequences = action_sequences
        self.T = action_sequences.shape[1]
        self.t = 0
        self.current_sequence = None
        self.n_seqs = self.sequences.shape[0]
        self.device = device

    def to(self, device):
        self.sequences = self.sequences.to(device)
        self.device = device
        return self

    def action(self, obs, deterministic=False):
        if self.t % self.T == 0:
            self.seq_idx = torch.randint(self.n_seqs, size=(1,)).item()
            self.current_sequence = self.sequences[self.seq_idx]
            print('IDX = {}'.format(self.seq_idx))
            print('SEQ = {}'.format(self.current_sequence))
            self.t = 0

        act = self.current_sequence[self.t]
        self.t += 1
        return act.to(self.device)

class InterventionMinimizePolicy(RandomActionSequencePolicy):
    """
    Pass in action sequences and a network that minimizes intervention probs
    """
    def __init__(self, env, action_sequences, net, image_key='image_rgb', device='cpu'):
        RandomActionSequencePolicy.__init__(self, env, action_sequences, device)
        self.net = net
        self.image_key = image_key
        self.probs = torch.zeros(action_sequences.shape[0])
        self.logits = torch.zeros(action_sequences.shape[0])

    def get_intervention_probs(self, obs):
        #img = obs[self.image_key]
        net_in = dict_map(obs, lambda x: x.unsqueeze(0))
        with torch.no_grad():
            preds = self.net.forward(net_in).squeeze()

        return torch.sigmoid(preds), preds

    def action(self, obs, deterministic=False, return_info=False):
        if self.t % self.T == 0:
            self.probs, self.logits = self.get_intervention_probs(obs)
            self.seq_idx = self.probs.argmin()
            self.current_sequence = self.sequences[self.seq_idx]
#            print("PROBS = {}".format(self.probs))
#            print('IDX = {}'.format(self.seq_idx))
#            print('SEQ = {}'.format(self.current_sequence))
            self.t = 0

        act = self.current_sequence[self.t]
        self.t += 1

        if return_info:
            info = {
                'act':act.to(self.device),
                'probs': self.probs.detach().cpu(),
                'logits': self.logits.detach().cpu()
            }
            return act.to(self.device), info
        else:
            return act.to(self.device)

    def to(self, device):
        self.sequences = self.sequences.to(device)
        self.device = device
        return self

class EnsembleInterventionMinimizePolicy(RandomActionSequencePolicy):
    """
    Pass in action sequences and a network that minimizes intervention probs
    """
    def __init__(self, env, action_sequences, nets, lam=0.0, image_key='image_rgb', device='cpu'):
        super(EnsembleInterventionMinimizePolicy, self).__init__(env, action_sequences, device)
        self.nets = nets
        self.image_key = image_key
        self.lam = lam # + -> explore, - -> exploit

    def get_intervention_probs(self, obs):
        img = obs[self.image_key]

        all_preds = []
        with torch.no_grad():
            for net in self.nets:
                preds = net.forward(img.unsqueeze(0)).squeeze()
                all_preds.append(preds)

        all_preds = torch.stack(all_preds, dim=0)

        return torch.sigmoid(all_preds)

    def action(self, obs, deterministic=False):
        if self.t % self.T == 0:
            probs = self.get_intervention_probs(obs)

            mean_prob = probs.mean(dim=0)
            uncertainty = probs.std(dim=0)
            scores = mean_prob + self.lam * uncertainty

            self.seq_idx = scores.argmin()
            self.current_sequence = self.sequences[self.seq_idx]

            print('IDX = {}'.format(self.seq_idx))
            print('SEQ = {}'.format(self.current_sequence))
            self.t = 0

        act = self.current_sequence[self.t]
        self.t += 1
        return act.to(self.device)

    def to(self, device):
        self.sequences = self.sequences.to(device)
        self.device = device
        return self
