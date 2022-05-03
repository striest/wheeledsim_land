import numpy as np
import torch
import argparse

from wheeledsim_land.util.util import dict_repeat, dict_stack, dict_to_torch, dict_map

class DiscreteInterventionMinimizePolicy:
    """
    Pass in action sequences and a network that minimizes intervention probs
    """
    def __init__(self, env, net, image_key='image_rgb', device='cpu'):
        self.net = net
        self.image_key = image_key
        self.probs = torch.zeros(net.outsize)
        self.logits = torch.zeros(net.outsize)
        self.device = device

    def get_intervention_probs(self, obs):
        #img = obs[self.image_key]
        net_in = dict_map(obs, lambda x: x.unsqueeze(0))
        print({k:v.shape for k,v in net_in.items()})
        with torch.no_grad():
            preds = self.net.forward(net_in).squeeze()

        return torch.sigmoid(preds), preds

    def action(self, obs, deterministic=False, return_info=False):
        self.probs, self.logits = self.get_intervention_probs(obs)

        act = self.probs.argmin()

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
        self.net = self.net.to(device)
        self.device = device
        return self
