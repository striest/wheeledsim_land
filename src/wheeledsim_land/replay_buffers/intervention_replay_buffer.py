import gym
import torch

from wheeledsim_rl.util.util import dict_stack, dict_to
from wheeledsim_rl.replaybuffers.dict_replaybuffer import NStepDictReplayBuffer

class InterventionReplayBuffer(NStepDictReplayBuffer):
    """
    Replay buffer for learning from interventions. Allows user to specify a sampling distribution of the intervention signal
    This is achieved by maintaining a list of indices for intervention/non-intervention and using that to sample.
    Expected that 0 = no intervention, 1 = intervention
    """
    def __init__(self, env, intervention_label='intervention', intervention_prob=0.5, capacity = int(1e7), device='cpu'):
        super(NStepDictReplayBuffer, self).__init__(env, capacity, device)
        self.intervention_label = intervention_label
        self.intervention_prob = intervention_prob
        self.intervention = torch.ones(self.capacity).bool()
        self.to(self.device)

    def insert(self, samples):
        nsamples = len(samples['action'])
        for i in range(nsamples):
            self.intervention[(self.n + i) % self.capacity] = (abs(samples['observation'][self.intervention_label][i]-1) < 1e-4)

        super(InterventionReplayBuffer, self).insert(samples)

    def sample(self, nsamples, N):
        """
        Get a batch of samples from the replay buffer.
        Index output as: [batch x time x feats]
        Note that we only want to sample when it causes an intervention
        """
        sample_idxs = self.compute_sample_idxs(nsamples, N)

        mask1 = torch.stack([self.intervention[(sample_idxs+i)%self.capacity] for i in range(N)], dim=-1).any(dim=-1)
        mask2 = self.intervention[sample_idxs]
        intervention_samples = sample_idxs[mask1 & ~mask2]
        non_intervention_samples = sample_idxs[~mask1]

        k1 = int(nsamples*self.intervention_prob)
        k2 = nsamples - k1

        if len(intervention_samples) == 0:
            idxs = non_intervention_samples[torch.randint(len(non_intervention_samples), size=(nsamples, ))]

        elif len(non_intervention_samples) == 0:
            idxs = intervention_samples[torch.randint(len(intervention_samples), size=(nsamples, ))]

        else:
            intervention_idxs = intervention_samples[torch.randint(len(intervention_samples), size=(k1, ))]
            non_intervention_idxs = non_intervention_samples[torch.randint(len(non_intervention_samples), size=(k2, ))]

            idxs = torch.cat([intervention_idxs, non_intervention_idxs])

        outs = [self.sample_idxs((idxs + i) % len(self)) for i in range(N)]
        out = dict_stack(outs, dim=1)

        return out

    def compute_sample_idxs(self, nsamples, N):
        all_idxs = torch.arange(min(len(self), self.capacity)).to(self.device)
        terminal_idxs = torch.nonzero(self.buffer['terminal'][:len(self)], as_tuple=False)[:, 0]
        if self.n > self.capacity:
            all_idxs = torch.arange(terminal_idxs[-1]+1).to(self.device)

        non_sample_idxs = torch.tensor([]).long().to(self.device)
        for i in range(N-1):
            non_sample_idxs = torch.cat([non_sample_idxs, terminal_idxs - i])

        #https://stackoverflow.com/questions/55110047/finding-non-intersection-of-two-pytorch-tensors
        combined = torch.cat((all_idxs, non_sample_idxs))
        uniques, counts = combined.unique(return_counts=True)
        sample_idxs = uniques[counts == 1]

        return sample_idxs.long()

    def to(self, device):
        self.device = device
        self.buffer = dict_to(self.buffer, self.device)
        self.intervention = self.intervention.to(self.device)
        return self

if __name__ == '__main__':
    from rosbag_to_dataset.config_parser.config_parser import ConfigParser
    from wheeledsim_rl.util.util import dict_map

    config_parser = ConfigParser()
    spec, converters, remap, rates = config_parser.parse_from_fp('../../../configs/pybullet_land.yaml')

    buf2 = InterventionReplayBuffer(spec, capacity=5000)
    buf = torch.load('buffer.pt')

    data = buf.sample_idxs(torch.arange(len(buf)))
    buf2.insert(data)

    batch = buf2.sample(16, 10)

    print(batch['observation']['intervention'].squeeze())
