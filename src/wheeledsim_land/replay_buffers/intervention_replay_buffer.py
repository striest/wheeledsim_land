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
    def __init__(self, env, intervention_label='intervention', intervention_prob=0.5, capacity = int(1e7), frame_offset=0, device='cpu'):
        """
        Args:
            env: The env with the observation/action space to log
            intervention_label: the field of the observation space that contains the intervention data
            intervention_prob: Sample datapoints with intervention=1 with this probability
            capacity: The number of datapoints to store
            frame_offset: Account for human reaction time by sliding the intervention data this many timesteps backward. Note that this only affects the sample data. We still store it unshifted
            device: Whether to store on CPU/GPU
        """
        super(NStepDictReplayBuffer, self).__init__(env, capacity, device)
        self.intervention_label = intervention_label
        self.intervention_prob = intervention_prob
        self.intervention = torch.ones(self.capacity).bool()
        self.frame_offset = frame_offset
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
        sample_idxs = self.compute_sample_idxs(nsamples, N + self.frame_offset)

        #Find all timesteps where an intervention occurred in the next T timesteps
        mask1 = torch.stack([self.intervention[(sample_idxs+i)%self.capacity] for i in range(N)], dim=-1).any(dim=-1)
        #Find all timesteps that have interventions within the frame offset
#        mask2 = torch.stack([self.intervention[(sample_idxs+i)%self.capacity] for i in range(self.frame_offset+1)], dim=-1).any(dim=-1)
        mask2 = self.intervention[sample_idxs]
        #Intervention samples that are non-intervention but become interventions
        intervention_samples = sample_idxs[mask1 & ~mask2]
        #Non-interventions are samples that dont contain an intervention in the next N+frame_offset samples
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

    buf2 = InterventionReplayBuffer(spec, capacity=5000, frame_offset=0)
    buf = torch.load('buffer.pt')

    data = buf.sample_idxs(torch.arange(len(buf)))
    buf2.insert(data)

    for i in range(1000):
        print(i, end='\r')
        try:
            batch = buf2.sample(16, 10)
        except:
            import pdb;pdb.set_trace()

    print(batch['observation']['intervention'].squeeze())
