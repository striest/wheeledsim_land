import gym
import torch

from wheeledsim_land.util.util import dict_stack, dict_to

class InterventionReplayBuffer:
    """
    Replay buffer for learning from interventions. Allows user to specify a sampling distribution of the intervention signal
    This is achieved by maintaining a list of indices for intervention/non-intervention and using that to sample.
    Expected that 0 = no intervention, 1 = intervention
    """
    def __init__(self, env, intervention_label='intervention', intervention_prob=0.5, capacity = int(1e7), frame_offset=0, device='cpu'):
        assert isinstance(env.observation_space, gym.spaces.Dict), 'Expects an env with dictionary observations'
        assert isinstance(env.action_space, gym.spaces.Box), 'Expects an env with continuous actions (not dictionary)'

        self.capacity = int(capacity)
        self.obs_dims  = {k:v.shape for k, v in env.observation_space.spaces.items()}
        self.n = 0 #the index to start insering into
        self.device = device

        #The actual buffer is a dict that stores torch tensors. 
        self.act_dim = env.action_space.shape[0]
        self.buffer = {
                    'observation': {k:torch.tensor([float('inf')], device=self.device).repeat(self.capacity, *space) for k, space in self.obs_dims.items()},
                    'action': torch.tensor([float('inf')], device=self.device).repeat(self.capacity, self.act_dim),
                    'reward':torch.tensor([float('inf')], device=self.device).repeat(self.capacity, 1),
                    'next_observation': {k:torch.tensor([float('inf')], device=self.device).repeat(self.capacity, *space) for k, space in self.obs_dims.items()},
                    'terminal': torch.tensor([True], device=self.device).repeat(self.capacity, 1)
                    }
        self.intervention_label = intervention_label
        self.intervention_prob = intervention_prob
        self.intervention = torch.ones(self.capacity).bool()
        self.frame_offset = frame_offset

        self.to(self.device)

    def insert(self, samples):
        assert len(samples['action']) == len(samples['reward']) == len(samples['terminal']), \
        "expected all elements of samples to have same length, got: {} (\'returns\' should be a different length though)".format([(k, len(samples[k])) for k in samples.keys()])

        nsamples = len(samples['action'])

        for i in range(nsamples):
            self.intervention[(self.n + i) % self.capacity] = (abs(samples['observation'][self.intervention_label][i]-1) < 1e-4)

        for k in self.buffer.keys():
            if k == 'observation' or k == 'next_observation':
                for i in range(nsamples):
                    for kk in samples[k].keys():
                        self.buffer[k][kk][(self.n + i) % self.capacity] = samples[k][kk][i]
            else:
                for i in range(nsamples):
                    self.buffer[k][(self.n + i) % self.capacity] = samples[k][i]

        self.n += nsamples

    def get_intervention_samples(self, N):
        """
        Get the number of intervention and non-intervention samples in the buffer.
        REturns:
            Tuple of (#interventions, #non-interventions)
        """
        sample_idxs = self.compute_sample_idxs(N + self.frame_offset)

        #Find all timesteps where an intervention occurred in the next T timesteps
        mask1 = torch.stack([self.intervention[(sample_idxs+i)%self.capacity] for i in range(N)], dim=-1).any(dim=-1)
        #Find all timesteps that have interventions within the frame offset
#        mask2 = torch.stack([self.intervention[(sample_idxs+i)%self.capacity] for i in range(self.frame_offset+1)], dim=-1).any(dim=-1)
        mask2 = self.intervention[sample_idxs]

        #THIS IS A HACK THAT WILL NOT WORK IF THERE IS TURN-IN-PLACE
        mask3 = (self.buffer['action'][sample_idxs, 0] - 0.5) > 0.

        #Intervention samples that are non-intervention but become interventions
        intervention_samples = sample_idxs[mask1 & ~mask2 & mask3]
        #Non-interventions are samples that dont contain an intervention in the next N+frame_offset samples
        non_intervention_samples = sample_idxs[~mask1 & mask3]

        return intervention_samples, non_intervention_samples

    def can_sample(self, N):
        """
        Get a batch of samples from the replay buffer.
        Index output as: [batch x time x feats]
        Note that we only want to sample when it causes an intervention
        """
        intervention_samples, non_intervention_samples = self.get_intervention_samples(N)

        return (len(intervention_samples) > 0) and (len(non_intervention_samples) > 0)

    def sample(self, nsamples, N):
        """
        Get a batch of samples from the replay buffer.
        Index output as: [batch x time x feats]
        Note that we only want to sample when it causes an intervention
        """
        intervention_samples, non_intervention_samples = self.get_intervention_samples(N)

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

    def compute_sample_idxs(self, N):
        all_idxs = torch.arange(min(len(self) + 1, self.capacity)).to(self.device)

        #To handle wrapping properly, we also need to say that the current idx is terminal
        terminal_idxs = torch.nonzero(self.buffer['terminal'][:len(self)])[:, 0]
        terminal_idxs = torch.cat([terminal_idxs, torch.tensor([self.n % self.capacity]).to(self.device)], dim=0)

        #This is not correct. 
#        if self.n > self.capacity:
#            all_idxs = torch.arange(terminal_idxs[-1]+1).to(self.device)

        non_sample_idxs = torch.tensor([]).long().to(self.device)
        for i in range(N-1):
            non_sample_idxs = torch.cat([non_sample_idxs, (terminal_idxs - i) % self.capacity])

        #https://stackoverflow.com/questions/55110047/finding-non-intersection-of-two-pytorch-tensors
        combined = torch.cat((all_idxs, non_sample_idxs))
        uniques, counts = combined.unique(return_counts=True)
        sample_idxs = uniques[counts == 1]

        return sample_idxs.long()

    def sample_idxs(self, idxs):
        idxs = idxs.to(self.device)
        out = {k:{kk:self.buffer[k][kk][idxs] for kk in self.buffer[k].keys()} if (k == 'observation' or k == 'next_observation') else self.buffer[k][idxs] for k in self.buffer.keys()}
        return out

    def __len__(self):
        return min(self.n, self.capacity)

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
        try:
            batch = buf2.sample(16, 10)
        except:
            import pdb;pdb.set_trace()

    print(batch['observation']['intervention'].squeeze())
