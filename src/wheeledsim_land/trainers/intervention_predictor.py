import torch

from wheeledsim_land.replay_buffers.intervention_replay_buffer import InterventionReplayBuffer

class InterventionPredictionTrainer:
    """
    Train a network to predict probabilities of intervention from images.
    """
    def __init__(self, policy, network, buf, opt, T=10, tscale=1.0, sscale=-0.6):
        self.policy = policy
        self.network = network
        self.buf = buf
        self.opt = opt

        self.T = T
        self.scaling = torch.tensor([tscale, sscale]).to(self.buf.device)
        self.scaled_acts = self.policy.sequences[:, 0] * self.scaling.unsqueeze(0)

    def update(self):
        batchsize=8
        batch = self.buf.sample(batchsize, self.T)

        seq_idxs = torch.argmin(torch.norm(batch['action'][:, 0].unsqueeze(1) - self.scaled_acts.unsqueeze(0), dim=-1), dim=-1)

#        print(self.scaling)
#        print('SACT:', self.scaled_acts)
#        print('ACTS:', batch['action'][:, 0])
#        print('SEQS:', seq_idxs)

        _x = batch['observation']['image_rgb'][:, 0]
        _y = (batch['observation']['intervention'].abs() > 1e-2).any(dim=1).squeeze().float()
        _ypred = self.network.forward(_x)
        _ypred_seq = _ypred[torch.arange(batchsize), seq_idxs]

        print(_y)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(_ypred_seq, _y)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        print('LOSS = {:.6f}'.format(loss.detach().item()))

