import torch

from wheeledsim_rl.networks.cnn_blocks.cnn_blocks import ResnetCNN

from wheeledsim_land.replay_buffers.intervention_replay_buffer import InterventionReplayBuffer

class EnsembleInterventionPredictionTrainer:
    """
    Train an ensemble of networks to predict probabilities of intervention from images.
    """
    def __init__(self, policy, networks, buf, opts, T=10, tscale=1.0, sscale=-0.6):
        self.policy = policy
        self.networks = networks
        self.buf = buf
        self.opts = opts

        self.T = T
        self.scaling = torch.tensor([tscale, sscale]).to(self.buf.device)
        self.scaled_acts = self.policy.sequences[:, 0] * self.scaling.unsqueeze(0)

    def update(self):
        #Think about parallelizing this?
        for i, (net, opt) in enumerate(zip(self.networks, self.opts)):
            print('Net {}/{}'.format(i+1, len(self.networks)))
            self.update_one(net, opt)

    def update_one(self, network, opt):
        batchsize=32
        batch = self.buf.sample(batchsize, self.T)

        seq_idxs = torch.argmin(torch.linalg.norm(batch['action'][:, 0].unsqueeze(1) - self.scaled_acts.unsqueeze(0), dim=-1), dim=-1)

#        print('SACT:', self.scaled_acts)
#        print('ACTS:', batch['action'][:, 0])
#        print('SEQS:', seq_idxs)

        _x = batch['observation']['image_rgb'][:, 0]
        _y = batch['next_observation']['intervention'].any(dim=1).squeeze().float()
        _ypred = network.forward(_x)
        _ypred_seq = _ypred[torch.arange(batchsize), seq_idxs]

        loss = torch.nn.functional.binary_cross_entropy_with_logits(_ypred_seq, _y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        print('LOSS = {:.6f}'.format(loss.detach().item()))

if __name__ == '__main__':
    from rosbag_to_dataset.config_parser.config_parser import ConfigParser
    from wheeledsim_rl.util.util import dict_map

    from wheeledsim_land.policies.to_joy import ToJoy
    from wheeledsim_land.policies.action_sequences import generate_action_sequences
    from wheeledsim_land.policies.action_sequence_policy import RandomActionSequencePolicy


    config_parser = ConfigParser()
    spec, converters, remap, rates = config_parser.parse_from_fp('../../../configs/pybullet_land.yaml')

    buf2 = InterventionReplayBuffer(spec, capacity=5000)
    buf = torch.load('buffer.pt')

    data = buf.sample_idxs(torch.arange(len(buf)))
    buf2.insert(data)

    net = ResnetCNN(insize=[3, 64, 64], outsize=5, n_blocks=2, pool=4, mlp_hiddens=[32, ])
    opt = torch.optim.Adam(net.parameters())

    seqs = generate_action_sequences(throttle=(1, 1), throttle_n=1, steer=(-1, 1), steer_n=5, t=10)
    policy = RandomActionSequencePolicy(env=None, action_sequences=seqs)

    trainer = InterventionPredictionTrainer(policy, net, buf, opt)

    for i in range(10000):
        trainer.update()

    torch.save(net, 'net.pt')
