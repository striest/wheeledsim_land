import numpy as np
import torch
from torch import nn

from wheeledsim_land.networks.mlp import MLP
from wheeledsim_land.networks.cnn_blocks.cnn_blocks import ResnetBlock

"""
Network that takes in an image and waypoint (as a 3-vector x,y,th) and produces probablilities
"""

class ResnetWaypointNet(nn.Module):
    def __init__(self, insize, outsize, n_blocks, mlp_hiddens, hidden_activation=nn.Tanh, dropout=0.0, pool=2, image_key='image_rgb', waypoint_key='waypoint', steer_angle_key='steering_angle', device='cpu'):
        """
        Args:
            insize: The size of the input images. Expects a 3-tuple (nchannels, height, width)
            outsize: A scalar, same as MLP
            n_blocks: The number of CNN blocks to use.
            The rest is the same as MLP
        """
        super(ResnetWaypointNet, self).__init__()
        self.cnn_insize = insize
        self.in_channels = insize[0]
        self.outsize = outsize

        self.cnn = nn.ModuleList()
        for i in range(n_blocks):
            self.cnn.append(ResnetBlock(in_channels=self.in_channels * 2**(i), out_channels=self.in_channels * 2**(i+1), pool=pool))
            self.cnn.append(nn.Dropout(p=dropout))
        self.cnn = torch.nn.Sequential(*self.cnn)

        with torch.no_grad():
            self.mlp_insize = self.cnn(torch.zeros(1, *insize)).flatten(start_dim=-3).shape[-1] + 4

        self.mlp = MLP(self.mlp_insize, outsize, mlp_hiddens, hidden_activation, dropout, device)

        self.image_key = image_key
        self.waypoint_key = waypoint_key
        self.steer_angle_key = steer_angle_key

    def forward(self, x):
        img = x[self.image_key]
        waypt = x[self.waypoint_key]
        sang = x[self.steer_angle_key]
        print(img.shape, waypt.shape, sang.shape)

        cnn_out = self.cnn.forward(img)
        mlp_img_in = cnn_out.flatten(start_dim=-3)
        mlp_in = torch.cat([mlp_img_in, waypt, sang], dim=-1)
        out = self.mlp.forward(mlp_in)
        return out

    def to(self, device):
        self.device = device
        self.cnn.to(device)
        self.mlp.to(device)
        return self
