import matplotlib.pyplot as plt
import torch
import numpy as np

from wheeledsim_rl.replaybuffers.dict_replaybuffer import NStepDictReplayBuffer
from wheeledsim_rl.util.util import dict_map, dict_to

class ReplayBufferViz:
    def __init__(self, buf):
        self.buf = buf
        self.fig, self.axes = self.init_figs(buf)

    def init_figs(self, buf):
        """
        Initialize a play fig, and a certain number of side figs.
        """
        n_figs = 2 + len(buf.buffer['observation'].keys())
        keys = list(buf.buffer['observation'].keys()) + ['action']

        figwidth = 2 + 3* n_figs//2
        figheight = 6

        spacing = 0.1
        wdx = 0.9 / int(n_figs/2) - spacing - 1/figwidth
        hdx = 0.9 / 2 - spacing

        fig = plt.figure('buffer_viz', figsize = (figwidth, figheight))
        axes = {}

        #Add a vertical progress fig and figs for each of the data
        progress_ax = fig.add_axes([0.1, 0.1, 1/figwidth, 0.8])

        progress_ax.set_title('Buffer')
        progress_ax.set_xlim(0, 1)
        progress_ax.set_ylim(-0.5, buf.capacity + 0.5)
        progress_ax.axhline(0)
        progress_ax.text(0.5, 0, '0')
        axes['buffer'] = progress_ax

        for i, k in enumerate(keys):
            hi = i%2
            wi = int(i/2)
            print([0.1 + spacing + wi*(wdx+spacing), 0.1 + spacing + hi*(hdx+spacing), wdx, hdx])
            ax = fig.add_axes([0.1 + 1/figwidth + spacing + wi*(wdx+spacing), 0.1 + hi*(hdx+spacing), wdx, hdx])
            ax.set_title(k)
            axes[k] = ax

        return fig, axes

    def update_figs(self):
        """
        Update the viz for the current state of buf
        """
        idx = (self.buf.n-1) % self.buf.capacity
        current_obs = dict_to(dict_map(self.buf.sample_idxs(torch.tensor([idx])), lambda x:x[0]), 'cpu')
        current_obs['observation']['action'] = current_obs['action']
        current_obs = current_obs['observation']

        #Update the buf viz.
        progress_ax = self.axes['buffer']
        progress_ax.cla()
        progress_ax.set_title('buffer')
        progress_ax.set_xlim(0, 1)
        progress_ax.set_ylim(-0.5, self.buf.capacity + 0.5)
        progress_ax.axhline(idx)
        progress_ax.text(0.5, idx, idx)

        for k, ax in self.axes.items():
            if k == 'buffer':
                continue

            ax.cla()
            data = current_obs[k]
            ax.set_title(k)
            if len(data.shape) == 3:
                #Assume image
                ax.imshow(data.permute(1, 2, 0))
            elif len(data.shape) == 1:
                if k == 'action':
                    ax.arrow(0., 0., data[1], data[0])
                    ax.set_xlim(-1.1, 1.1)
                    ax.set_ylim(-0.1, 1.1)
                elif k == 'intervention':
                    ax.scatter(0, data, marker='x', c='r')
                    ax.set_ylim(-0.1, 1.1)

