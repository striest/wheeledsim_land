import torch
import numpy as np
import matplotlib.pyplot as plt

class InterventionPredictionViz:
    """
    Predict intervention predictions for a nn
    need to pass image in the update func
    """
    def __init__(self, net, seqs):
        self.net = net
        self.seqs = seqs
        self.fig, self.axs = plt.subplots(1, 2, figsize=(6, 3))
        self.fig.suptitle('Intervention Prediction')

    def update(self, img):
        for ax in self.axs:
            ax.cla()

        self.axs[0].set_title('Image')
        self.axs[1].set_title('Intervention Pred')
        self.axs[1].set_ylim(-0.1, 1.1)

        self.axs[0].imshow(img.permute(1, 2, 0).cpu(), extent=(0., 1., 0., 1.))

        with torch.no_grad():
            preds = self.net.forward(img.unsqueeze(0))
            probs = torch.sigmoid(preds.squeeze().flip(dims=[0]))
            self.axs[1].plot(probs.cpu())

            #Lol jank
            self.axs[0].arrow([0., 0.], [probs.argmin().cpu().item() / len(probs), 0.8], c='r', label='policy', width=0.1)
