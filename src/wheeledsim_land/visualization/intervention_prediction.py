import torch
import numpy as np
import matplotlib.pyplot as plt

class InterventionPredictionViz:
    """
    Predict intervention predictions for a nn
    need to pass image in the update func
    """
    def __init__(self, policy):
        self.policy = policy

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
            probs = self.policy.get_intervention_probs({self.policy.image_key:img})
            probs = torch.sigmoid(probs.squeeze().flip(dims=[0]))
            self.axs[1].plot(probs.cpu())

            #Lol jank
            self.axs[0].arrow(0.5, 0., (probs.argmin().cpu().item() - int(len(probs)/2)) / len(probs), 0.8, color='r', label='policy', width=0.02)

class EnsembleInterventionPredictionViz:
    """
    Predict intervention predictions for a nn
    need to pass image in the update func
    """
    def __init__(self, policy):
        self.policy = policy

        self.fig, self.axs = plt.subplots(1, 3, figsize=(9, 3))
        self.fig.suptitle('Intervention Prediction')

    def update(self, img):
        for ax in self.axs:
            ax.cla()

        self.axs[0].set_title('Image')
        self.axs[1].set_title('Intervention Pred')
        self.axs[1].set_ylim(-0.1, 1.1)
        self.axs[2].set_title('Costs')
        self.axs[2].set_ylim(-0.1, 1.1)

        self.axs[0].imshow(img.permute(1, 2, 0).cpu(), extent=(0., 1., 0., 1.))

        with torch.no_grad():
            probs = self.policy.get_intervention_probs({self.policy.image_key:img})
            probs = torch.sigmoid(probs.squeeze().flip(dims=[-1]))
            mean_prob = probs.mean(dim=0)
            unc = probs.std(dim=0)
            scores = mean_prob + self.policy.lam * unc
            for prob in probs:
                self.axs[1].plot(prob.cpu())

            self.axs[1].plot(unc.cpu(), linestyle='dashed', marker='.', c='k')

            self.axs[2].plot(scores.cpu(), linestyle='dashed', marker='.', c='k')

            #Lol jank
            self.axs[0].arrow(0.5, 0., (scores.argmin().cpu().item() - int(probs.shape[1]/2)) / probs.shape[1], 0.8, color='r', label='policy', width=0.02)
