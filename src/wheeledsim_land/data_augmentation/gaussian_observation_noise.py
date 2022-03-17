import torch

class GaussianObservationNoise:
    """
    Implement random noise on observations
    """
    def __init__(self, noise_scale):
        """
        Args:
            keys: A {obs key:sigma} dictionary describing the noise scale to apply to each observation key
        """
        self.noise_scale = noise_scale

    def forward(self, x):
        # TODO: Figure out if it's a good idea to apply different noises to obs and nobs.
        obs_noise = {k:torch.randn_like(x['observation'][k]) * v for k,v in self.noise_scale.items()}
        nobs_noise = {k:torch.randn_like(x['observation'][k]) * v for k,v in self.noise_scale.items()}

        for k in self.noise_scale.keys():
            x['observation'][k] += obs_noise[k]
            x['next_observation'][k] += nobs_noise[k]

        return x
