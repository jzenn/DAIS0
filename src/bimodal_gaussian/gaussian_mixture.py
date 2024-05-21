import torch
from torch import nn


class FixedBimodalGaussianMixture(nn.Module):
    def __init__(self, means, covariances, weights, device):
        super(FixedBimodalGaussianMixture, self).__init__()
        #
        self.register_buffer("means", torch.stack(means))
        self.register_buffer("covariances", torch.stack(covariances))
        self.register_buffer("weights", weights)
        #
        self.distributions = [
            torch.distributions.MultivariateNormal(
                loc=self.means[i].to(device),
                covariance_matrix=self.covariances[i].to(device),
            )
            for i in range(len(self.weights))
        ]

    def log_prob(self, z, **kwargs):
        num_components = len(self.weights)
        log_probs = torch.zeros(num_components, z.shape[0], z.shape[1], device=z.device)

        for i in range(num_components):
            log_probs[i] = torch.log(self.weights[i]) + self.distributions[i].log_prob(
                z
            )

        stable_log_probs = torch.logsumexp(log_probs, dim=0)

        return stable_log_probs

    def all_log_probs_and_reconstruction(self, z, **kwargs):
        log_prob = self.log_prob(z, **kwargs)
        return log_prob, *[torch.tensor(torch.nan).to(z.device)] * 3


class FixedBimodalGaussianMixtureAISModel(nn.Module):
    def __init__(self, args, log_joint, log_variational, ais, device):
        super(FixedBimodalGaussianMixtureAISModel, self).__init__()
        self.q = log_variational

        self.B = args.batch_size
        self.N = args.n_particles
        self.D = args.zdim

        self.ais = ais
        self.device = device

    def forward(self, **kwargs):
        z = self.q.sample((self.B, self.N)).to(self.device)
        return_dict = self.ais(z, x=None, **kwargs)
        return return_dict
