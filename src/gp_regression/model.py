import torch
from torch import nn


class GPJoint(nn.Module):
    def __init__(
        self,
        prior_mean,
        prior_cholesky_factor,
        prior_covariance,
        observations,
        observation_scale,
    ):
        super(GPJoint, self).__init__()
        #
        self.register_buffer("prior_mean", prior_mean)
        self.register_buffer("prior_covariance", prior_covariance)
        self.register_buffer("prior_cholesky_factor", prior_cholesky_factor)
        self.register_buffer("observations", observations)
        self.register_buffer("observation_scale", observation_scale)
        #
        if prior_cholesky_factor is None:
            self.prior_distribution = torch.distributions.MultivariateNormal(
                loc=self.prior_mean, covariance_matrix=self.prior_covariance
            )
        else:
            self.prior_distribution = torch.distributions.MultivariateNormal(
                loc=self.prior_mean, scale_tril=self.prior_cholesky_factor
            )

    def log_prob(self, z, **kwargs):
        prior = self.prior_distribution.log_prob(z)
        likelihood = torch.distributions.Normal(
            loc=z, scale=self.observation_scale
        ).log_prob(self.observations)
        log_prob = prior + likelihood.sum(-1)
        return log_prob

    def all_log_probs_and_reconstruction(self, z, **kwargs):
        log_prob = self.log_prob(z, **kwargs)
        return log_prob, *[torch.tensor(torch.nan).to(z.device)] * 3


class GPAISModel(nn.Module):
    def __init__(self, args, log_joint, log_variational, ais):
        super(GPAISModel, self).__init__()
        self.q = log_variational

        self.B = args.batch_size
        self.N = args.n_particles
        self.D = args.zdim

        self.ais = ais

    def forward(self, **kwargs):
        z = self.q.sample((self.B, self.N))
        return_dict = self.ais(z, x=None, **kwargs)
        return return_dict
