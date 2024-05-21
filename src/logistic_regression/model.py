import torch


class BayesianLogisticRegressionLogJoint(torch.nn.Module):
    def __init__(self, prior_mean, prior_cholesky_factor):
        super(BayesianLogisticRegressionLogJoint, self).__init__()
        #
        self.register_buffer("prior_mean", prior_mean)
        self.register_buffer("prior_cholesky_factor", prior_cholesky_factor)
        self.prior_distribution_weights = torch.distributions.MultivariateNormal(
            loc=self.prior_mean, scale_tril=self.prior_cholesky_factor
        )
        self.prior_distribution_intercept = torch.distributions.Normal(
            loc=torch.zeros(1), scale=torch.tensor([1.0])
        )

    def log_likelihood(self, z, x, y):
        logits = (-(z[..., :-1] * x + z[..., -1].unsqueeze(-1))).sum(-1)
        log_likelihood = torch.distributions.Bernoulli(logits=logits).log_prob(
            y.float()
        )
        return log_likelihood

    def log_prior(self, z):
        weights_prior = self.prior_distribution_weights.log_prob(z[..., :-1])
        intercept_prior = self.prior_distribution_intercept.log_prob(z[..., -1])
        return weights_prior + intercept_prior

    def log_prob(self, z, x, y, **kwargs):
        log_prior = self.log_prior(z)
        log_likelihood = self.log_likelihood(z, x, y).sum(0, keepdims=True)
        log_prob = log_prior + log_likelihood
        return log_prob

    def all_log_probs_and_reconstruction(self, z, **kwargs):
        x = kwargs["x"]
        y = kwargs["y"]
        log_prob = self.log_prob(z, x, y)
        return log_prob, *[torch.tensor(torch.nan).to(z.device)] * 3


class LogisticRegressionAISModel(torch.nn.Module):
    def __init__(self, args, log_joint, log_variational, ais):
        super(LogisticRegressionAISModel, self).__init__()
        self.q = log_variational

        self.B = args.batch_size
        self.N = args.n_particles
        self.D = args.zdim

        self.ais = ais

    def forward(self, x, y, **kwargs):
        z = self.q.sample((1, self.N))
        x = x.unsqueeze(1).repeat(1, self.N, 1)
        y = y.unsqueeze(1).repeat(1, self.N)
        return_dict = self.ais(z, x=x, y=y, **kwargs)
        return return_dict
