import torch

from src.distribution.normal import normal_log_prob, sample_normal
from src.sampling.dais import DifferentiableAnnealedImportanceSampling
from src.sampling.momentum import (
    DiagonalMassMatrix,
    MomentumRefreshmentFactor,
    ScaledDiagonalMassMatrix,
)


class LangevinMomentumDiffusionDAISDoucet(DifferentiableAnnealedImportanceSampling):
    def __init__(self, args, log_joint, log_variational, deltas, betas):
        super(LangevinMomentumDiffusionDAISDoucet, self).__init__(
            args,
            log_joint=log_joint,
            log_variational=log_variational,
            deltas=deltas,
            betas=betas,
        )
        # mass matrix
        if args.scaled_M:
            self.mass_matrix = ScaledDiagonalMassMatrix(self.D)
        else:
            self.mass_matrix = DiagonalMassMatrix(self.D)
        # momentum refreshment factor
        self.eta = MomentumRefreshmentFactor()

    def get_initial_momentum(self, z):
        B, N, D = z.shape
        v_0 = sample_normal(
            torch.zeros(N, D, device=z.device), self.mass_matrix.cholesky(), (B, N)
        )
        return v_0

    def initialize_elbo(self, z, **kwargs):
        v = kwargs["v"]
        # log prior
        initial_log_variational_post = self.initial_dist.log_prob(z, **kwargs)
        log_momentum = normal_log_prob(
            v,
            torch.zeros_like(v),
            self.mass_matrix.precision(),
            self.mass_matrix.logdet(),
        )
        # initialize elbo
        elbo = -initial_log_variational_post - log_momentum
        return elbo, initial_log_variational_post + log_momentum

    def get_last_elbo_increment(self, z, **kwargs):
        v = kwargs["v"]
        last_log_joint = kwargs["last_log_joint"]
        # log momentum
        last_log_momentum = normal_log_prob(
            v,
            torch.zeros_like(v),
            self.mass_matrix.precision(),
            self.mass_matrix.logdet(),
        )
        elbo_increment = last_log_joint + last_log_momentum
        return elbo_increment

    def leapfrog_step(self, z, k, **kwargs):
        v = kwargs["v"]
        z = z + self.deltas[k] / 2 * v @ self.mass_matrix.inv()
        v = v + self.deltas[k] * self.grad_log_gamma(self.betas[k], z, k=k, **kwargs)
        z = z + self.deltas[k] / 2 * v @ self.mass_matrix.inv()
        return z, v

    def get_normal_params(self):
        covariance_factor = 1 - self.eta() ** 2
        covariance_diagonal = covariance_factor * self.mass_matrix.diag()
        cholesky_factor = torch.diag(torch.sqrt(covariance_diagonal))
        precision = torch.diag(1.0 / covariance_diagonal)
        logdet = torch.sum(torch.log(covariance_diagonal))
        return {
            "cholesky_factor": cholesky_factor,
            "precision": precision,
            "logdet": logdet,
        }

    def forward_transition(self, v, k, normal_params, **kwargs):
        forward_mean = self.eta() * v
        v_inter = sample_normal(
            forward_mean, normal_params["cholesky_factor"], (v.shape[0], v.shape[1])
        )
        v_inter_forward_log_prob = normal_log_prob(
            v_inter, forward_mean, normal_params["precision"], normal_params["logdet"]
        )
        return v_inter, v_inter_forward_log_prob

    def backward_transition(self, v, v_inter, k, normal_params, **kwargs):
        backward_mean = self.eta() * v_inter
        v_backward_log_prob = normal_log_prob(
            v, backward_mean, normal_params["precision"], normal_params["logdet"]
        )
        return v_backward_log_prob

    def transition(self, z, k, **kwargs):
        v = kwargs["v"]
        normal_params = self.get_normal_params()
        # forward transition
        v_inter, v_inter_forward_log_prob = self.forward_transition(v, k, normal_params)

        # backward transition
        v_backward_log_prob = self.backward_transition(
            v, v_inter, k, normal_params, z=z
        )

        # leapfrog step
        kwargs_ = kwargs.copy()
        kwargs_["v"] = v_inter
        z_new, v_new = self.leapfrog_step(z, k, **kwargs_)
        return z_new, v_new, v_inter_forward_log_prob, v_backward_log_prob


class LangevinMomentumDiffusionDAISZhang(LangevinMomentumDiffusionDAISDoucet):
    def __init__(self, args, log_joint, log_variational, deltas, betas):
        super(LangevinMomentumDiffusionDAISZhang, self).__init__(
            args,
            log_joint=log_joint,
            log_variational=log_variational,
            deltas=deltas,
            betas=betas,
        )

    def initialize_elbo(self, z, **kwargs):
        v = kwargs["v"]
        # log prior
        initial_log_variational_post = self.initial_dist.log_prob(z, **kwargs)
        # initialize elbo
        elbo = -initial_log_variational_post
        return elbo, initial_log_variational_post

    def get_last_elbo_increment(self, z, **kwargs):
        last_log_joint = kwargs["last_log_joint"]
        elbo_increment = last_log_joint
        return elbo_increment

    def forward_transition(self, v_inter, k, normal_params, **kwargs):
        forward_mean = self.eta() * v_inter
        v_new = sample_normal(
            forward_mean,
            normal_params["cholesky_factor"],
            (v_inter.shape[0], v_inter.shape[1]),
        )
        return v_new

    def transition(self, z, k, **kwargs):
        v = kwargs["v"]
        z_new, v_inter = self.leapfrog_step(z, k, **kwargs)
        v_new = self.forward_transition(v_inter, k, self.get_normal_params())

        # normal distribution for transitions
        mass_matrix_diag = self.mass_matrix.diag()
        zeros = torch.zeros_like(v_inter)
        precision = torch.diag(1.0 / mass_matrix_diag)
        logdet = torch.sum(torch.log(mass_matrix_diag))

        v_inter_backward_log_prob = normal_log_prob(v_inter, zeros, precision, logdet)
        v_forward_log_prob = normal_log_prob(v, zeros, precision, logdet)

        return z_new, v_new, v_forward_log_prob, v_inter_backward_log_prob
