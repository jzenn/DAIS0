from abc import abstractmethod

import torch

from src.distribution.normal import normal_log_prob
from src.sampling.ais import AnnealedImportanceSampling


def log_mean_exp(x):
    max_, _ = torch.max(x, 1, keepdim=True)
    return torch.log(torch.mean(torch.exp(x - max_), 1)) + torch.squeeze(max_)


class DifferentiableAnnealedImportanceSampling(AnnealedImportanceSampling):
    def __init__(self, args, log_joint, log_variational, deltas, betas):
        super(DifferentiableAnnealedImportanceSampling, self).__init__(
            args,
            log_joint=log_joint,
            log_variational=log_variational,
            deltas=deltas,
            betas=betas,
        )

    @abstractmethod
    def transition(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward_transition(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def backward_transition(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_initial_momentum(self, z):
        raise NotImplementedError

    @abstractmethod
    def initialize_elbo(self, z, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_last_elbo_increment(self, z, **kwargs):
        raise NotImplementedError

    def get_weight(
        self, z, z_new, k, z_new_forward_log_prob, z_backward_log_prob, **kwargs
    ):
        numerator = z_backward_log_prob
        denominator = z_new_forward_log_prob
        return numerator, denominator

    def get_metropolis_hastings_prob(
        self, z, z_new, k, z_new_forward_log_prob, z_backward_log_prob, **kwargs
    ):
        with torch.no_grad():
            # no momentum variable (Langevin)
            log_metropolis_hastings_correction = (
                z_backward_log_prob
                + self.log_gamma(self.betas[k], z_new, k=k, **kwargs)
            ) - (
                z_new_forward_log_prob + self.log_gamma(self.betas[k], z, k=k, **kwargs)
            )
            # momentum variable (Langevin + Momentum)
            v = kwargs["v"]
            v_new = kwargs["v_new"]
            if v is not None:
                momentum_precision, momentum_log_det = (
                    self.mass_matrix.precision(),
                    self.mass_matrix.logdet(),
                )
                log_metropolis_hastings_correction = (
                    log_metropolis_hastings_correction
                    + normal_log_prob(
                        v_new,
                        torch.zeros_like(v_new),
                        momentum_precision,
                        momentum_log_det,
                    )
                    - normal_log_prob(
                        v, torch.zeros_like(v), momentum_precision, momentum_log_det
                    )
                )
        return log_metropolis_hastings_correction

    def get_elbo_increment(self, log_numerator, log_denominator, **kwargs):
        log_normalization_increment = log_numerator - log_denominator
        return log_normalization_increment

    def forward(self, z, **kwargs):
        with torch.set_grad_enabled(self.training):
            v = self.get_initial_momentum(z)
            # initialize elbo
            elbo, initial_log_variational_post = self.initialize_elbo(z, v=v, **kwargs)
            # get elbo of initial samples
            with torch.no_grad():
                (
                    initial_target_log_prob,
                    initial_log_likelihood,
                    _,
                    initial_mean_samples,
                ) = self.target_dist.all_log_probs_and_reconstruction(z, **kwargs)
                elbo_of_initial_samples = (
                    initial_target_log_prob - self.initial_dist.log_prob(z, **kwargs)
                )
            # transitions
            for k in range(1, self.K + 1):
                # transition
                (
                    z_new,
                    v_new,
                    z_new_forward_log_prob,
                    z_backward_log_prob,
                ) = self.transition(z, k, v=v, **kwargs)
                # compute weight
                numerator, denominator = self.get_weight(
                    z,
                    z_new,
                    k,
                    z_new_forward_log_prob,
                    z_backward_log_prob,
                    v=v,
                    v_new=v_new,
                    **kwargs,
                )
                # update elbo
                log_normalization_increment = self.get_elbo_increment(
                    numerator, denominator
                )
                elbo = elbo + log_normalization_increment
                # update z's and momentum variables
                z = z_new
                v = v_new

            # last step
            (
                last_log_joint,
                last_log_likelihood,
                last_log_prior,
                reconstruction_mean,
            ) = self.target_dist.all_log_probs_and_reconstruction(z, **kwargs)
            # update elbo with terminal distribution
            last_log_normalization_increment = self.get_last_elbo_increment(
                z, last_log_joint=last_log_joint, v=v
            )
            elbo = elbo + last_log_normalization_increment
            # apply log-mean-exp (importance weighted average)
            apply_log_mean_exp = not self.args.do_not_apply_log_mean_exp_to_elbo
            # *.mean(-1) is to average over particles
            elbo = log_mean_exp(elbo) if apply_log_mean_exp else elbo.mean(-1)
            # sample from prior and reconstruct
            mean_samples = None
            if hasattr(self.target_dist, "prior"):
                with torch.no_grad():
                    z_samples = (
                        self.target_dist.prior.get_distribution()
                        .sample(z.shape)
                        .to(z.device)
                    )
                    (
                        _,
                        _,
                        _,
                        mean_samples,
                    ) = self.target_dist.all_log_probs_and_reconstruction(
                        z_samples, **kwargs
                    )

        return {
            "elbo": elbo,
            "elbo_of_initial_samples": log_mean_exp(elbo_of_initial_samples),
            #
            "last_z": z,
            "last_log_joint": last_log_joint.mean(-1),
            "last_log_joint_max": last_log_joint.max(-1)[0],
            "last_log_likelihood": last_log_likelihood.mean(-1),
            "last_log_likelihood_max": last_log_likelihood.max(-1)[0],
            "last_log_prior": last_log_prior.mean(-1),
            "initial_log_variational_post": initial_log_variational_post.mean(-1),
            "reconstruction_mean": reconstruction_mean,
            "initial_reconstruction_mean": initial_mean_samples,
            "initial_log_likelihood": initial_log_likelihood,
            "mean_samples": mean_samples,
        }


class IdentityDAIS(AnnealedImportanceSampling):
    def __init__(self, args, log_joint, log_variational, deltas, betas):
        super(IdentityDAIS, self).__init__(
            args,
            log_joint=log_joint,
            log_variational=log_variational,
            deltas=deltas,
            betas=betas,
        )
        self.importance_weighted = args.importance_weighted

    def forward(self, z, **kwargs):
        with torch.set_grad_enabled(self.training):
            log_variational_post = self.initial_dist.log_prob(z, **kwargs)
            (
                log_joint,
                log_likelihood,
                log_prior,
                reconstruction_mean,
            ) = self.target_dist.all_log_probs_and_reconstruction(z, **kwargs)
            # produce samples from initial distribution and reconstruct
            mean_samples = None
            if hasattr(self.target_dist, "prior"):
                with torch.no_grad():
                    z_samples = (
                        self.target_dist.prior.get_distribution()
                        .sample(z.shape)
                        .to(z.device)
                    )
                    (
                        _,
                        _,
                        _,
                        mean_samples,
                    ) = self.target_dist.all_log_probs_and_reconstruction(
                        z_samples, **kwargs
                    )
            elbo = log_joint - log_variational_post
            elbo = log_mean_exp(elbo) if self.importance_weighted else elbo.mean(-1)
        return {
            "elbo": elbo,
            "elbo_of_initial_samples": elbo,
            "initial_log_likelihood": log_likelihood,
            #
            "last_z": z,
            "last_log_joint": log_joint.mean(-1),
            "last_log_joint_max": log_joint.max(-1)[0],
            "last_log_likelihood": log_likelihood.mean(-1),
            "last_log_likelihood_max": log_likelihood.max(-1)[0],
            "last_log_prior": log_prior.mean(-1),
            "initial_log_variational_post": log_variational_post.mean(-1),
            "reconstruction_mean": reconstruction_mean,
            "initial_reconstruction_mean": reconstruction_mean,
            "mean_samples": mean_samples,
        }
