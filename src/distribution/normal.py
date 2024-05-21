# taken and adapted from the submission of Zhang et al.
# at https://openreview.net/forum?id=6rqjgrL7Lq

import math

import torch


class Normal:
    def __init__(self, mean, covariance):
        assert (
            mean.ndim == 3
            and covariance.D() == 2
            and mean.shape[1] == 1
            and mean.shape[2] == 1
        )
        self.mean = mean
        self.covariance = covariance
        self.precision = torch.inverse(covariance)
        self.cholesky_lower_covariance = torch.linalg.cholesky(covariance)
        self.d = mean.shape[0]
        self.log_det_covariance = torch.logdet(covariance)

    def log_prob(self, x):
        assert x.shape[-1] == self.d
        diff = x - self.mean.T
        M = torch.sum(diff @ self.precision * diff, -1)
        return -0.5 * (self.d * math.log(2 * math.pi) + M + self.log_det_covariance)

    def sample(self, shape: tuple):
        Eps = torch.empty(
            (self.d, *shape), dtype=self.mean.dtype, device=self.mean.device
        ).normal_()
        return Eps.permute(
            1, 2, 0
        ) @ self.cholesky_lower_covariance.T + self.mean.permute(
            *torch.arange(self.mean.ndim - 1, -1, -1)
        )


def normal_log_prob(x, mean, precision, log_det_covariance):
    d = x.shape[-1]
    diff = x - mean
    M = torch.sum(diff @ precision * diff, -1)
    return -0.5 * (d * math.log(2 * math.pi) + M + log_det_covariance)


def sample_normal(mean, cholesky_lower_covariance, shape):
    d = mean.shape[-1]
    eps = torch.empty((*shape, d), dtype=mean.dtype, device=mean.device).normal_()
    return eps @ cholesky_lower_covariance.T + mean
