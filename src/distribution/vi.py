import math

import torch
from torch import nn


class MeanFieldVariationalDistribution(nn.Module):
    def __init__(self, dim, diagonal_covariance=1.0, random_initialization=False):
        super(MeanFieldVariationalDistribution, self).__init__()
        self.dim = dim
        self.mean = nn.Parameter(
            torch.randn(dim) if random_initialization else torch.zeros(dim)
        )
        self.log_diagonal = nn.Parameter(
            torch.log(torch.ones(dim) * diagonal_covariance)
        )

    def scale_tril(self):
        return torch.diag(torch.exp(self.log_diagonal / 2.0))

    def precision(self):
        return torch.diag(torch.exp(-self.log_diagonal))

    def log_det_covariance(self):
        return torch.sum(self.log_diagonal)

    def log_prob(self, z, **kwargs):
        diff = z - self.mean.unsqueeze(0).unsqueeze(0)
        M = torch.sum(diff @ self.precision() * diff, -1)
        return -0.5 * (self.dim * math.log(2 * math.pi) + M + self.log_det_covariance())

    def sample(self, shape):
        Eps = torch.empty(
            (*shape, self.dim), dtype=self.mean.dtype, device=self.mean.device
        ).normal_()
        return Eps @ self.scale_tril() + self.mean.unsqueeze(0).unsqueeze(0)
