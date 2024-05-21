import torch
from torch import nn
from torch.nn import functional as F


class LearnableBetas(nn.Module):
    def __init__(self, steps):
        super(LearnableBetas, self).__init__()
        self.K = steps
        self.beta_logits = nn.Parameter(torch.ones(steps))

    def forward(self):
        beta_deltas = F.softmax(self.beta_logits, dim=0)
        betas = torch.cumsum(beta_deltas, dim=0)
        betas = torch.cat((torch.zeros(1).to(self.beta_logits.device), betas))
        return betas

    def get_all_betas(self):
        return self.forward()

    def __getitem__(self, k):
        return self.forward()[k]
