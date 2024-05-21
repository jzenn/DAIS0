import torch
from torch import nn


class LearnableDeltas(nn.Module):
    def __init__(self, args, log_parameter_offset=-5.0):
        super(LearnableDeltas, self).__init__()
        self.K = args.n_transitions
        self.log_deltas = nn.Parameter(torch.zeros(self.K + 1) + log_parameter_offset)

    def forward(self, k):
        return self.log_deltas[k].exp()

    def get_all_deltas(self):
        return torch.tensor([self[i] for i in range(self.K + 1)])

    def __getitem__(self, k):
        return self.forward(k)
