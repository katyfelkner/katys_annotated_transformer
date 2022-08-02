import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """TODO docstring"""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # project probabilities onto vocab (i.e. this is the output layer)
        self.proj == nn.Linear(d_model, vocab)

    def forward(self, x):
        # softmax to get vocab probabilities
        return F.log_softmax(self.proj(x), dim=-1)

