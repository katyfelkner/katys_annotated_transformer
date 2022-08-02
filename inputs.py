import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn


class Embeddings(nn.Module):
    """embedding matrix, shared between enc. and dec. embedding layers and output layer"""
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # embedding matrix with vocab entries, each of size d_model
        self.emb = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)  # scale embedding weights
        # TODO why do we need to scale


class PositionalEncoding(nn.Module):
    """implement sinusoidal positional encoding"""
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # TODO why do we want to apply dropout to embeddings and PE
        self.dropout = nn.Dropout(p=dropout)

        # precompute positional encodings in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # positions 0, 2, 4,... model_dim
        pe[:, 0::2] = torch.sin(position * div_term)
        # positions 1, 3, ... model_dim - 1
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # buffers are persistent, not trainable

    def forward(self, x):
        x = x + Variable(self.pe[:, x.size(1)], requires_grad=False)
        return self.dropout(x)

    