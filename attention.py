import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import transformer_utils


def attention(query, key, value, mask=None, dropout=None):
    """Compute scaled dot-product attention"""
    # d_k = d_v = model_size / number of heads
    d_k = query.size(-1)
    # corresponds to e_ij value
    # transpose(-2, -1) swaps last 2 dimensions, i.e. swap width and height but keep depth
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # scaling factor
    if mask is not None:
        # small epsilon for numerical stability
        scores = scores.masked_fill(mask == 0, -1e9)
    # corresponds to alpha values
    p_attn = F.softmax(scores, dim=-1)
    # apply dropout if required
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """multi headed attention layer """
    def __init__(self, h, d_model, dropout=0.1):
        """:param h: number of attetion heads
        :param d_model: model dimension
        :param dropout: attention dropout probability"""
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0  # model dimension must be divisible by number of heads
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = transformer_utils.clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """implement multiheaded attention forward pass"""
        # mask if needed
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # do all linear projects from d_model -> h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linear_layers, (query, key, value))]

        # apply attention on all the projected vectors
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # concatenate and apply final linear layer (W^O)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linear_layers[-1](x)