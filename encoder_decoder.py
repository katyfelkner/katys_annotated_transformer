import torch
import torch.nn as nn
import seaborn
import transformer_utils

# this sets graph default styling
seaborn.set_context(context="talk")


class EncoderDecoder(nn.Module):
    """
    standard encoder/decoder. This is a base model for the transformer.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        TODO: write this docstring once I understand the method
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        TODO: write this docstring once I understand the method
        """
        encoded = self.encode(src, src_mask)
        decoded = self.decode(encoded, src_mask, tgt, tgt_mask)
        return decoded

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class EncoderStack(nn.Module):
    """stack of N encoder layers"""
    def __init__(self, layer, N):
        super(EncoderStack, self).__init__()
        self.layers = transformer_utils.clones(layer, N)
        # TODO what exactly is layer norm
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """TODO what is the mask, write this doc"""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    """perform layer normalization - normalization by all input values to a given layer"""
    def __init__(self, features, eps=1e-6):
        """:param features: size of vectors to normalize"""
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))  # gain initialized to 1
        self.b_2 = nn.Parameter(torch.zeros(features))  # bias initialized to 0
        self.eps = eps  # epsilon for numerical stability - avoid divide by 0

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * ((x - mean) / (std + self.eps)) + self.b_2


class SublayerConnection(nn.Module):
    """TODO this docstring"""
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)  # norming operator for this layer
        self.dropout = nn.Dropout(dropout)  # dropout w/ fixed probability

    def forward(self, x, sublayer):
        """apply norm and dropout to sublayer output.
        actual function of the sublayer (attn or ff) managed by sublayer object
        then apply a residual connection around the sublayer."""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """an encoder layer is a multihead attention layer, followed by a ff layer,
    with sublayer connections around each. individual sublayers passed by caller."""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = transformer_utils.clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda a: self.self_attn(a, a, a, mask))
        return self.sublayers[1](x, self.feed_forward)


class DecoderStack(nn.Module):
    """N layer decoder stack"""
    def __init__(self, layer, N):
        super(DecoderStack, self).__init__()
        self.layers = transformer_utils.clones(layer, N)
        self.norm = LayerNorm(layer.size)  # TODO why does norm belong to the stack instead of each layer

    def forward(self, x, mask):
        """pass input and mask through each layer"""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """one decoder layer: self-attn, cross-attn, and feedforward.
    Residual connections and norms at each sublayer"""
    def __init__(self, size, self_attn, cross_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayers = transformer_utils.clones(SublayerConnection(size, dropout), 3)
        self.size = size

    def forward(self, x, memory, src_mask, tgt_mask):
        """TODO what is memory in this case
        TODO still confused on these lambda functions"""
        x = self.sublayers[0](x, lambda a: self.self_attn(a, a, a, tgt_mask))  # self attn
        x = self.sublayers[1](x, lambda a: self.self_attn(a, memory, memory, src_mask))
        return self.sublayers[2](x, self.feed_forward)
