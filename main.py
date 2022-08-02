import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
import encoder_decoder, transformer_utils, attention, feedforward, inputs, generator


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """helper function: construct transformer model from given hyperparameters"""
    attn = attention.MultiHeadAttention(h, d_model)
    ff = feedforward.PositionwiseFeedForward(d_model, d_ff, dropout)
    position = inputs.PositionalEncoding(d_model, dropout)
    model = encoder_decoder.EncoderDecoder(
        encoder_decoder.EncoderStack(encoder_decoder.EncoderLayer(
            d_model, copy.deepcopy(attn), copy.deepcopy(ff), dropout), N),
        encoder_decoder.DecoderStack(encoder_decoder.DecoderLayer(
            d_model, copy.deepcopy(attn), copy.deepcopy(attn), copy.deepcopy(ff), dropout), N),
        # source embeddings
        nn.Sequential(inputs.Embeddings(d_model, src_vocab), copy.deepcopy(position)),
        nn.Sequential(inputs.Embeddings(d_model, tgt_vocab), copy.deepcopy(position)),
        generator.Generator(d_model, tgt_vocab)
    )

    # parameter initialization
    # https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

class Batch:
    """hold a batch of data and its mask during training."""
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2) # TODO WHAT THE HECK
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """mask to hide padding and future words"""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            transformer_utils.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def run_epoch(data_iter, model, loss_compute):
    """training loop (including logging)"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss = loss_compute(out, batch.tgt_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.ntokens, tokens/elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    """Keep augmenting batch and calculate total number of tokens and padding"""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt) + 2)  # TODO why plus 2
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class NoamOpt:
    """optim wrapper to implement learning rate
    LR is linear during warmup, then follows Adam after that"""
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0  # current step
        self.warmup = warmup  # number warmup steps
        self.factor = factor  # TODO what is this exactly
        self.model_size = model_size  # model dimension
        self._rate = 0  # current learning rate

    def step(self):
        """Update parameters and rate"""
        self._step += 1  # increment step count
        self.optimizer.step()  # newer torch: step before lr update
        rate = self.rate()
        for p in self.optimizer.param_groups:  # iterate over all parameters
            # TODO why set the lr for each param group rather than globally?
            p['lr'] = rate
        self._rate = rate

    def rate(self, step=None):
        """Implement learning rate"""
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    """ utility function to get standard Noam optimizer"""
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

