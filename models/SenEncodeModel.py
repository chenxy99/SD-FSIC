from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.FCModel import LSTMCore

class SenEncodeModel(nn.Module):
    def __init__(self, opt):
        super(SenEncodeModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        # self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = getattr(opt, 'max_length', 20) or opt.seq_length  # maximum sample length

        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self._core = LSTMCore(opt)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def forward(self, seq, seq_masks):
        batch_size = seq.size(0)
        if seq.ndim == 3:  # B * seq_per_img * seq_len
            seq = seq.reshape(-1, seq.shape[2])
            seq_masks = seq_masks.reshape(-1, seq_masks.shape[2])
        seq_per_img = seq.shape[0] // batch_size
        state = self.init_hidden(batch_size * seq_per_img)
        embedding_list = []
        for i in range(seq.shape[0]):
            embedding_list.append([])
        stop_sign =(seq_masks.sum(1) - 1).int()
        for i in range(seq.size(1) - 1):
            it = seq[:, i].clone()
            xt = self.embed(it)
            output, state = self._core(xt, state)
            if i >= 1:
                non_zero_index = torch.nonzero(stop_sign == i)
                for index in non_zero_index:
                    # embedding_list[index] = output[index]
                    embedding_list[index] = state[0][0, index]
        non_zero_index = torch.nonzero(stop_sign == seq.size(1) - 1)
        for index in non_zero_index:
            embedding_list[index] = state[0][0, index]
        embeddings = torch.cat(embedding_list, dim=0).squeeze(1)
        return embeddings
