from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# batch_size = 10
# seq_per_img = 5
# pos_index = torch.zeros(batch_size * seq_per_img, dtype=torch.int64)
# neg_index = torch.zeros(batch_size * seq_per_img, dtype=torch.int64)
# for i in range(batch_size):
#     for j in range(seq_per_img):
#         index = seq_per_img * i + j
#         pos_rand = torch.randint(0, seq_per_img, (1,))
#         if pos_rand != j:
#             pos_index[index] = pos_rand + seq_per_img * i
#         neg_rand = torch.randint(0, batch_size * seq_per_img, (1,))
#         if neg_rand < seq_per_img * i or neg_rand >= seq_per_img * (i + 1):
#             neg_index[index] = neg_rand

def construct_triplet(embedding, seq):
    batch_size = seq.shape[0]
    seq_per_img = seq.shape[1]
    anchor = embedding
    pos_index = torch.zeros(batch_size*seq_per_img, dtype=torch.int64, device=anchor.device)
    neg_index = torch.zeros(batch_size*seq_per_img, dtype=torch.int64, device=anchor.device)
    for i in range(batch_size):
        for j in range(seq_per_img):
            index = seq_per_img * i + j
            pos_rand = torch.randint(0, seq_per_img, (1,))
            if pos_rand != j:
                pos_index[index] = pos_rand + seq_per_img * i
            neg_rand = torch.randint(0, batch_size*seq_per_img, (1,))
            if neg_rand < seq_per_img * i or neg_rand >= seq_per_img * (i+1):
                neg_index[index] = neg_rand

    positive = torch.index_select(embedding, 0, pos_index.to(anchor.device))
    negative = torch.index_select(embedding, 0, neg_index.to(anchor.device))
    return anchor, positive, negative
