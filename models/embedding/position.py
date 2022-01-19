#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project ：aste
@IDE     ：PyCharm
@Author  ：Mr. Wireless
@Date    ：2021/12/24 10:19 
@Desc    ：
==================================================
"""
import itertools
from typing import List

import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class RelationPositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super(RelationPositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.randn((1, max_len, d_model))
        self.register_buffer('pe', pe)

    def forward(self, aspect_opinion_pairs: List, max_len):
        """
        B * L * D
        example:
            OS X is solid with lots of innovations such as quicklook which save heaps of time .
            [([0, 1], [3], 'POS'), ([0, 1], [7], 'POS'), ([10], [12, 13, 14, 15], 'POS')]
            => [6, 6, 0, 2, 0, 0, 0, 6, 0, 0, 2, 0, 2, 2, 2, 2, 0]

        :return: [6, 6, 0, 2, 0, 0, 0, 6, 0, 0, 2, 0, 2, 2, 2, 2, 0]
        """
        batch_size = len(aspect_opinion_pairs)
        position = self.pe.repeat(batch_size, 1, 1)

        for batch in range(batch_size):
            if aspect_opinion_pairs[batch]:
                for aspect, opinion in aspect_opinion_pairs[batch]:
                    middle = abs(aspect[1] - opinion[0])
                    position[batch][aspect[0]:aspect[1]] = middle
                    position[batch][opinion[0]:opinion[1]] = middle

        return position[:, :max_len, :]
