#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project ：aste
@IDE     ：PyCharm
@Author  ：Mr. Wireless
@Date    ：2021/12/22 10:09 
@Desc    ：
==================================================
"""
import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.nn import LSTM, init
import itertools

from transformers import BertModel

from utils.tager import SpanLabel


class SpanRepresentation(nn.Module):
    """
    We define each span representation si,j ∈ S as:
            si,j =   [hi; hj ; f_width(i, j)] if BiLSTM
                     [xi; xj ; f_width(i, j)] if BERT
    where f_width(i, j) produces a trainable feature embedding representing the span width (i.e., j −i+ 1)
    Besides the concatenation of the start token, end token, and width representations,the span representation si,j
    can also be formed by max-pooling or mean-pooling across all token representations of the span from position i to j.
    The experimental results can be found in the ablation study.
    """

    def __init__(self, span_width_embedding_dim, span_maximum_length):
        super(SpanRepresentation, self).__init__()
        self.span_maximum_length = span_maximum_length
        self.bucket_bins = [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64]
        self.span_width_embedding = nn.Embedding(len(self.bucket_bins), span_width_embedding_dim)

    def bucket_embedding(self, width, device):
        em = [ix for ix, v in enumerate(self.bucket_bins) if width >= v][-1]
        return self.span_width_embedding(torch.LongTensor([em]).to(device))

    def forward(self, x: Tensor, batch_max_seq_len):
        """
        [[2, 5], [0, 1], [1, 2], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12]]
        :param x: batch * len * dim
        :param term_cat:
        :return:
        """
        batch_size, sequence_length, _ = x.size()
        device = x.device

        len_arrange = torch.arange(0, batch_max_seq_len, device=device)
        span_indices = []

        max_window = min(batch_max_seq_len, self.span_maximum_length)

        for window in range(1, max_window + 1):
            if window == 1:
                indics = [(x.item(), x.item()) for x in len_arrange]
            else:
                res = len_arrange.unfold(0, window, 1)
                indics = [(idx[0].item(), idx[-1].item()) for idx in res]
            span_indices.extend(indics)

        spans = [torch.cat(
            (x[:, s[0], :], x[:, s[1], :],
             self.bucket_embedding(abs(s[1] - s[0] + 1), device).repeat(
                 (batch_size, 1)).to(device)),
            dim=1) for s in span_indices]

        return torch.stack(spans, dim=1), span_indices


class PrunedTargetOpinion:
    """
    For a sentence X
    of length n, the number of enumerated spans is O(n^2), while the number of possible pairs between
    all opinion and target candidate spans is O(n^4) at the later stage (i.e., the triplet module). As such,
    it is not computationally practical to consider all possible pairwise interactions when using a span-based
    approach. Previous works (Luan et al., 2019; Wadden  et al., 2019) employ a pruning strategy to
    reduce the number of spans, but they only prune the spans to a single pool which is a mix of different
    mention types. This strategy does not fully consider
    """

    def __init__(self):
        pass

    def __call__(self, spans_probability, nz):
        target_indices = torch.topk(spans_probability[:, :, SpanLabel.ASPECT.value], nz, dim=-1).indices
        opinion_indices = torch.topk(spans_probability[:, :, SpanLabel.OPINION.value], nz, dim=-1).indices
        return target_indices, opinion_indices


class TargetOpinionPairRepresentation(nn.Module):
    """
    Target Opinion Pair Representation We obtain the target-opinion pair representation by coupling each target candidate representation
    St_a,b ∈ St with each opinion candidate representation So_a,b ∈ So:
        G(St_a,b,So_c,d) = [St_a,b; So_c,d; f_distance(a, b, c, d)] (5)
    where f_distance(a, b, c, d) produces a trainable feature embedding based on the distance (i.e., min(|b − c|, |a − d|)) between the target
    and opinion span
    """

    def __init__(self, distance_embeddings_dim):
        super(TargetOpinionPairRepresentation, self).__init__()
        self.bucket_bins = [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64]
        self.distance_embeddings = nn.Embedding(len(self.bucket_bins), distance_embeddings_dim)

    def min_distance(self, a, b, c, d):
        return min(abs(b - c), abs(a - d))

    def bucket_embedding(self, width, device):
        em = [ix for ix, v in enumerate(self.bucket_bins) if width >= v][-1]
        return self.distance_embeddings(torch.LongTensor([em]).to(device))

    def forward(self, spans, span_indices, target_indices, opinion_indices):
        """

        :param spans:
        :param span_indices:
        :param target_indices:
        :type
        :param opinion_indices:
        :return:
            candidate_indices :
                List[List[Tuple(a,b,c,d)]]
            relation_indices :
                List[List[Tuple(span1,span2)]]
        """
        batch_size = spans.size(0)
        device = spans.device

        # candidate_indices :[(a,b,c,d)]
        # relation_indices :[(span1,span2)]
        candidate_indices, relation_indices = [], []
        for batch in range(batch_size):
            pairs = list(itertools.product(target_indices[batch].cpu().tolist(), opinion_indices[batch].cpu().tolist()))
            relation_indices.append(pairs)
            candidate_ind = []
            for pair in pairs:
                a, b = span_indices[pair[0]]
                c, d = span_indices[pair[1]]
                candidate_ind.append((a, b, c, d))
            candidate_indices.append(candidate_ind)

        candidate_pool = []
        for batch in range(batch_size):
            relations = [
                torch.cat((spans[batch, c[0], :], spans[batch, c[1], :],
                           self.bucket_embedding(
                               self.min_distance(*span_indices[c[0]], *span_indices[c[1]]), device).squeeze(0))
                          , dim=0) for c in
                relation_indices[batch]]
            candidate_pool.append(torch.stack(relations))

        return torch.stack(candidate_pool), candidate_indices, relation_indices


class SpanAsteModel(nn.Module):
    """
    This repository is a pytorch version that implements Ali's ACL 2021 research paper
    `Learning Span-Level Interactions for Aspect Sentiment Triplet Extraction`
    paper:https://aclanthology.org/2021.acl-long.367/
    """

    def __init__(
            self,
            pretrain_model,
            target_dim: "int",
            relation_dim: "int",
            ffnn_hidden_dim: "int" = 150,
            span_width_embedding_dim: "int" = 20,
            span_maximum_length: "int" = 8,
            span_pruned_threshold: "int" = 0.5,
            pair_distance_embeddings_dim: "int" = 128,
            device="cpu"
    ) -> None:
        """
        :param input_dim: The number of expected features in the input `x`.
        :type int
        :param target_dim: The number of expected features for target .
        :type int
        :param relation_dim: The number of expected features for pairs .
        :type int
        :param lstm_layer: Number of lstm layers.
        :type int (default:1)
        :param lstm_hidden_dim: The number of features in the lstm hidden state `h`.
        :type int (default:1)
        :param lstm_bidirectional:
        :type boolean (default:300)
        :param ffnn_hidden_dim: The number of features in the feedforward hidden state `h`.
        :type int (default:150)
        :param span_width_embedding_dim: The number of features in the span width embedding layer.
        :type int (default:20)
        :param span_maximum_length: The maximum span length.
        :type int (default:8)
        :param span_pruned_threshold: threshold hyper-parameter for span pruned.
        :type int (default:0.5)
        :param pair_distance_embeddings_dim: The number of features in the target-opinion pair distance embedding layer.
        :type int (default:128)
        """
        super(SpanAsteModel, self).__init__()
        self.span_pruned_threshold = span_pruned_threshold
        self.pretrain_model = pretrain_model
        self.device = device

        self.bert = BertModel.from_pretrained(pretrain_model)
        encoding_dim = self.bert.config.hidden_size

        self.span_representation = SpanRepresentation(span_width_embedding_dim, span_maximum_length)
        span_dim = encoding_dim * 2 + span_width_embedding_dim
        self.span_ffnn = torch.nn.Sequential(
            nn.Linear(span_dim, ffnn_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(ffnn_hidden_dim, ffnn_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(ffnn_hidden_dim, target_dim, bias=True),
            nn.Softmax(-1)
        )
        self.pruned_target_opinion = PrunedTargetOpinion()
        self.target_opinion_pair_representation = TargetOpinionPairRepresentation(pair_distance_embeddings_dim)
        pairs_dim = 2 * span_dim + pair_distance_embeddings_dim
        self.pairs_ffnn = torch.nn.Sequential(
            nn.Linear(pairs_dim, ffnn_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(ffnn_hidden_dim, ffnn_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(ffnn_hidden_dim, relation_dim, bias=True),
            nn.Softmax(-1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.span_ffnn.named_parameters():
            if "weight" in name:
                init.xavier_normal_(param)
        for name, param in self.pairs_ffnn.named_parameters():
            if "weight" in name:
                init.xavier_normal_(param)

    def forward(self, input_ids, attention_mask, token_type_ids, seq_len):
        """
        :param x: B * L * D
        :param adj: B * L * L
        :return:
        """
        # y_t (B,L,T)
        # h_t (B,L,num_directions*H)

        batch_size, sequence_len = input_ids.size()
        batch_max_seq_len = max(seq_len)

        bert_output = self.bert(input_ids, attention_mask, token_type_ids)
        x = bert_output.last_hidden_state
        spans, span_indices = self.span_representation(x, batch_max_seq_len)
        spans_probability = self.span_ffnn(spans)
        nz = int(batch_max_seq_len * self.span_pruned_threshold)

        target_indices, opinion_indices = self.pruned_target_opinion(spans_probability, nz)

        # spans, span_indices, target_indices, opinion_indices
        candidates, candidate_indices, relation_indices = self.target_opinion_pair_representation(
            spans, span_indices, target_indices, opinion_indices)

        candidate_probability = self.pairs_ffnn(candidates)

        # batch span indices
        span_indices = [span_indices for _ in range(batch_size)]

        return spans_probability, span_indices, candidate_probability, candidate_indices
