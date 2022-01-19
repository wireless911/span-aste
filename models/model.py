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
import torch
from torch import nn, Tensor
from torch.nn import LSTM, init
import itertools
from utils.tager import SpanLabel


class SpanRepresentation:
    """
    We define each span representation si,j ∈ S as:
            si,j =   [hi; hj ; f_width(i, j)] if BiLSTM
                     [xi; xj ; f_width(i, j)] if BERT
    where f_width(i, j) produces a trainable feature embedding representing the span width (i.e., j −i+ 1)
    Besides the concatenation of the start token, end token, and width representations,the span representation si,j
    can also be formed by max-pooling or mean-pooling across all token representations of the span from position i to j.
    The experimental results can be found in the ablation study.
    """

    def __init__(self, span_width_embedding_dim, max_window_size: int = 5):
        self.max_window_size = max_window_size
        self.span_width_embedding = nn.Embedding(512, span_width_embedding_dim)

    def __call__(self, x: Tensor):
        """
        [[2, 5], [0, 1], [1, 2], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12]]
        :param x: batch * len * dim
        :param term_cat:
        :return:
        """
        batch_size, sequence_length, _ = x.size()
        device = x.device

        len_arrange = torch.arange(0, sequence_length, device=device)
        span_indices = []

        for window in range(1, self.max_window_size + 1):
            if window == 1:
                indics = [(x.item(), x.item()) for x in len_arrange]
            else:
                res = len_arrange.unfold(0, window, 1)
                indics = [(idx[0].item(), idx[-1].item()) for idx in res]
            span_indices.extend(indics)

        spans = [torch.cat(
            (x[:, s[0], :], x[:, s[1], :],
             self.span_width_embedding(torch.LongTensor([abs(s[1] - s[0] + 1)])).repeat(
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


class TargetOpinionPairRepresentation:
    """
    Target Opinion Pair Representation We obtain the target-opinion pair representation by coupling each target candidate representation
    St_a,b ∈ St with each opinion candidate representation So_a,b ∈ So:
        G(St_a,b,So_c,d) = [St_a,b; So_c,d; f_distance(a, b, c, d)] (5)
    where f_distance(a, b, c, d) produces a trainable feature embedding based on the distance (i.e., min(|b − c|, |a − d|)) between the target
    and opinion span
    """

    def __init__(self, distance_embeddings_dim):
        self.distance_embeddings = nn.Embedding(512, distance_embeddings_dim)

    def min_distance(self, a, b, c, d):
        return torch.LongTensor([min(abs(b - c), abs(a - d))])

    def __call__(self, spans, span_indices, target_indices, opinion_indices):
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
                           self.distance_embeddings(
                               self.min_distance(*span_indices[c[0]], *span_indices[c[1]])).to(device).squeeze(0))
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
            input_dim: "int",
            target_dim: "int",
            relation_dim: "int",
            lstm_layer: "int" = 1,
            lstm_hidden_dim: "int" = 300,
            lstm_bidirectional: "bool" = True,
            ffnn_hidden_dim: "int" = 150,
            span_width_embedding_dim: "int" = 20,
            span_pruned_threshold: "int" = 0.5,
            pair_distance_embeddings_dim: "int" = 128,
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
        :param span_pruned_threshold: threshold hyper-parameter for span pruned.
        :type int (default:0.5)
        :param pair_distance_embeddings_dim: The number of features in the target-opinion pair distance embedding layer.
        :type int (default:128)
        """
        super(SpanAsteModel, self).__init__()
        self.span_pruned_threshold = span_pruned_threshold
        num_directions = 2 if lstm_bidirectional else 1
        self.lstm_encoding = LSTM(input_dim, num_layers=lstm_layer, hidden_size=lstm_hidden_dim, batch_first=True,
                                  bidirectional=lstm_bidirectional, dropout=0.5)
        self.span_representation = SpanRepresentation(span_width_embedding_dim)
        span_dim = lstm_hidden_dim * num_directions * 2 + span_width_embedding_dim
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

    def forward(self, x: torch.Tensor):
        """
        :param x: B * L * D
        :param adj: B * L * L
        :return:
        """
        # y_t (B,L,T)
        # h_t (B,L,num_directions*H)
        batch_size, sequence_len, _ = x.size()
        output, (hn, cn) = self.lstm_encoding(x)
        spans, span_indices = self.span_representation(output)
        spans_probability = self.span_ffnn(spans)
        nz = int(sequence_len * self.span_pruned_threshold)

        target_indices, opinion_indices = self.pruned_target_opinion(spans_probability, nz)

        # spans, span_indices, target_indices, opinion_indices
        candidates, candidate_indices, relation_indices = self.target_opinion_pair_representation(
            spans, span_indices, target_indices, opinion_indices)

        candidate_probability = self.pairs_ffnn(candidates)

        # batch span indices
        span_indices = [span_indices for _ in range(batch_size)]

        return spans_probability, span_indices, candidate_probability, candidate_indices
