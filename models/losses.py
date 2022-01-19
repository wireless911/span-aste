#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project ：span-aste
@IDE     ：PyCharm
@Author  ：Mr. Wireless
@Date    ：2022/1/19 11:19 
@Desc    ：
==================================================
"""
import torch


def log_likelihood(probability, indices, gold_indices, gold_labels):
    """
    The training objective is defined as the sum of the negative log-likelihood from both the mention module and triplet module.
    where m∗i,j is the gold mention type of the span si,j ,and r∗is the gold sentiment relation of the target and opinion span
    pair (St_a,b, So_c,d). S indicates the enumerated span pool; Stand So are the pruned target and opinion span candidates.
    :param probability: the probability from span or candidates
    :type Tensor
    :param indices: the indices for predicted span or candidates
    :type List[List[Tuple(i,j)]] or List[List[Tuple(a,b,c,d)]]
    :param span:
    :param labels:
    :type List[List[0/1)]]
    :return: negative log-likelihood
    """
    # Statistically predict the indices of the correct mention or candidates
    gold_indice_labels = []
    for batch_idx, label in enumerate(gold_indices):
        for i, l in enumerate(label):
            if l in indices[batch_idx]:
                idx = indices[batch_idx].index(l)
                gold_indice_labels.append((batch_idx, idx, gold_labels[batch_idx][i]))

    # sum of the negative log-likelihood from both the mention module and triplet module
    loss = [-torch.log(probability[c[0], c[1], c[2]]) for c in gold_indice_labels]
    loss = torch.stack(loss).sum()
    return loss
