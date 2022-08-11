#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project ：span-aste
@IDE     ：PyCharm
@Author  ：Mr. Wireless
@Date    ：2022/1/19 13:34 
@Desc    ：
==================================================
"""
import torch


def gold_labels(span_indices, spans, span_labels):
    """
    Organizing gold labels and indices
    :param span_indices:
    :param spans:
    :param span_labels:
    :return:
        gold_indices:
        gold_labels:
    """
    # gold span labels
    gold_indices, gold_labels = [], []
    for batch_idx, indices in enumerate(span_indices):
        gold_ind, gold_lab = [], []
        for indice in indices:
            if indice in spans[batch_idx]:
                ix = spans[batch_idx].index(indice)
                gold_lab.append(span_labels[batch_idx][ix])
            else:
                gold_lab.append(0)
            gold_ind.append(indice)
        gold_indices.append(gold_ind)
        gold_labels.append(gold_lab)

    return gold_indices, gold_labels


def collate_fn(data):
    """批处理，填充同一batch中句子最大的长度"""
    input_ids, attention_mask, token_type_ids, spans, relations, span_labels, relation_labels, seq_len = zip(*data)

    return input_ids, attention_mask, token_type_ids, spans, relations, span_labels, relation_labels, seq_len
