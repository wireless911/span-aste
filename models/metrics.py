#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project ：span-aste
@IDE     ：PyCharm
@Author  ：Mr. Wireless
@Date    ：2022/1/19 11:18 
@Desc    ：
==================================================
"""
import torch


def metrics(probability, labels):
    """
    Collection metrics include (precision、recall、f1)
    :param probability:
    :param labels:
    :return: precision, recall, f1
    """
    epsilon = 1e-6
    num_correct = torch.logical_and(labels == probability.argmax(-1), probability.argmax(-1) != 0).sum().item()
    num_proposed = (probability.argmax(-1) != 0).sum().item()
    num_gold = (labels != 0).sum().item()

    precision = num_correct / (num_proposed + epsilon)
    recall = num_correct / (num_gold + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    return precision, recall, f1
