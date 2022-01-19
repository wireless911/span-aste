#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project ：aste
@IDE     ：PyCharm
@Author  ：Mr. Wireless
@Date    ：2021/12/21 14:04
@Desc    ：
==================================================
"""
from enum import IntEnum
from typing import Tuple, List, Text
from pydantic import BaseModel


class SpanLabel(IntEnum):
    INVALID = 0
    ASPECT = 1
    OPINION = 2


class RelationLabel(IntEnum):
    INVALID = 0
    POS = 1
    NEG = 2
    NEU = 3


class SentimentTriple(BaseModel):
    aspect: List
    opinion: List
    sentiment: Text

    @classmethod
    def from_sentiment_triple(cls, labels: Tuple[List, List, Text]):
        """read from sentiment triple"""
        assert len(labels) == 3
        # 处理single词
        new_labels = []
        for label in labels:
            if isinstance(label, str):
                new_labels.append(label)
            else:
                # 处理single词
                if len(label) == 1:
                    new_labels.append(label * 2)
                # 处理多词
                elif len(label) > 2:
                    new_labels.append([label[0], label[1]])
                else:
                    new_labels.append(label)
        return cls(
            aspect=new_labels[0],
            opinion=new_labels[1],
            sentiment=new_labels[2],
        )


class SentenceTagger:
    """例句标注"""

    def __init__(self, sentiments: List[SentimentTriple]):
        self.sentiments = sentiments
        self.sentiments_mapping = {"POS": RelationLabel.POS.value,
                                   "NEG": RelationLabel.NEG.value,
                                   "NEU": RelationLabel.NEU.value}

    @property
    def spans_labels(self):
        spans, span_labels = [], []
        for triplets in self.sentiments:
            spans.append(tuple(triplets.aspect))
            span_labels.append(SpanLabel.ASPECT.value)
            spans.append(tuple(triplets.opinion))
            span_labels.append(SpanLabel.OPINION.value)
        return spans, span_labels

    @property
    def relations(self):

        relations, relation_labels = [], []
        for triplets in self.sentiments:
            relation = []
            relation.extend(triplets.aspect)
            relation.extend(triplets.opinion)
            relation_labels.append(self.sentiments_mapping[triplets.sentiment])

            relations.append(tuple(relation))
        return relations, relation_labels
