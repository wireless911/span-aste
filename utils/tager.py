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

        relation = {"认可": "POS", "不认可": "NEG", "中性": "NEU"}
        assert len(labels) == 3
        return cls(
            aspect=labels[0],
            opinion=labels[1],
            sentiment=relation[labels[2]] if labels[2] in relation.keys() else labels[2]
        )


class SentenceTagger:
    """例句标注"""

    def __init__(self, sentiments: List[SentimentTriple]):
        self.sentiments = sentiments
        self.sentiments_mapping = {
            "POS": RelationLabel.POS.value,
            "NEG": RelationLabel.NEG.value,
            "NEU": RelationLabel.NEU.value
        }

    @property
    def spans(self):
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
