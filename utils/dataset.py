#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project ：aste
@IDE     ：PyCharm
@Author  ：Mr. Wireless
@Date    ：2021/12/20 19:28 
@Desc    ：
==================================================
"""
from typing import Text

import torch
from torch.utils.data import Dataset
from utils.tager import SentimentTriple, SentenceTagger
from models.embedding.word2vector import GloveWord2Vector
from models.tokenizers.tokenizer import BasicTokenizer


class CustomDataset(Dataset):
    """
    An customer class representing txt data reading
    """

    def __init__(self,
                 file: "Text",
                 tokenizer: "BasicTokenizer",
                 word2vector: "GloveWord2Vector"
                 ) -> "None":
        self.tokenizer = tokenizer
        self.word2vector = word2vector

        with open(file, "r", encoding="utf8") as f:
            data = f.readlines()
        self.sentence_list = []
        for d in data:
            text, label = d.strip().split("####")
            row = {"text": text, "labels": eval(label)}
            self.sentence_list.append(row)

    def __getitem__(self, idx: "int"):
        text, labels = self.sentence_list[idx]["text"], self.sentence_list[idx]["labels"]
        tokens = self.tokenizer.tokenize(text)

        x = self.word2vector(tokens)
        sequence_length = len(tokens)

        sentiments_triples = [SentimentTriple.from_sentiment_triple(label) for label in labels]
        sentence_tager = SentenceTagger(sentiments_triples)

        span_indices, span_labels = sentence_tager.spans_labels
        relations, relation_labels = sentence_tager.relations

        return x, \
               span_indices, \
               span_labels, \
               relations, \
               relation_labels, \
               torch.tensor(sequence_length)

    def __len__(self):
        return len(self.sentence_list)
