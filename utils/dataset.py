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
from torch.utils.data import Dataset
from utils.processor import InputExample, DataProcessor
from transformers import BertTokenizer


class CustomDataset(Dataset):
    """
    An customer class representing txt data reading
    """

    def __init__(self,
                 data_type: "Text",
                 data_dir: "Text",
                 processor: "DataProcessor",
                 tokenizer: "BertTokenizer",
                 max_seq_length: "int"
                 ) -> "None":
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.sentence_list = []
        if data_type == 'train':
            examples = processor.get_train_examples(data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(data_dir)
        else:
            examples = processor.get_test_examples(data_dir)
        self.examples = examples

    def __getitem__(self, idx: "int"):
        example = self.examples[idx]  # type:InputExample
        inputs = self.tokenizer(example.text_a, max_length=self.max_seq_length, padding='max_length', truncation=True)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        token_type_ids = inputs.token_type_ids
        spans = example.spans
        relations = example.relations
        span_labels = example.span_labels
        relation_labels = example.relation_labels
        seq_len = len([i for i in input_ids if i != 0])

        return input_ids, attention_mask, token_type_ids, spans, relations, span_labels, relation_labels, seq_len

    def __len__(self):
        return len(self.examples)
