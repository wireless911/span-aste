#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：insights-span-aste
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：hpuhzh@outlook.com
# @Date    ：03/08/2022 9:14 
# ====================================

import copy
import json
import os

from utils.tager import SentimentTriple, SentenceTagger


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, text_a, spans, span_labels, relations, relation_labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.spans = spans
        self.spans = spans
        self.relations = relations
        self.relations = relations
        self.span_labels = span_labels
        self.relation_labels = relation_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, input_len, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()


class Res15DataProcessor(DataProcessor):
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_txt(os.path.join(data_dir, "train_triplets.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_txt(os.path.join(data_dir, "dev_triplets.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_txt(os.path.join(data_dir, "test_triplets.txt")), "test")

    def fetch_offset(self, pre_text, curr_text):
        start_offset = len(self.tokenizer.encode(pre_text)[:-1])  # 去掉102
        curr_len = len(self.tokenizer.encode(curr_text)[1:-1])  # 去掉101、102
        end_offset = start_offset + curr_len
        return start_offset, end_offset

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = line["text"]
            tokens = text.split()
            labels = line["labels"]
            inputs = self.tokenizer.encode_plus(text, max_length=self.max_length, padding='max_length', truncation=True)
            sentiment_triples = []
            for label in labels:
                aspecet, opinion, sentiment = label
                if len(aspecet) == 1:
                    aspecet = aspecet * 2
                elif len(aspecet) > 2:
                    aspecet = [aspecet[0], aspecet[-1]]
                else:
                    pass
                if len(opinion) == 1:
                    opinion = opinion * 2
                elif len(opinion) > 2:
                    opinion = [opinion[0], opinion[-1]]
                else:
                    pass
                a1, a2 = aspecet
                o1, o2 = opinion
                # fetch offsets
                a_start_idx, a_end_idx = self.fetch_offset(" ".join(tokens[:a1]), " ".join(tokens[a1:a2 + 1]))
                o_start_idx, o_end_idx = self.fetch_offset(" ".join(tokens[:o1]), " ".join(tokens[o1:o2 + 1]))

                sentiment_triple = SentimentTriple.from_sentiment_triple(
                    ([a_start_idx, a_end_idx], [o_start_idx, o_end_idx], sentiment))
                sentiment_triples.append(sentiment_triple)

            sentence_tagger = SentenceTagger(sentiment_triples)
            spans, span_labels = sentence_tagger.spans
            relations, relation_labels = sentence_tagger.relations

            # BIOS
            examples.append(
                InputExample(guid=guid, text_a=text, spans=spans, relations=relations, span_labels=span_labels,
                             relation_labels=relation_labels))
        return examples

    def _read_txt(self, file_path):
        lines = []
        with open(file_path, "r", encoding="utf8") as f:
            data = f.readlines()
            for d in data:
                text, label = d.strip().split("####")
                row = {"text": text, "labels": eval(label)}
                lines.append(row)
        return lines





