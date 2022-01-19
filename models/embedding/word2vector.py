#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project ：aste
@IDE     ：PyCharm
@Author  ：Mr. Wireless
@Date    ：2021/12/22 13:42 
@Desc    ：
==================================================
"""
from typing import Optional, Text, List
import torch
from gensim.models import KeyedVectors
from torch import Tensor


class GloveWord2Vector:
    """glove embedding"""

    def __init__(self, vocab_file: Text, device: Text = "cpu") -> None:
        self.glove_model = KeyedVectors.load_word2vec_format(vocab_file)
        self.device = device
        self.oov = ","

    def get_vector(self, token: Text, lower_case: bool = True):
        token = token.lower() if lower_case and isinstance(token, str) else token
        return self.glove_model[token] if token in self.glove_model else self.glove_model[self.oov]

    def __call__(self, tokens: List[Text], lower_case: bool = True) -> Tensor:
        vec = torch.tensor(
            [self.get_vector(token, lower_case=lower_case)
             for token in
             tokens], device=self.device)
        return vec


if __name__ == '__main__':
    glove = GloveWord2Vector("../../cropus/w2v.txt")
    glove(["hello"])
