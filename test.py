#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project ：span-aste
@IDE     ：PyCharm
@Author  ：Mr. Wireless
@Date    ：2022/1/19 9:10 
@Desc    ：
==================================================
"""
import argparse

import torch
from torch.utils.data import DataLoader

from models.collate import collate_fn
from models.tokenizers.tokenizer import BasicTokenizer
from models.embedding.word2vector import GloveWord2Vector
from models.model import SpanAsteModel
from trainer import SpanAsteTrainer
from utils.dataset import CustomDataset
from utils.tager import SpanLabel, RelationLabel

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device:{device}")
batch_size = 16

SEED = 1024
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True, type=str, default="output/checkpoint.pkl",
                    help="the model of span-aste output")
parser.add_argument("-w", "--glove_word2vector", required=True, type=str, default="vector_cache/42B_w2v.txt",
                    help="the glove word2vector file path")
parser.add_argument("-d", "--dataset", required=True, type=str, default="data/ASTE-Data-V2-EMNLP2020/15res/",
                    help="the dataset for test")
parser.add_argument("-b", "--batch_size", type=int, default=8, help="number of batch_size")
parser.add_argument("--lstm_hidden", type=int, default=300, help="hidden size of BiLstm model")
parser.add_argument("--lstm_layers", type=int, default=1, help="number of BiLstm layers")

args = parser.parse_args()

print("Loading GloVe word2vector...", args.glove_word2vector)
tokenizer = BasicTokenizer()
glove_w2v = GloveWord2Vector(args.glove_word2vector)

print("Loading test Dataset...", args.dataset)
# Load dataset
test_dataset = CustomDataset(
    args.dataset + "test_triplets.txt",
    tokenizer, glove_w2v
)

print("Construct Dataloader...")
batch_size = args.batch_size
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

print("Building SPAN-ASTE model...")
# get dimension of target and relation
target_dim, relation_dim = len(SpanLabel), len(RelationLabel)
# get the dimension of glove vector
input_dim = glove_w2v.glove_model.vector_size
# build span-aste model
model = SpanAsteModel(
    input_dim,
    target_dim,
    relation_dim,
    lstm_layer=args.lstm_layers,
    lstm_hidden_dim=args.lstm_hidden
)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("Creating SPAN-ASTE Trainer...")
trainer = SpanAsteTrainer(model, optimizer, device)

print("Loading model state from output...", args.model)
model.load_state_dict(torch.load(args.model)["model_state_dict"])

trainer.test(test_dataloader)
