#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project ：span-aste
@IDE     ：PyCharm
@Author  ：Mr. Wireless
@Date    ：2022/1/18 16:19 
@Desc    ：
==================================================
"""
import argparse
import time
import torch
from utils.dataset import CustomDataset
from torch.utils.data import DataLoader
from models.collate import collate_fn
from models.tokenizers.tokenizer import BasicTokenizer
from models.embedding.word2vector import GloveWord2Vector
from models.model import SpanAsteModel
from utils.tager import SpanLabel
from utils.tager import RelationLabel
from trainer import SpanAsteTrainer


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device:{device}")
SEED = 1024
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


parser = argparse.ArgumentParser()
parser.add_argument("-w", "--glove_word2vector", required=True, type=str, default="vector_cache/42B_w2v.txt",
                    help="the glove word2vector file path")
parser.add_argument("-d", "--dataset", required=True, type=str, default="data/ASTE-Data-V2-EMNLP2020/15res/",
                    help="the dataset for train")
parser.add_argument("-o", "--output_path", required=True, type=str, default="output",
                    help="the model.pkl save path")
parser.add_argument("-b", "--batch_size", type=int, default=8, help="number of batch_size")
parser.add_argument("-e", "--epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--lstm_hidden", type=int, default=300, help="hidden size of BiLstm model")
parser.add_argument("--lstm_layers", type=int, default=1, help="number of BiLstm layers")
parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
parser.add_argument("--lr", type=float, default=1e-3, choices=[1e-3, 1e-4], help="learning rate of adam")
args = parser.parse_args()

print("Loading GloVe word2vector...", args.glove_word2vector)
tokenizer = BasicTokenizer()
glove_w2v = GloveWord2Vector(args.glove_word2vector)

print("Loading Train & Eval Dataset...", args.dataset)
# Load dataset
train_dataset = CustomDataset(
    args.dataset + "train_triplets.txt",
    tokenizer, glove_w2v
)
eval_dataset = CustomDataset(
    args.dataset + "dev_triplets.txt",
    tokenizer, glove_w2v
)
print("Construct Dataloader...")
batch_size = args.batch_size
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

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

print("Building Optimizer...", args.optimizer)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("Creating SPAN-ASTE Trainer...")
trainer = SpanAsteTrainer(model, optimizer, device)

epochs = args.epochs
best_eval_f1 = -1
for epoch in range(0, epochs):
    epoch_start_time = time.time()
    print('+' * 152)
    trainer.train(train_dataloader, epoch)
    eval_relation_f1 = trainer.eval(eval_dataloader, epoch)
    if eval_relation_f1 > best_eval_f1:
        trainer.save_model(epoch)
        best_eval_f1 = eval_relation_f1
    print('+' * 152)
    print('| end of epoch {:3d} | time: {:5.2f}s best_relation_1: {:8.3f}| '
          .format(epoch,
                  time.time() - epoch_start_time, best_eval_f1))
