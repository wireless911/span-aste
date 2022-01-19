#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project ：span-aste
@IDE     ：PyCharm
@Author  ：Mr. Wireless
@Date    ：2022/1/19 10:35 
@Desc    ：
==================================================
"""
from typing import Text
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
from models.collate import gold_labels
from models.losses import log_likelihood
from models.metrics import metrics


class SpanAsteTrainer:
    """SpanAste model trainer"""

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: Text):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        current = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
        self.writer = SummaryWriter(f'logs/{current}')
        self.writer.flush()

    def train(self, train_dataloader, epoch):
        """train task"""
        self.model.train()
        for batch, data in enumerate(train_dataloader):
            x, spans, span_labels, relations, relation_labels, sequence_length = data

            spans_probability, span_indices, relations_probability, candidate_indices = self.model(x.to(self.device))

            batch_size, max_span_num, _ = spans_probability.size()

            gold_span_indices, gold_span_labels = gold_labels(span_indices, spans, span_labels)
            loss_ner = log_likelihood(spans_probability, span_indices, gold_span_indices, gold_span_labels)
            precision_ner, recall_ner, f1_ner = metrics(spans_probability,
                                                        torch.tensor(gold_span_labels, device=self.device))

            gold_relation_indices, gold_relation_labels = gold_labels(candidate_indices, relations, relation_labels)
            loss_relation = log_likelihood(relations_probability, candidate_indices, gold_relation_indices,
                                           gold_relation_labels)
            precision_relation, recall_relation, f1_relation = metrics(relations_probability,
                                                                       torch.tensor(gold_relation_labels,
                                                                                    device=self.device))

            loss = 0.2 * loss_ner + loss_relation

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print('| train'
                  '| epoch {:3d} | {:5d}/{:5d} batches '
                  '| ner P {:8.3f} R {:8.3f} F1 {:8.3f} loss {:8.3f}'
                  '| relation P {:8.3f} R {:8.3f} F1 {:8.3f} loss {:8.3f}'
                  .format(epoch, batch, len(train_dataloader),
                          precision_ner, recall_ner, f1_ner, loss_ner.item(),
                          precision_relation, recall_relation, f1_relation, loss_relation.item()
                          ))

            # ...log the running ner metrics
            self.writer.add_scalar('training precision_ner', precision_ner, epoch * len(train_dataloader) + batch)
            self.writer.add_scalar('training recall_ner', recall_ner, epoch * len(train_dataloader) + batch)
            self.writer.add_scalar('training f1_ner', f1_ner, epoch * len(train_dataloader) + batch)
            self.writer.add_scalar('training loss_ner', loss_ner.item(), epoch * len(train_dataloader) + batch)

            # ...log the running relation metrics
            self.writer.add_scalar('training precision_relation', precision_relation,
                                   epoch * len(train_dataloader) + batch)
            self.writer.add_scalar('training recall_relation', recall_relation,
                                   epoch * len(train_dataloader) + batch)
            self.writer.add_scalar('training f1_relation', f1_relation, epoch * len(train_dataloader) + batch)
            self.writer.add_scalar('training loss_relation', loss_relation.item(),
                                   epoch * len(train_dataloader) + batch)

    def eval(self, eval_dataloader, epoch):
        """eval task"""
        self.model.eval()
        with torch.no_grad():
            total_precision_ner = 0
            total_recall_ner = 0
            total_f1_ner = 0
            total_loss_ner = 0
            total_precision_relation = 0
            total_recall_relation = 0
            total_f1_relation = 0
            total_loss_relation = 0
            count = 0
            for batch, data in enumerate(eval_dataloader):
                x, spans, span_labels, relations, relation_labels, sequence_length = data
                spans_probability, span_indices, relations_probability, candidate_indices = self.model(
                    x.to(self.device))

                batch_size, max_span_num, _ = spans_probability.size()

                gold_span_indices, gold_span_labels = gold_labels(span_indices, spans, span_labels)
                loss_ner = log_likelihood(spans_probability, span_indices, gold_span_indices, gold_span_labels)
                precision_ner, recall_ner, f1_ner = metrics(spans_probability,
                                                            torch.tensor(gold_span_labels, device=self.device))

                gold_relation_indices, gold_relation_labels = gold_labels(candidate_indices, relations, relation_labels)
                loss_relation = log_likelihood(relations_probability, candidate_indices, gold_relation_indices,
                                               gold_relation_labels)
                precision_relation, recall_relation, f1_relation = metrics(relations_probability,
                                                                           torch.tensor(gold_relation_labels,
                                                                                        device=self.device))

                total_precision_ner += precision_ner
                total_recall_ner += recall_ner
                total_f1_ner += f1_ner
                total_loss_ner += loss_ner.item()
                total_precision_relation += precision_relation
                total_recall_relation += recall_relation
                total_f1_relation += f1_relation
                total_loss_relation += loss_relation.item()
                count += 1
            print('-' * 152)
            print('| eval'
                  '| epoch {:3d}'
                  '| ner P {:8.3f} R {:8.3f} F1 {:8.3f} loss {:8.3f}'
                  '| relation P {:8.3f} R {:8.3f} F1 {:8.3f} loss {:8.3f}'
                  .format(epoch,
                          total_precision_ner / count, total_recall_ner / count, total_f1_ner / count,
                          total_loss_ner / count,
                          total_precision_relation / count, total_recall_relation / count, total_f1_relation / count,
                          total_loss_relation / count
                          ))

            # ...log the running loss
            self.writer.add_scalar('eval f1_ner', total_f1_ner / count, epoch)
            self.writer.add_scalar('eval loss_ner', total_loss_ner / count, epoch)

            self.writer.add_scalar('eval f1_relation', total_f1_relation / count, epoch)
            self.writer.add_scalar('eval loss_relation', total_loss_relation / count, epoch)

            return total_f1_relation / count

    def test(self, test_dataloader):
        """test task"""
        self.model.eval()
        with torch.no_grad():
            total_precision_ner = 0
            total_recall_ner = 0
            total_f1_ner = 0
            total_loss_ner = 0
            total_precision_relation = 0
            total_recall_relation = 0
            total_f1_relation = 0
            total_loss_relation = 0
            count = 0
            for batch, data in enumerate(test_dataloader):
                x, spans, span_labels, relations, relation_labels, sequence_length = data
                spans_probability, span_indices, relations_probability, candidate_indices = self.model(
                    x.to(self.device))

                batch_size, max_span_num, _ = spans_probability.size()

                gold_span_indices, gold_span_labels = gold_labels(span_indices, spans, span_labels)
                loss_ner = log_likelihood(spans_probability, span_indices, gold_span_indices, gold_span_labels)
                precision_ner, recall_ner, f1_ner = metrics(spans_probability,
                                                            torch.tensor(gold_span_labels, device=self.device))

                gold_relation_indices, gold_relation_labels = gold_labels(candidate_indices, relations, relation_labels)
                loss_relation = log_likelihood(relations_probability, candidate_indices, gold_relation_indices,
                                               gold_relation_labels)
                precision_relation, recall_relation, f1_relation = metrics(relations_probability,
                                                                           torch.tensor(gold_relation_labels,
                                                                                        device=self.device))

                total_precision_ner += precision_ner
                total_recall_ner += recall_ner
                total_f1_ner += f1_ner
                total_loss_ner += loss_ner.item()
                total_precision_relation += precision_relation
                total_recall_relation += recall_relation
                total_f1_relation += f1_relation
                total_loss_relation += loss_relation.item()
                count += 1
            print('-' * 152)
            print('| test'
                  '| ner P {:8.3f} R {:8.3f} F1 {:8.3f} loss {:8.3f}'
                  '| relation P {:8.3f} R {:8.3f} F1 {:8.3f} loss {:8.3f}'
                  .format(total_precision_ner / count, total_recall_ner / count, total_f1_ner / count,
                          total_loss_ner / count,
                          total_precision_relation / count, total_recall_relation / count, total_f1_relation / count,
                          total_loss_relation / count
                          ))

            return total_f1_relation / count

    def save_model(self, epoch):
        # save the best model according to eval_relation_f1
        checkpoint = {"model_state_dict": self.model.state_dict(),
                      "optimizer_state_dict": self.optimizer.state_dict(),
                      "epoch": epoch}
        path_checkpoint = f"output/checkpoint_epoch_{epoch}.pkl"
        torch.save(checkpoint, path_checkpoint)
