import time
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import CustomDataset
from torch.utils.data import DataLoader
import torch
from models.tokenizers.tokenizer import BasicTokenizer
from models.embedding.word2vector import GloveWord2Vector
from models.model import SpanAsteModel
from utils.tager import SpanLabel
import datetime
from utils.tager import RelationLabel

current = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
writer = SummaryWriter(f'logs/{current}')
writer.flush()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device:{device}")
batch_size = 4

SEED = 1024
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def collate_fn(data):
    """批处理，填充同一batch中句子最大的长度"""
    x, spans, span_labels, relations, relation_labels, sequence_length = zip(*data)
    max_len = max(sequence_length)
    x = torch.stack(
        [torch.cat([item, torch.zeros(max_len - item.size(0), item.size(1))]) for item in x])
    sequence_length = torch.stack([item for item in sequence_length])

    return x.float().to(device), \
           spans, \
           span_labels, \
           relations, \
           relation_labels, \
           sequence_length.long().to(device)


tokenizer = BasicTokenizer()
glove_w2v = GloveWord2Vector("corpus/42B_w2v.txt")

train_dataset = CustomDataset("data/ASTE-Data-V2-EMNLP2020/15res/train_triplets.txt", tokenizer, glove_w2v)
eval_dataset = CustomDataset("data/ASTE-Data-V2-EMNLP2020/15res/dev_triplets.txt", tokenizer, glove_w2v)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

target_dim, relation_dim = len(SpanLabel), len(RelationLabel)
input_dim = glove_w2v.glove_model.vector_size

model = SpanAsteModel(input_dim, target_dim, relation_dim)

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss(reduction="sum")


def gold_labels(span_indices, spans, span_labels):
    """
    Organizing gold labels and indices
    :param span_indices:
    :param spans:
    :param span_labels:
    :return:
        gold_indices:
        gold_labels:
    """
    # gold span labels
    gold_indices, gold_labels = [], []
    for batch_idx, indices in enumerate(span_indices):
        gold_ind, gold_lab = [], []
        for indice in indices:
            if indice in spans[batch_idx]:
                ix = spans[batch_idx].index(indice)
                gold_lab.append(span_labels[batch_idx][ix])
            else:
                gold_lab.append(0)
            gold_ind.append(indice)
        gold_indices.append(gold_ind)
        gold_labels.append(gold_lab)

    return gold_indices, gold_labels


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


def log_likelihood(probability, indices, span, labels):
    """
    The training objective is defined as the sum of the negative log-likelihood from both the mention module and triplet module.
    where m∗i,j is the gold mention type of the span si,j ,and r∗is the gold sentiment relation of the target and opinion span
    pair (St_a,b, So_c,d). S indicates the enumerated span pool; Stand So are the pruned target and opinion span candidates.
    :param probability: the probability from span or candidates
    :type Tensor
    :param indices: the indices for predicted span or candidates
    :type List[List[Tuple(i,j)]] or List[List[Tuple(a,b,c,d)]]
    :param span:
    :param labels:
    :type List[List[0/1)]]
    :return: negative log-likelihood
    """
    # Statistically predict the indices of the correct mention or candidates
    gold_indices = []
    for batch_idx, label in enumerate(span):
        for i, l in enumerate(label):
            if l in indices[batch_idx]:
                idx = indices[batch_idx].index(l)
                gold_indices.append((batch_idx, idx, labels[batch_idx][i]))

    # sum of the negative log-likelihood from both the mention module and triplet module
    loss = [-torch.log(probability[c[0], c[1], c[2]]) for c in gold_indices]
    loss = torch.stack(loss).sum()
    return loss


def train(dataloader, epoch):
    model.train()
    for batch, data in enumerate(dataloader):
        x, spans, span_labels, relations, relation_labels, sequence_length = data

        spans_probability, span_indices, relations_probability, candidate_indices = model(x)

        batch_size, max_span_num, _ = spans_probability.size()

        # # gold span labels
        # new_span_labels = []
        # new_spans = []
        # for batch_idx, candidate_indice in enumerate(span_indices):
        #     span_label = []
        #     span = []
        #     for can in candidate_indice:
        #         if can in spans[batch_idx]:
        #             ix = spans[batch_idx].index(can)
        #             span_label.append(span_labels[batch_idx][ix])
        #         else:
        #             span_label.append(0)
        #         span.append(can)
        #     new_spans.append(span)
        #     new_span_labels.append(span_label)

        gold_span_indices, gold_span_labels = gold_labels(span_indices, spans, span_labels)
        loss_ner = log_likelihood(spans_probability, span_indices, gold_span_indices, gold_span_labels)
        precision_ner, recall_ner, f1_ner = metrics(spans_probability, torch.tensor(gold_span_labels, device=device))

        # gold realtion labels
        # new_relation_labels = []
        # new_relations = []
        # for batch_idx, candidate_indice in enumerate(candidate_indices):
        #     relation_label = []
        #     relation = []
        #     for can in candidate_indice:
        #         if can in relations[batch_idx]:
        #             ix = relations[batch_idx].index(can)
        #             relation_label.append(relation_labels[batch_idx][ix])
        #         else:
        #             relation_label.append(0)
        #         relation.append(can)
        #     new_relations.append(relation)
        #     new_relation_labels.append(relation_label)

        gold_relation_indices, gold_relation_labels = gold_labels(candidate_indices, relations, relation_labels)
        loss_relation = log_likelihood(relations_probability, candidate_indices, gold_relation_indices,
                                       gold_relation_labels)
        precision_relation, recall_relation, f1_relation = metrics(relations_probability,
                                                                   torch.tensor(gold_relation_labels, device=device))

        loss = 0.2 * loss_ner + loss_relation

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('| train'
              '| epoch {:3d} | {:5d}/{:5d} batches '
              '| ner P {:8.3f} R {:8.3f} F1 {:8.3f} loss {:8.3f}'
              '| relation P {:8.3f} R {:8.3f} F1 {:8.3f} loss {:8.3f}'
              .format(epoch, batch, len(dataloader),
                      precision_ner, recall_ner, f1_ner, loss_ner.item(),
                      precision_relation, recall_relation, f1_relation, loss_relation.item()
                      ))

        # ...log the running ner metrics
        writer.add_scalar('training precision_ner', precision_ner, epoch * len(dataloader) + batch)
        writer.add_scalar('training recall_ner', recall_ner, epoch * len(dataloader) + batch)
        writer.add_scalar('training f1_ner', f1_ner, epoch * len(dataloader) + batch)
        writer.add_scalar('training loss_ner', loss_ner.item(), epoch * len(dataloader) + batch)

        # ...log the running relation metrics
        writer.add_scalar('training precision_relation', precision_relation, epoch * len(dataloader) + batch)
        writer.add_scalar('training recall_relation', recall_relation, epoch * len(dataloader) + batch)
        writer.add_scalar('training f1_relation', f1_relation, epoch * len(dataloader) + batch)
        writer.add_scalar('training loss_relation', loss_relation.item(), epoch * len(dataloader) + batch)

    return


def eval(dataloader, epoch):
    """eval model"""
    model.eval()
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
        for batch, data in enumerate(dataloader):
            x, spans, span_labels, relations, relation_labels, sequence_length = data
            spans_probability, span_indices, pred, candidate_indices = model(x)
            batch_size, max_span_num, _ = spans_probability.size()

            # gold span labels
            new_span_labels = []
            new_spans = []
            for batch_idx, candidate_indice in enumerate(span_indices):
                span_label = []
                span = []
                for can in candidate_indice:
                    if can in spans[batch_idx]:
                        ix = spans[batch_idx].index(can)
                        span_label.append(span_labels[batch_idx][ix])
                    else:
                        span_label.append(0)
                    span.append(can)
                new_spans.append(span)
                new_span_labels.append(span_label)

            loss_ner = log_likelihood(spans_probability, span_indices, new_spans, new_span_labels)
            precision_ner, recall_ner, f1_ner = metrics(spans_probability, torch.tensor(new_span_labels, device=device))

            # gold realtion labels
            new_relation_labels = []
            new_relations = []
            for batch_idx, candidate_indice in enumerate(candidate_indices):
                relation_label = []
                relation = []
                for can in candidate_indice:
                    if can in relations[batch_idx]:
                        ix = relations[batch_idx].index(can)
                        relation_label.append(relation_labels[batch_idx][ix])
                    else:
                        relation_label.append(0)
                    relation.append(can)
                new_relations.append(relation)
                new_relation_labels.append(relation_label)

            loss_relation = log_likelihood(pred, candidate_indices, new_relations, new_relation_labels)

            precision_relation, recall_relation, f1_relation = metrics(pred,
                                                                       torch.tensor(new_relation_labels, device=device))

            total_precision_ner += precision_ner
            total_recall_ner += recall_ner
            total_f1_ner += f1_ner
            total_loss_ner += loss_ner.item()
            total_precision_relation += precision_relation
            total_recall_relation += recall_relation
            total_f1_relation += f1_relation
            total_loss_relation += loss_relation.item()
            count += 1
        print('-' * 150)
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
        print('-' * 150)

        # ...log the running loss
        writer.add_scalar('eval f1_ner', total_f1_ner / count, epoch)
        writer.add_scalar('eval loss_ner', total_loss_ner / count, epoch)

        writer.add_scalar('eval f1_relation', total_f1_relation / count, epoch)
        writer.add_scalar('eval loss_relation', total_loss_relation / count, epoch)

        return total_f1_relation / count


def save_model(model, optimizer, epoch):
    # save model
    checkpoint = {"model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "epoch": epoch}
    path_checkpoint = f"output/checkpoint_epoch_{epoch}.pkl"
    torch.save(checkpoint, path_checkpoint)


epochs = 200
best_f1 = -1
for epoch in range(0, epochs):
    epoch_start_time = time.time()
    print('+' * 150)
    train(train_dataloader, epoch)
    relation_f1 = eval(eval_dataloader, epoch)
    if relation_f1 > best_f1:
        save_model(model, optimizer, epoch)
        best_f1 = relation_f1
    print('+' * 150)
    print('| end of epoch {:3d} | time: {:5.2f}s best_relation_1: {:8.3f}| '
          .format(epoch,
                  time.time() - epoch_start_time, best_f1))
