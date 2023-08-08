## Span-ASTE-Pytorch
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![PyTorch 1.8.1](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![cuDNN 7.6.5](https://img.shields.io/badge/cudnn-7.6.5-green.svg?style=plastic)

This repository is a pytorch version that implements Ali's ACL 2021 research
paper [Learning Span-Level Interactions for Aspect Sentiment Triplet Extraction](https://aclanthology.org/2021.acl-long.367/)
.

-  The [original repository](https://github.com/chiayewken/Span-ASTE.git) of the paper is based on
  the [allennlp](https://docs.allennlp.org/main/) implementation
-  Thanks to Ali's dataset [SemEval-Triplet-data](https://github.com/xuuuluuu/SemEval-Triplet-data.git) that was open
  sourced, so that we can use it for research


![image](https://github.com/wireless911/span-aste/assets/40172030/4116fd42-2457-407a-8613-c76921a72eb0)


### Usage

1. Download dataset from here [SemEval-Triplet-data](https://github.com/xuuuluuu/SemEval-Triplet-data.git),

2. 训练模型
```shell
python train.py \
  --bert_model bert-base-uncased \
  --batch_size 1 \
  --learning_rate 5e-5 \
  --weight_decay 1e-2 \
  --warmup_proportion 0.1 \
  --train_path data/15res \
  --dev_path data/15res \
  --save_dir ./checkpoint \
  --max_seq_len 512 \
  --num_epochs 10 \
  --logging_steps 30 \
  --valid_steps 50
```
5. 模型评估
```shell
python evaluate.py \
  --test_path  data/15res \
  --bert_model bert-base-uncased  \
  --model_path checkpoint/model_best \
  --batch_size 1 \ 
  --max_seq_len 512
  

```








