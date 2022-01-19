## Span-ASTE-Pytorch
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![PyTorch 1.8.1](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![cuDNN 7.6.5](https://img.shields.io/badge/cudnn-7.6.5-green.svg?style=plastic)

This repository is a pytorch version that implements Ali's ACL 2021 research
paper [Learning Span-Level Interactions for Aspect Sentiment Triplet Extraction](https://aclanthology.org/2021.acl-long.367/)
.

- 🤗 The [original repository](https://github.com/chiayewken/Span-ASTE.git) of the paper is based on
  the [allennlp](https://docs.allennlp.org/main/) implementation
- 🤗 Thanks to Ali's dataset [SemEval-Triplet-data](https://github.com/xuuuluuu/SemEval-Triplet-data.git) that was open
  sourced, so that we can use it for research

### Usage

1. 🍉 Download dataset from here [SemEval-Triplet-data](https://github.com/xuuuluuu/SemEval-Triplet-data.git),
  ASTE-Data-V2-EMNLP2020 is used in my repository
2. 🥭 Download [GloVe](https://github.com/stanfordnlp/GloVe.git) pre-trained word vectors,
3. 🍑 Convert `glove_input_file` in GloVe format to word2vec format and write it to `word2vec_output_file
```shell
from gensim.scripts.glove2word2vec import glove2word2vec

glove2word2vec("path/to/dir/glove_input_file", "path/to/dir/word2vec_output_file")

```
4. 🍓 train the span-aste model
```shell
python train.py --glove_word2vector vector_cache/w2v.txt \
          --dataset data/ASTE-Data-V2-EMNLP2020/15res/ \
          --output_path output/
```
5. 🍇 test the span-aste model
```shell
python test.py --model_path  \
          --glove_word2vector corpus/w2v.txt \
          --dataset data/ASTE-Data-V2-EMNLP2020/15res/\
          -model `path/to/model/model.pkl`

```








