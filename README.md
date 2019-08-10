# BERT

## Description

this repository is implementation of [BERT; Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

![](https://pytorch.org/assets/images/bert1.png)

ABSTRACT:

> We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be finetuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.
> BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

## Usage

- pre-training 

- prepare data, format must be like below

```
[sentence1] [sentence2]
[sentence3] ...

```

- create data loader

dataloader is at `dataset/`

- pre-train

pretrain method is at `trainer/`

- fine-tuning

once you pretrain BERT model, you can use saved pretrain model for finetuning. please edit script yourself and finetune.

## Reference

- [BERT-pytorch](https://github.com/codertimo/BERT-pytorch)
- [Qiita; BERT for Japanese](https://qiita.com/Kosuke-Szk/items/4b74b5cce84f423b7125)

