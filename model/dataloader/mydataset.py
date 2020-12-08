# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form_torch_demo
File Name: mydataset.py
Author: gaoyw
Create Date: 2020/12/8
-------------------------------------------------
"""
import json
from typing import List

import torch
from torch.utils.data import Dataset

from config.hyper import hyper
from utils.utils import read_vocab


class MultiHeadDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, file_path):
        self.word2index = read_vocab(hyper.vocab_path)
        self.relation2index = read_vocab(hyper.label_path)
        self.bio2index = read_vocab(hyper.bio_label_path)

        self.selection_list = []
        self.text_list = []
        self.bio_list = []
        self.spo_list = []

        for line in open(file_path, 'r', encoding='utf8'):
            line = line.strip("\n")
            instance = json.loads(line)

            self.selection_list.append(instance['selection'])
            self.text_list.append(instance['text'])
            self.bio_list.append(instance['bio'])
            self.spo_list.append(instance['spo_list'])

    def __getitem__(self, index):
        selection = self.selection_list[index]
        text = self.text_list[index]
        bio = self.bio_list[index]
        spo = self.spo_list[index]
        tokens_id = self.text2tensor(text)
        bio_id = self.bio2tensor(bio)
        selection_id = self.selection2tensor(text, selection)

        return tokens_id, bio_id, selection_id, len(text), spo, text, bio

    def __len__(self):
        return len(self.text_list)

    def text2tensor(self, text: List[str]) -> torch.tensor:
        # TODO: tokenizer
        oov = self.word2index['<unk>']
        padded_list = list(map(lambda x: self.word2index.get(x, oov), text))
        padded_list.extend([self.word2index['<pad>']] * (hyper.max_text_len - len(text)))
        return torch.tensor(padded_list)

    def bio2tensor(self, bio):
        # here we pad bio with "O". Then, in our model, we will mask this "O" padding.
        # in multi-head selection, we will use "<pad>" token embedding instead.
        padded_list = list(map(lambda x: self.bio2index[x], bio))
        padded_list.extend([self.bio2index['O']] * (hyper.max_text_len - len(bio)))
        return torch.tensor(padded_list)

    def selection2tensor(self, text, selection):
        # s p o
        result = torch.zeros((hyper.max_text_len, len(self.relation2index), hyper.max_text_len))
        NA = self.relation2index['N']
        result[:, NA, :] = 1
        for triplet in selection:
            object = triplet['object']
            subject = triplet['subject']
            predicate = triplet['predicate']
            result[subject, predicate, object] = 1
            result[subject, NA, object] = 0
        return result
