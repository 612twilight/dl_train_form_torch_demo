# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form_torch_demo
File Name: mydataloader.py
Author: gaoyw
Create Date: 2020/12/8
-------------------------------------------------
"""
from functools import partial

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader


class Batch_reader(object):
    def __init__(self, data):
        """

        :param data: batch_size个元素
        """
        transposed_data = list(zip(*data))  # 解包的过程，将dataset里的元素重新整合

        self.tokens_id = pad_sequence(transposed_data[0], batch_first=True)
        self.bio_id = pad_sequence(transposed_data[1], batch_first=True)
        self.selection_id = torch.stack(transposed_data[2], 0)
        self.length = transposed_data[3]
        self.spo_gold = transposed_data[4]
        self.text = transposed_data[5]
        self.bio = transposed_data[6]

    def pin_memory(self):
        self.tokens_id = self.tokens_id.pin_memory()
        self.bio_id = self.bio_id.pin_memory()
        self.selection_id = self.selection_id.pin_memory()
        return self


def collate_fn(batch):
    return Batch_reader(batch)


Selection_loader = partial(DataLoader, collate_fn=collate_fn, pin_memory=True)
