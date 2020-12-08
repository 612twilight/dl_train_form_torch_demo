# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form_torch_demo
File Name: utils.py
Author: gaoyw
Create Date: 2020/12/8
-------------------------------------------------
"""


def find_entity(pos, text, sequence_tags):
    """

    :param pos: 位置
    :param text: 字序列
    :param sequence_tags: 标签序列
    :return:
    """
    entity = []
    if sequence_tags[pos] in ('B', 'O'):
        entity.append(text[pos])
    else:
        temp_entity = []
        while sequence_tags[pos] == 'I':
            temp_entity.append(text[pos])
            pos -= 1
            if pos < 0:
                break
            if sequence_tags[pos] == 'B':
                temp_entity.append(text[pos])
                break
        entity = list(reversed(temp_entity))
    return entity


def read_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf8') as reader:
        lines = reader.readlines()
    vocab = [line.strip() for line in lines]
    return dict(zip(vocab, range(len(vocab))))


