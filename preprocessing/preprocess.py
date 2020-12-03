# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form_torch_demo
File Name: preprocess.py
Author: gaoyw
Create Date: 2020/12/3
-------------------------------------------------
"""
from preprocessing.base_class.base_processe_raw_data import BaseProcessing
from config.hyper import hyper
from collections import Counter
import json
import os
from typing import Dict, List, Tuple, Set, Optional


class Preprocessing(BaseProcessing):
    def __init__(self):
        super(BaseProcessing).__init__()
        self.relation_vocab_set = set()
        self.bio_vocab = {}
        self.word_vocab = Counter()
        self.relation2index = None

    def build_vocab(self):
        """
        根据训练数据构建字典
        :return:
        """
        # 读取训练集数据
        self.one_pass_train()
        # 建立字典
        self.gen_relation_vocab()
        self.gen_vocab()

    def one_pass_train(self):
        sent = []
        with open(self.train_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    if sent:
                        self.word_vocab.update(sent)
                    sent = []
                else:
                    assert len(self._parse_line(line)) == 5
                    num, word, etype, relation, head_list = self._parse_line(line)
                    relation = eval(relation)
                    sent.append(word)
                    if relation != ['N']:
                        self.relation_vocab_set.update(relation)
            self.word_vocab.update(sent)

    def gen_vocab(self, min_freq: int = 0):
        result = {'<pad>': 0, '<unk>': 1}
        i = 2
        for k, v in self.word_vocab.items():
            if v > min_freq:
                result[k] = i
                i += 1
        with open(hyper.vocab_path, 'w', encoding="utf8") as writer:
            for word in result:
                writer.write(word + "\n")

    def gen_relation_vocab(self):
        self.relation_vocab_set.add("N")
        relation_vocab = list(sorted(self.relation_vocab_set))
        self.relation2index = dict(zip(relation_vocab, range(len(relation_vocab))))
        with open(hyper.label_path, 'w', encoding="utf8") as writer:
            for label in relation_vocab:
                writer.write(label + "\n")

    def handle_labeled_data(self, in_file_path, out_file_path):
        """
        处理已经有标注的数据
        :param in_file_path:
        :param out_file_path:
        :return:
        """
        sent = []
        bio = []
        selection_dics = []  # temp
        with open(in_file_path, 'r') as s, open(out_file_path, 'w') as t:
            for line in s:
                if line.startswith('#'):
                    if sent != []:
                        triplets = self._process_sent(sent, selection_dics, bio)
                        result = {'text': sent, 'spo_list': triplets,
                                  'bio': bio, 'selection': selection_dics}
                        if len(sent) <= hyper.max_text_len:
                            t.write(json.dumps(result))
                            t.write('\n')
                    sent = []
                    bio = []
                    selection_dics = []  # temp
                else:
                    assert len(self._parse_line(line)) == 5
                    num, word, etype, relation, head_list = self._parse_line(line)
                    head_list = eval(head_list)
                    relation = eval(relation)
                    sent.append(word)
                    bio.append(etype[0])  # only BIO
                    if relation != ['N']:
                        self.relation_vocab_set.update(relation)
                        for r, h in zip(relation, head_list):
                            selection_dics.append(
                                {'subject': int(num), 'predicate': self.relation2index[r], 'object': h})
            if len(sent) <= hyper.max_text_len:
                triplets = self._process_sent(sent, selection_dics, bio)
                result = {'text': sent, 'spo_list': triplets,
                          'bio': bio, 'selection': selection_dics}
                t.write(json.dumps(result))

    @staticmethod
    def _find_entity(pos, text, sequence_tags):
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

    def _process_sent(self, sent: List[str], dic: List[Dict[str, int]], bio: List[str]) -> List[Dict]:
        id2relation = {v: k for k, v in self.relation2index.items()}
        result = []
        for triplets_id in dic:
            s, p, o = triplets_id['subject'], triplets_id['predicate'], triplets_id['object']
            p = id2relation[p]
            s = self._find_entity(s, sent, bio)
            o = self._find_entity(o, sent, bio)
            result.append({'subject': s, 'predicate': p, 'object': o})
        return result

    def handle_unlabeled_data(self, in_file_path, out_file_path):
        """
        处理无标注的数据
        :param in_file_path:
        :param out_file_path:
        :return:
        """

        sent = []
        bio = []
        selection_dics = []  # temp
        with open(in_file_path, 'r') as s, open(out_file_path, 'w') as t:
            for line in s:
                if line.startswith('#'):
                    if sent != []:
                        triplets = self._process_sent(sent, selection_dics, bio)
                        result = {'text': sent, 'spo_list': triplets,
                                  'bio': bio, 'selection': selection_dics}
                        if len(sent) <= hyper.max_text_len:
                            t.write(json.dumps(result))
                            t.write('\n')
                    sent = []
                    bio = []
                    selection_dics = []  # temp
                else:
                    assert len(self._parse_line(line)) == 5
                    num, word, etype, relation, head_list = line.split()
                    head_list = eval(head_list)
                    relation = eval(relation)
                    sent.append(word)
                    bio.append(etype[0])  # only BIO
                    if relation != ['N']:
                        self.relation_vocab_set.update(relation)
                        for r, h in zip(relation, head_list):
                            selection_dics.append(
                                {'subject': int(num), 'predicate': self.relation2index[r], 'object': h})
            if len(sent) <= hyper.max_text_len:
                triplets = self._process_sent(sent, selection_dics, bio)
                result = {'text': sent, 'spo_list': triplets,
                          'bio': bio, 'selection': selection_dics}
                t.write(json.dumps(result))

        pass

    @staticmethod
    def _parse_line(line):
        result = line.split()
        if len(result) == 5:
            return result
        else:
            a, b, c = result[:3]
            de = result[3:]
            d, e = [], []
            cur = d
            for t in de:
                cur.append(t)
                if t.endswith(']'):
                    cur = e
            return a, b, c, ''.join(d), ''.join(e)
