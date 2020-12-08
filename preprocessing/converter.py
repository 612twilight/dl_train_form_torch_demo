# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form_torch_demo
File Name: converter.py
Author: gaoyw
Create Date: 2020/12/4
-------------------------------------------------
"""
from preprocessing.base_class.base_converter import BaseDataTypeConverter
from typing import Dict, List
from config.hyper import hyper
import json
from utils.utils import find_entity


class DataTypeConverter(BaseDataTypeConverter):
    def __init__(self, relation2index):
        super(DataTypeConverter, self).__init__()
        self.relation2index = relation2index
        self.mid_data = []

    def raw2mid_data(self, raw_path):
        sent = []
        bio = []
        selection_dics = []  # temp
        with open(raw_path, 'r', encoding='utf8') as s:
            for line in s:
                if line.startswith('#'):
                    if sent:
                        triplets = self._process_sent(sent, selection_dics, bio)
                        result = {'text': sent, 'spo_list': triplets,
                                  'bio': bio, 'selection': selection_dics}
                        if len(sent) <= hyper.max_text_len:
                            self.mid_data.append(result)

                    sent = []
                    bio = []
                    selection_dics = []  # temp
                else:
                    num, word, etype, relation, head_list = line.split("\t")[:5]
                    head_list = eval(head_list)
                    relation = eval(relation)
                    sent.append(word)
                    bio.append(etype[0])  # only BIO
                    if relation != ['N']:
                        for r, h in zip(relation, head_list):
                            selection_dics.append(
                                {'subject': int(num), 'predicate': self.relation2index[r], 'object': h})
            if len(sent) <= hyper.max_text_len:
                triplets = self._process_sent(sent, selection_dics, bio)
                result = {'text': sent, 'spo_list': triplets,
                          'bio': bio, 'selection': selection_dics}
                self.mid_data.append(result)

    def mid_data2process(self, process_path):
        with open(process_path, 'w', encoding='utf8') as t:
            for sample in self.mid_data:
                t.write(json.dumps(sample, ensure_ascii=False) + '\n')

    def _process_sent(self, sent: List[str], dic: List[Dict[str, int]], bio: List[str]) -> List:
        id2relation = {v: k for k, v in self.relation2index.items()}
        result = []
        for triplets_id in dic:
            s, p, o = triplets_id['subject'], triplets_id['predicate'], triplets_id['object']
            p = id2relation[p]
            s = find_entity(s, sent, bio)
            o = find_entity(o, sent, bio)

            result.append({'subject': s, 'predicate': p, 'object': o})
        return result

    def predict2mid_data(self, predict_path):
        """
        没有标签的数据，但是对于这个问题，需要填上空标签
        :param predict_path:
        :return:
        """
        sent = []
        with open(predict_path, 'r', encoding='utf8') as s:
            for line in s:
                if line.startswith('#'):
                    if sent:
                        result = {'text': sent, 'spo_list': [], 'bio': bio, 'selection': []}
                        if len(sent) <= hyper.max_text_len:
                            self.mid_data.append(result)
                    sent = []
                    bio = []
                else:
                    num, word = line.split()[:2]
                    sent.append(word)
                    bio.append("O")  # only BIO
            if len(sent) <= hyper.max_text_len:
                result = {'text': sent, 'spo_list': [], 'bio': bio, 'selection': []}
                self.mid_data.append(result)
