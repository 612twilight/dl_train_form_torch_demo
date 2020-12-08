# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form_torch_demo
File Name: preprocess.py
Author: gaoyw
Create Date: 2020/12/3
-------------------------------------------------
"""
from collections import Counter

from config.hyper import hyper
from preprocessing.base_class.base_processe_raw_data import BaseProcessing
from preprocessing.converter import DataTypeConverter


class Preprocessing(BaseProcessing):
    def __init__(self):
        super(Preprocessing, self).__init__()
        self.relation_vocab_set = set()
        self.bio_vocab = {}
        self.word_vocab = Counter()
        self.relation2index = None  # 执行build_vocab才可以有这个对象，否则只能从文件中读取加载
        self.converter = None

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
        self.gen_bio_vocab()
        self.converter = DataTypeConverter(self.relation2index)

    def one_pass_train(self):
        """
        这里我们默认认为训练数据里面有全部的标签，需要在最初分割数据的时候就要做到
        :return:
        """
        sent = []
        with open(self.train_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    if sent:
                        self.word_vocab.update(sent)
                    sent = []
                else:
                    num, word, etype, relation, head_list = line.split("\t")[:5]
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

    def gen_bio_vocab(self):
        result = ['<pad>', 'B', 'I', 'O']
        with open(hyper.bio_label_path, 'w', encoding="utf8") as writer:
            for label in result:
                writer.write(label + "\n")

    def handle_labeled_data(self, in_file_path, out_file_path):
        """
        处理已经有标注的数据
        :param in_file_path:
        :param out_file_path:
        :return:
        """
        self.converter.raw2process(in_file_path, out_file_path)

    def handle_unlabeled_data(self, in_file_path, out_file_path):
        """
        处理无标注的数据
        :param in_file_path:
        :param out_file_path:
        :return:
        """
        self.converter.predict2process(in_file_path, out_file_path)


if __name__ == '__main__':
    preprocessor = Preprocessing()
    preprocessor.build_vocab()
