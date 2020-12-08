# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form
File Name: base_processe_raw_data.py
Author: gaoyw
Create Date: 2020/12/3
-------------------------------------------------
"""
import os

from config.hyper import hyper
from logger import log


class BaseProcessing(object):
    def __init__(self, train_file="raw_data/train.txt",
                 dev_file="raw_data/dev.txt",
                 test_file="raw_data/test.txt",
                 predict_file="raw_data/predict.txt",
                 prepared_train_file="prepared_data/train.txt",
                 prepared_dev_file="prepared_data/dev.txt",
                 prepared_test_file="prepared_data/test.txt",
                 prepared_predict_file="prepared_data/predict.txt"
                 ):
        self.train_file = train_file
        self.test_file = test_file
        self.dev_file = dev_file
        self.predict_file = predict_file
        self.prepared_train_file = prepared_train_file
        self.prepared_test_file = prepared_test_file
        self.prepared_dev_file = prepared_dev_file
        self.prepared_predict_file = prepared_predict_file
        self.dataset_check()
        # self.build_vocab()
        # self.handle_data()
        hyper.save_config()

    def preprocess(self):
        self.build_vocab()
        self.handle_data()

    def build_vocab(self):
        """
        根据您训练集构建字典
        :return:
        """
        raise NotImplementedError()

    def handle_labeled_data(self, in_file_path, out_file_path):
        """
        处理train dev test这种数据
        :return:
        """
        raise NotImplementedError()

    def handle_unlabeled_data(self, in_file_path, out_file_path):
        """
        处理predict这种数据
        :return:
        """
        raise NotImplementedError()

    def handle_data(self):
        """
        将原始数据进行预处理，转为模型输入的数据格式
        转换后的数据应当存储到
        :return:
        """
        self.handle_labeled_data(self.train_file, self.prepared_train_file)
        if hyper.contains_dev:
            self.handle_labeled_data(self.dev_file, self.prepared_dev_file)
        if hyper.contains_test:
            self.handle_labeled_data(self.test_file, self.prepared_test_file)
        if hyper.contains_predict:
            self.handle_unlabeled_data(self.predict_file, self.prepared_predict_file)

    def dataset_check(self):
        """
        检测数据集是否存在，并记录
        :return:
        """
        if not os.path.exists(self.train_file):
            log.error("没有找到训练集数据，训练集数据应位于{}".format(self.train_file))
            exit(-1)

        if not os.path.exists(self.dev_file):
            log.info("没有找到验证集数据，验证集数据可以位于{}".format(self.dev_file))
            hyper.contains_dev = False
        else:
            log.info("找到验证集数据，验证集数据位于{}".format(self.dev_file))
            hyper.contains_dev = True

        if not os.path.exists(self.test_file):
            log.info("没有找到测试集数据，测试集数据可以位于{}".format(self.test_file))
            hyper.contains_test = False
        else:
            log.info("找到测试集数据，测试集数据位于{}".format(self.test_file))
            hyper.contains_test = True

        if not os.path.exists(self.predict_file):
            log.info("没有找到待预测数据，待预测数据可以位于{}".format(self.predict_file))
            hyper.contains_predict = False
        else:
            log.info("找到待预测数据，待预测数据位于{}".format(self.predict_file))
            hyper.contains_predict = True
