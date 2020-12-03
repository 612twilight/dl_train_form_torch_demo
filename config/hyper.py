# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form
File Name: hyper.py
Author: gaoyw
Create Date: 2020/12/3
-------------------------------------------------
"""
import configparser
import json


class Hyper(object):
    def __init__(self, config_file="config/config.json"):
        self.config_file = config_file
        self.contains_dev = False  # 是否包含验证集
        self.contains_test = False  # 是否包含测试集
        self.contains_predict = False  # 是否包含待预测数据
        self.__dict__.update(json.load(open(self.config_file, 'r', encoding='utf8')))

    def save_config(self):
        """
        将hyper的数据重新存储到config文件中
        :return:
        """
        with open(self.config_file, 'w', encoding="utf8") as writer:
            writer.write(json.dumps(hyper.__dict__))


hyper = Hyper()
