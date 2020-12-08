# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form_torch_demo
File Name: main.py
Author: gaoyw
Create Date: 2020/12/8
-------------------------------------------------
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name',
                    '-e',
                    type=str,
                    default='conll_bert_re',
                    help='experiments/exp_name.json')
parser.add_argument('--mode',
                    '-m',
                    type=str,
                    default='preprocessing',
                    help='preprocessing|train|evaluation')
args = parser.parse_args()
