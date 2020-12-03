# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form
File Name: base_converter.py
Author: gaoyw
Create Date: 2020/12/3
-------------------------------------------------
"""


class BaseDataTypeConverter(object):
    def __init__(self, raw_path, process_path, dev_output_path, organized_path):
        """
        将数据格式之间来回转换，注意这里只涉及到数据转换，而不涉及到任何数据预处理
        :param raw_path: 格式等于raw_data下的数据
        :param process_path: 格式等于prepared_data下的数据
        :param dev_output_path: 格式是prepared_data下的数据并合理组织上预测的标签
        :param organized_path: 格式是任意的数据格式 但是应当包含预测标签与原始标签，如果他们之间谁缺了，就将另一个标签作为替代复制一份

        相互转换的规则：
        raw2process process2raw 无损转换
        raw2dev_out raw2organized process2dev_out process2organized 复制原始标签作为预测标签
        dev_out2raw organized2raw dev_out2process organized2process 仅保留原始标签，移除预测标签
        """
        self.raw_path = raw_path
        self.process_path = process_path
        self.dev_output_path = dev_output_path
        self.organized_path = organized_path
        self.mid_data = None

    def raw2mid_data(self):
        raise NotImplementedError()

    def process2mid_data(self):
        raise NotImplementedError()

    def dev_out2mid_data(self):
        raise NotImplementedError()

    def organized2mid_data(self):
        raise NotImplementedError()

    def mid_data2raw(self):
        raise NotImplementedError()

    def mid_data2process(self):
        raise NotImplementedError()

    def mid_data2dev_out(self):
        raise NotImplementedError()

    def mid_data2organized(self):
        raise NotImplementedError()

    def raw2process(self):
        self.raw2mid_data()
        self.mid_data2process()

    def process2raw(self):
        self.process2mid_data()
        self.mid_data2raw()

    def raw2dev_out(self):
        self.raw2mid_data()
        self.mid_data2dev_out()

    def raw2organized(self):
        self.raw2mid_data()
        self.mid_data2organized()

    def process2dev_out(self):
        self.process2mid_data()
        self.mid_data2dev_out()

    def process2organized(self):
        self.process2mid_data()
        self.mid_data2organized()

    def dev_out2raw(self):
        self.dev_out2mid_data()
        self.mid_data2raw()

    def organized2raw(self):
        self.organized2mid_data()
        self.mid_data2raw()

    def dev_out2process(self):
        self.dev_out2mid_data()
        self.mid_data2process()

    def organized2process(self):
        self.organized2mid_data()
        self.mid_data2process()

    def organized2dev_out(self):
        self.organized2mid_data()
        self.mid_data2dev_out()
