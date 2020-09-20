# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time  : 2020/7/23 下午2:19
# @Author: 张伟
# @EMAIL: Jtyoui@qq.com
# @Notes : 基础的数据处理
import json
import os

import tensorflow as tf

from config import *


class DealText:

    def __init__(self, text_path):
        self.text_path = text_path
        self.dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)

    def reader_text(self):
        """读取文件
        将每一行字符和标签读出来，按\n\r来分割每一句话
        """
        words, labels = [], []
        with open(file=self.text_path, mode='r', encoding=ENCODING)as fp:
            word_ls, label_ls = [], []
            for line in fp:
                if line.strip() and len(line.strip().split()) == 2:
                    word, label = line.strip().split()
                    word_ls.append(word)
                    label_ls.append(label)
                elif word_ls:
                    words.append(TEXT_SPLIT_SEP.join(word_ls))
                    labels.append(TEXT_SPLIT_SEP.join(label_ls))
                    word_ls.clear()
                    label_ls.clear()
        return words, labels

    def get_sequence(self, words: list, name: str):
        """获取words的序列

        输入一维列表，如： words: ['我,和，他','他,和,你']
                                    分割符配置 TEXT_SPLIT_SEP=',' \n
                                    最大值配置 MAX_UNM_WORD=3000 \n
                                    文档编码 ENCODING='utf8'

                       name: 列表会解析成文件保存。保存的名字也name为参数。

        :param words: 输入一个列表,['AB','BC'],列表中的每一句话都是用 TEXT_SPLIT_SEP 符号给分割，参考：config配置文件
        :param name: 加载自定义映射表
        :return: 该字符的序列
        """
        assert name in ['words', 'labels'], TypeError('name必须填写words或者labels')
        token = tf.keras.preprocessing.text.Tokenizer(int(MAX_UNM_CHAR), split=TEXT_SPLIT_SEP, filters=TEXT_SPLIT_SEP)
        index_word_file = os.path.join(self.dir, f'index_{name}.json')
        if not os.path.exists(index_word_file):
            token.fit_on_texts(words)  # 构建文件序列
            if name == 'labels':
                # 排序规则。如果长度为1,就是O，根据ascii值伪1
                token.word_index = {k: i for i, (k, v) in enumerate(
                    sorted(token.word_index.items(), key=lambda x: x[0][::-1] if len(x[0]) != 1 else chr(1)))}
                token.index_word = {v: k for k, v in token.word_index.items()}
            with open(index_word_file, mode='w', encoding=ENCODING)as wf:
                json.dump(token.index_word, fp=wf, ensure_ascii=False, indent=4)
        else:
            with open(index_word_file, mode='r', encoding=ENCODING) as rf:
                token.index_word = json.load(rf)
                token.word_index = {v: int(k) for k, v in token.index_word.items()}
        if name == 'words':
            os.environ['VOCAB_SIZE'] = str(len(token.word_index) + 1)  # 更新vocab的大小
        if name == 'labels':
            os.environ['TAG_SIZE'] = str(len(token.word_index))  # 更新TAG的大小
        sequence = token.texts_to_sequences(words)
        return sequence
