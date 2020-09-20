# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time  : 2020/7/23 下午2:21
# @Author: 张伟
# @EMAIL: Jtyoui@qq.com
# @Notes : 配置文件
from os import environ

MAX_UNM_CHAR = environ.setdefault('MAX_UNM_CHAR', '3000')
TEXT_SPLIT_SEP = environ.setdefault('TEXT_SPLIT_SEP', '\x02')
ENCODING = environ.setdefault('ENCODING', 'utf8')
VOCAB_SIZE = environ.setdefault('VOCAB_SIZE', 'AUTO')
EMBEDDING_SIZE = environ.setdefault('EMBEDDING_SIZE', '500')
TAG_SIZE = environ.setdefault('TAG_SIZE', 'AUTO')
BUFFER_SIZE = environ.setdefault('BUFFER_SIZE', '1000')
BATCH_SIZE = environ.setdefault('BATCH_SIZE', '32')
EPOCHS = environ.setdefault('EPOCHS', '10')
MAX_WORD_LENGTH = environ.setdefault('MAX_WORD_LENGTH', '250')
LSTM_SIZE = environ.setdefault('LSTM_SIZE', '256')
CNN_FILTER = environ.setdefault('CNN_FILTER', '32')
LEARNING_RATE = environ.setdefault('LEARNING_RATE', '0.001')
SAVE_MODEL_DIR = environ.setdefault('SAVE_MODEL_DIR', 'models/4')
TRAIN_DATA = environ.setdefault('TRAIN_DATA', 'data/train.txt')
TEST_DATA = environ.setdefault('TEST_DATA', 'data/test.txt')
