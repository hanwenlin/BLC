# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time  : 2020/7/23 下午2:18
# @Author: 张伟
# @EMAIL: Jtyoui@qq.com
# @Notes : 构建模型文件
import os

import tensorflow as tf
import tensorflow_addons as tfa

from config import *


# tf.config.experimental_run_functions_eagerly(True)  # 调试


class NERModel(tf.keras.Model):
    def get_config(self):
        pass

    def __init__(self):
        super().__init__()
        tag_size = int(os.environ['TAG_SIZE'])
        self.embedding = tf.keras.layers.Embedding(int(os.environ['VOCAB_SIZE']), output_dim=int(EMBEDDING_SIZE))
        self.bi_lst = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=int(LSTM_SIZE), return_sequences=True))
        self.dense = tf.keras.layers.Dense(units=tag_size)
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.params = tf.Variable(tf.random.uniform(shape=(tag_size, tag_size)))

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, int(MAX_WORD_LENGTH)), dtype=tf.int32, name='word'),
                                  tf.TensorSpec(shape=(None, int(MAX_WORD_LENGTH)), dtype=tf.int32, name='label')])
    def call(self, word, label):
        text_length = tf.math.reduce_sum(tf.cast(tf.math.not_equal(word, 0), dtype=tf.int32), axis=-1)  # (BATCH_SIZE,)
        x = self.embedding(word)  # (BATCH_SIZE, MAX_WORD_LENGTH, EMBEDDING_SIZE)
        x = tf.expand_dims(x, axis=1)  # (BATCH_SIZE,1, MAX_WORD_LENGTH, EMBEDDING_SIZE)
        x = tf.squeeze(x, axis=1)  # (BATCH_SIZE, MAX_WORD_LENGTH, 32)
        x = self.dropout(x)  # (BATCH_SIZE, MAX_WORD_LENGTH, EMBEDDING_SIZE)
        x = self.bi_lst(x)  # (BATCH_SIZE, MAX_WORD_LENGTH, int(LSTM_SIZE) * 2)
        x = self.dense(x)  # (BATCH_SIZE, MAX_WORD_LENGTH, TAG_SIZE)
        label = tf.convert_to_tensor(label, dtype=tf.int32)
        log_likelihood, self.params = tfa.text.crf_log_likelihood(inputs=x,
                                                                  tag_indices=label,
                                                                  sequence_lengths=text_length,
                                                                  transition_params=self.params)
        return x, text_length, log_likelihood
