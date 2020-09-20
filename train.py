# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time  : 2020/7/23 下午2:21
# @Author: 张伟
# @EMAIL: Jtyoui@qq.com
# @Notes : 训练文件
import tensorflow as tf
import sys
from config import *
from entity_extraction import DealText, NERModel
import os
import json
import tensorflow_addons as tfa
import argparse


def parse_args(argv):
    parser = argparse.ArgumentParser(description='训练执行脚本: python train.py train/test')
    arg = parser.add_argument
    arg('train', help='开始训练')
    arg('test', help='开始测试')
    return parser.parse_args(argv)


args = parse_args(sys.argv)  # 获取脚本的执行命令：python train.py train/test
assert args.test in ['train', 'test'], TypeError('脚本执行命令错误:请执行python train.py -h 查看帮助文档')

dt = DealText(TRAIN_DATA) if args.test == 'train' else DealText(TEST_DATA)

words, labels = dt.reader_text()
word_sequence = dt.get_sequence(words, name='words')
label_sequence = dt.get_sequence(labels, name='labels')
word_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequences=word_sequence,
                                                              maxlen=int(MAX_WORD_LENGTH),
                                                              padding='post',
                                                              dtype='int32',
                                                              truncating='post')

label_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequences=label_sequence,
                                                               maxlen=int(MAX_WORD_LENGTH),
                                                               padding='post',
                                                               dtype='int32',
                                                               truncating='post')
tensor_slices = tf.data.Dataset.from_tensor_slices((word_sequence, label_sequence))
model = NERModel() if args.test == 'train' else tf.saved_model.load(SAVE_MODEL_DIR)

if args.test == 'train':
    optimizers = tf.keras.optimizers.Adam(learning_rate=float(LEARNING_RATE))
    for epoch in range(int(EPOCHS)):
        dataset = tensor_slices.shuffle(buffer_size=int(BUFFER_SIZE)).batch(batch_size=int(BATCH_SIZE))
        for index, (word, label) in enumerate(dataset):
            with tf.GradientTape() as tape:
                y_pred, text_lens, log_likelihood = model.call(word=word, label=label)
                loss = - tf.reduce_mean(log_likelihood)
            grads = tape.gradient(target=loss, sources=model.trainable_variables)
            optimizers.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
            if index % 20 == 0:
                print(f'训练中......第{epoch}次迭代中的第{index}个次数,当前损失值为:{loss}')
    print('训练完毕！正在保存')
    tf.saved_model.save(model, SAVE_MODEL_DIR, signatures={'call': model.call})
else:
    index_tag = json.load(open(os.path.join(dt.dir, 'index_labels.json'), mode='r', encoding=ENCODING))
    index_word = json.load(open(os.path.join(dt.dir, 'index_words.json'), mode='r', encoding=ENCODING))
    index_tag = {int(k): v.upper() for k, v in index_tag.items()}
    dataset = tensor_slices.shuffle(buffer_size=int(BUFFER_SIZE)).batch(batch_size=int(BATCH_SIZE))
    wf = open(os.path.join(dt.dir, 'assess.txt'), mode='w', encoding=ENCODING)
    for words, labels in dataset:
        y_pred, text_lens, _ = model.call(word=words, label=labels)
        for index, (y, l) in enumerate(zip(y_pred, text_lens)):
            word = [index_word[str(i)] for i in words[index][:l].numpy()]
            label = [index_tag[i] for i in labels[index][:l].numpy()]
            pred_label = tfa.text.viterbi_decode(y[:l], model.params)[0]
            pred_label = [index_tag[i] for i in pred_label]
            for w, p, r in zip(word, label, pred_label):
                wf.write(w + '\t' + p + '\t' + r + '\n')
            wf.write('\n')
        wf.flush()
    wf.close()
