# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time  : 2020/7/23 下午2:21
# @Author: 张伟
# @EMAIL: Jtyoui@qq.com
# @Notes : 训练文件
from kashgari.tasks.labeling import BiLSTM_CRF_Model
from kashgari.embeddings import BertEmbedding
from kashgari.callbacks import EvalCallBack
from tensorflow.keras.utils import get_file
from kashgari.corpus import DataReader
from tensorflow import keras
import zipfile
import json
import os

SEQUENCE_LENGTH = 200  # 序列长度
EPOCHS = 20
EARL_STOPPING_PATIENCE = 10
REDUCE_RL_PATIENCE = 5
TRAIN_FILE = 'data/train.txt'
TEST_FILE = 'data/test.txt'
MODEL_PATH = 'model'


def download_bert_if_needs(parent_dir: str) -> str:
    bert_path = os.path.join(parent_dir, 'chinese_L-12_H-768_A-12')
    if not os.path.exists(bert_path):
        zip_file_path = get_file('chinese_L-12_H-768_A-12.zip',
                                 'http://oss.jtyoui.com/model/chinese_L-12_H-768_A-12.zip',
                                 untar=False,
                                 cache_subdir='',
                                 cache_dir=parent_dir
                                 )
        unzipped_file = zipfile.ZipFile(zip_file_path, "r")
        unzipped_file.extractall(path=parent_dir)
    return bert_path


train_x, train_y = DataReader.read_conll_format_file(TRAIN_FILE)
test_x, test_y = DataReader.read_conll_format_file(TEST_FILE)

BERT_PATH = download_bert_if_needs(os.path.dirname(__file__))
embeddings = BertEmbedding(BERT_PATH)
model = BiLSTM_CRF_Model(embeddings, sequence_length=SEQUENCE_LENGTH)
early_stop = keras.callbacks.EarlyStopping(patience=EARL_STOPPING_PATIENCE)
reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=REDUCE_RL_PATIENCE)
eval_callback = EvalCallBack(kash_model=model, x_data=test_x, y_data=test_y, truncating=True, step=1)
callbacks = [early_stop, reduce_lr_callback, eval_callback]
model.fit(train_x, train_y, test_x, test_y, callbacks=callbacks, epochs=EPOCHS)
with open(os.path.join(MODEL_PATH, 'model_config.json'), 'w', encoding='utf-8') as f:
    f.write(json.dumps(model.to_dict(), indent=4, ensure_ascii=False))
model.embedding.embed_model.save_weights(os.path.join(MODEL_PATH, 'embed_model_weights.h5'))
model.tf_model.save_weights(os.path.join(MODEL_PATH, 'model_weights.h5'))
