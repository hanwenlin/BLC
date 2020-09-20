# EntityExtraction
TensorFlow2+Python3.8实现实体抽取模型

## 实现的模型
[BERT+BiLSTM+CRF]()     
[CNN+BiLSTM+CRF](https://github.com/jtyoui/BLC/tree/cnn+bilstm+crf)    
[BiLSTM+CRF](https://github.com/jtyoui/BLC/tree/bilstm+crf) 

## 数据格式
    只需要训练集和测试集：数据格式
    
    字符 （空格）标签
    向 B-DIS
    日 M-DIS
    葵 M-DIS
    缺 M-DIS
    素 M-DIS
    症 E-DIS
    症 O
    状 O
    一 O
    缺 O
    氮 O
    苗 O

## 超参数
```python
from os import environ
MAX_UNM_CHAR = environ.setdefault('MAX_UNM_CHAR', '3000') #保存最多字符编码
ENCODING = environ.setdefault('ENCODING', 'utf8') # 文本编码
VOCAB_SIZE = environ.setdefault('VOCAB_SIZE', 'AUTO') # 不需要设置
EMBEDDING_SIZE = environ.setdefault('EMBEDDING_SIZE', '500') #EMBEDDING的大小
TAG_SIZE = environ.setdefault('TAG_SIZE', 'AUTO') # 不需要设置
BUFFER_SIZE = environ.setdefault('BUFFER_SIZE', '1000') # 一次训练的缓存条数大小
BATCH_SIZE = environ.setdefault('BATCH_SIZE', '32') #BATCH大小
EPOCHS = environ.setdefault('EPOCHS', '10') #EPOCHS数量
MAX_WORD_LENGTH = environ.setdefault('MAX_WORD_LENGTH', '250') #这句话最大字符长度
LSTM_SIZE = environ.setdefault('LSTM_SIZE', '256') #LSTM大小 
CNN_FILTER = environ.setdefault('CNN_FILTER', '32')#CNN过滤
LEARNING_RATE = environ.setdefault('LEARNING_RATE', '0.001') #学习率
SAVE_MODEL_DIR = environ.setdefault('SAVE_MODEL_DIR', 'models/1') #模型保存的路径
CNN_FILTER = environ.setdefault('CNN_FILTER', '32') # CNN大小
TRAIN_DATA = environ.setdefault('TRAIN_DATA', 'data/train.txt') #训练集
TEST_DATA = environ.setdefault('TEST_DATA', 'data/test.txt')# 测试集
```

## 训练与测试
    训练  python train.py train 
    测试  python train.py test

## 生成评估报告
    测试后会在data文件里面生成assess.txt文件
    执行master分支中的CoNLL脚本
    python CoNLL-2000.py data/assess.txt
    
## 评估结果
```text
平均准确度:  98.88%; 平均精确度:  85.72%; 平均召回率:  84.30%; 平均F1值:  85.54%
              LOC: 精确度:  88.11%; 召回率:  85.27%; F1值:  86.66%
              ORG: 精确度:  86.11%; 召回率:  85.08%; F1值:  87.22%
              PER: 精确度:  82.94%; 召回率:  82.56%; F1值:  82.75%  
```

## 模型保存文件夹格式
```text
/ner        # 保存文件夹取的名字
    /1      # 版本号为1的模型文件
        /assets
        /variables
        saved_model.pb
    ...
    /N      # 版本号为N的模型文件
        /assets
        /variables
        saved_model.pb
```

## 模型部署
    docker pull tensorflow/serving
    docker run -d -p 8501:8501 -v /mnt/ner:/models/ner -e MODEL_NAME=ner --name=tensorflow-serving tensorflow/serving
    
## REStFul接口调用
```python
import requests
import json
import os

dirs = os.path.dirname(__file__)
index_tag = json.load(open(os.path.join(dirs, 'data', 'index_labels.json'), mode='r', encoding='utf-8'))
tag_index = {tag: int(index) for index, tag in index_tag.items()}
index_word = json.load(open(os.path.join(dirs, 'data', 'index_words.json'), mode='r', encoding='utf-8'))
word_index = {word: int(index) for index, word in index_word.items()}

url = 'http://IP:8501/v1/models/ner:predict'
headers = {"content-type": "application/json"}

def ner(words: str):
    length = len(words)
    assert length <= 200, ValueError('输入的字符串最大长度为200')
    word = [word_index.get(i, 0) for i in words]
    data = json.dumps({
        'inputs': {
            "word": [word + [0] * (200 - length)],
            'label': [[0] * 200],
        },
        'signature_name': 'call'
    })
    result = requests.post(url, data=data, headers=headers).json()
    value = result['outputs']['output_0']
    ls = [v.index(max(v)) for v in value[0]]

    print(ls[:length])
    print(words)

if __name__ == '__main__':
    ner('我叫李伟')
```