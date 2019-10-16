import pandas as pd
import os
import random
from langconv import *


def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence

train_df=pd.read_csv("./data/Train_DataSet.csv")
train_label_df=pd.read_csv("./data/Train_DataSet_Label.csv")
test_df=pd.read_csv("./data/Test_DataSet.csv")
train_df=train_df.merge(train_label_df,on='id',how='left')
train_df['label']=train_df['label'].fillna(-1)
train_df=train_df[train_df['label']!=-1]
train_df['label']=train_df['label'].astype(int)
test_df['label']=0

test_df['content']=test_df['content'].fillna(' ')
train_df['content']=train_df['content'].fillna(' ')
test_df['title']=test_df['title'].fillna(' ')
train_df['title']=train_df['title'].fillna(' ')

train_title = []
for line in train_df['title']:
    line = Traditional2Simplified(line)
    train_title.append(line.replace('\n', '').replace('\\n', '').replace(' ', '').replace('\t', ''))
train_df['title'] = train_title

test_title = []
for line in test_df['title']:
    line = Traditional2Simplified(line)
    test_title.append(line.replace('\n', '').replace('\\n', '').replace(' ', '').replace('\t', ''))
test_df['title'] = test_title

train_content = []
for line in train_df['content']:
    line = Traditional2Simplified(line)
    train_content.append(line.replace('\n', '').replace('\\n', '').replace(' ', '').replace('\t', ''))
train_df['content'] = train_content

test_content = []
for line in test_df['content']:
    line = Traditional2Simplified(line)
    test_content.append(line.replace('\n', '').replace('\\n', '').replace(' ', '').replace('\t', ''))
test_df['content'] = test_content

index=set(range(train_df.shape[0]))
K_fold=[]
for i in range(5):
    if i == 4:
        tmp=index
    else:
        tmp=random.sample(index,int(1.0/5*train_df.shape[0]))
    index=index-set(tmp)
    print("Number:",len(tmp))
    K_fold.append(tmp)
    

for i in range(5):
    print("Fold",i)
    if os.path.exists('./data/data_{}'.format(i)):
        os.system("rm -rf ./data/data_{}".format(i))
    os.system("mkdir ./data/data_{}".format(i))
    dev_index=list(K_fold[i])
    train_index=[]
    for j in range(5):
        if j!=i:
            train_index+=K_fold[j]
    train_df.iloc[train_index].to_csv("./data/data_{}/train.csv".format(i))
    train_df.iloc[dev_index].to_csv("./data/data_{}/dev.csv".format(i))
    test_df.to_csv("./data/data_{}/test.csv".format(i))
