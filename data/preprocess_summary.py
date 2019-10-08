import pandas as pd
import os
import random
from pyhanlp import *
import numpy as np

TextRankSentence = JClass("com.hankcs.hanlp.summary.TextRankSentence")

train_df = pd.read_csv("./data/Train_DataSet.csv")
train_label_df = pd.read_csv("./data/Train_DataSet_Label.csv")
test_df = pd.read_csv("./data/Test_DataSet.csv")
train_df = train_df.merge(train_label_df, on='id', how='left')
train_df['label'] = train_df['label'].fillna(-1)
train_df = train_df[train_df['label'] != -1]
train_df['label'] = train_df['label'].astype(int)
test_df['label'] = 0

test_content = test_df['content'].fillna(' ')
test_content_ = []
for ele in test_content:
    if len(ele) >= 100:
        ele_ = HanLP.getSummary(ele, 600)
        # ele_ = HanLP.extractSummary(ele, 20)
        test_content_.append(ele_)
    else:
        test_content_.append(ele)
test_df['content'] = np.asarray(test_content_)

train_content = train_df['content'].fillna(' ').values
train_content_ = []
for ele in train_content:
    if len(ele) >= 100:
        ele_ = HanLP.getSummary(ele, 600)
        # ele_ = HanLP.extractSummary(ele, 20)
        train_content_.append(ele_)
    else:
        train_content_.append(ele)
train_df['content'] = np.asarray(train_content_)
test_df['title'] = test_df['title'].fillna(' ')
train_df['title'] = train_df['title'].fillna(' ')

index = set(range(train_df.shape[0]))
K_fold = []
for i in range(5):
    if i == 4:
        tmp = index
    else:
        tmp = random.sample(index, int(1.0 / 5 * train_df.shape[0]))
    index = index - set(tmp)
    print("Number:", len(tmp))
    K_fold.append(tmp)

for i in range(5):
    print("Fold", i)
    if os.path.exists('./data/data_{}'.format(i)):
        os.system("rm -rf ./data/data_{}".format(i))
    os.system("mkdir ./data/data_{}".format(i))
    dev_index = list(K_fold[i])
    train_index = []
    for j in range(5):
        if j != i:
            train_index += K_fold[j]
    train_df.iloc[train_index].to_csv("./data/data_{}/train.csv".format(i))
    train_df.iloc[dev_index].to_csv("./data/data_{}/dev.csv".format(i))
    test_df.to_csv("./data/data_{}/test.csv".format(i))
