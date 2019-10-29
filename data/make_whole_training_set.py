import pandas as pd
import os
import random

train_df = pd.read_csv("./data/Train_DataSet.csv")
train_label_df = pd.read_csv("./data/Train_DataSet_Label.csv")
test_df = pd.read_csv("./data/Test_DataSet.csv")
train_df = train_df.merge(train_label_df, on='id', how='left')
train_df['label'] = train_df['label'].fillna(-1)
train_df = train_df[train_df['label'] != -1]
train_df['label'] = train_df['label'].astype(int)
test_df['label'] = 0

test_df['content'] = test_df['content'].fillna(' ')
train_df['content'] = train_df['content'].fillna(' ')
test_df['title'] = test_df['title'].fillna(' ')
train_df['title'] = train_df['title'].fillna(' ')

train_df.to_csv('./data/train.csv')
