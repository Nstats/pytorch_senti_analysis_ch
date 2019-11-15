import pandas as pd
import os
import numpy as np
import random

k = 3
def balance_data(df):
    '''
    :param
        df: a dataframe(['id', 'titla','content', 'label']) with unbalanced data
    :return:
        df_out: a dataframe(['id', 'titla','content', 'label']) with balanced data
    '''
    df_0 = df[df['label'] == 0]
    df_1 = df[df['label'] == 1]
    df_2 = df[df['label'] == 2]
    print('len(df_0)=', len(df_0), 'len(df_1)=', len(df_1), 'len(df_2)=', len(df_2))
    maxNum = max(len(df_0), len(df_1), len(df_2))

    if len(df_0) < maxNum:
        tmp = df_0.sample(n=len(df_0), replace=True).values
        for i in range(len(tmp)):
            context_len = len(tmp[i][2])
            random_num = min(int(0.3*context_len), 5)
            random_index = np.random.randint(0, context_len-random_num-1, random_num)
            for j in random_index:
                tmp[i][2] = tmp[i][2].replace(tmp[i][2][j], '', 1)
        df = df.append(pd.DataFrame(tmp, columns=['id', 'title', 'content', 'label']))
    df_out = df
    print('Now we have {} training samples.'.format(df_out.shape[0]))
    return df_out


train_df_v1 = pd.read_csv("./data/Train_DataSet.csv")
train_label_df_v1 = pd.read_csv("./data/Train_DataSet_Label.csv")
test_df_v1 = pd.read_csv("./data/Test_DataSet.csv")
train_df_v1 = train_df_v1.merge(train_label_df_v1, on='id', how='left')
train_df_v1['label'] = train_df_v1['label'].fillna(-1)
train_df_v1 = train_df_v1[train_df_v1['label'] != -1]
train_df_v1['label'] = train_df_v1['label'].astype(int)
test_df_v1['label'] = 0

test_df_v1['content'] = test_df_v1['content'].fillna('.')
train_df_v1['content'] = train_df_v1['content'].fillna('.')
test_df_v1['title'] = test_df_v1['title'].fillna('.')
train_df_v1['title'] = train_df_v1['title'].fillna('.')

train_df_v2 = pd.read_csv("./data/Second_DataSet.csv")
train_label_df_v2 = pd.read_csv("./data/Second_DataSet_Label.csv")
test_df_v2 = pd.read_csv("./data/Second_TestDataSet.csv")
train_df_v2 = train_df_v2.merge(train_label_df_v2, on='id', how='left')
train_df_v2['label'] = train_df_v2['label'].fillna(-1)
train_df_v2 = train_df_v2[train_df_v2['label'] != -1]
train_df_v2['label'] = train_df_v2['label'].astype(int)
test_df_v2['label'] = 0

test_df_v2['content'] = test_df_v2['content'].fillna('.')
train_df_v2['content'] = train_df_v2['content'].fillna('.')
test_df_v2['title'] = test_df_v2['title'].fillna('.')
train_df_v2['title'] = train_df_v2['title'].fillna('.')

train_df = train_df_v2
test_df = test_df_v2

index = set(range(train_df.shape[0]))
K_fold = []
for i in range(k):
    if i == k-1:
        tmp = index
    else:
        tmp = random.sample(index, int(1.0 / k * train_df.shape[0]))
    index = index - set(tmp)
    print("Number:", len(tmp))
    K_fold.append(tmp)

for i in range(k):
    print("Fold", i)
    if os.path.exists('./data/data_{}'.format(i)):
        os.system("rm -rf ./data/data_{}".format(i))
    os.system("mkdir ./data/data_{}".format(i))
    dev_index = list(K_fold[i])
    train_index = []
    for j in range(k):
        if j != i:
            train_index += K_fold[j]
    train_df_balanced = balance_data(train_df.iloc[train_index])
    train_df_balanced.to_csv("./data/data_{}/train.csv".format(i), index=False)
    train_df.iloc[dev_index].to_csv("./data/data_{}/dev.csv".format(i), index=False)
    test_df.to_csv("./data/data_{}/test.csv".format(i), index=False)
