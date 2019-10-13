import pandas as pd
import os
import re
import random


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
    print(len(df_0), len(df_1), len(df_2))
    maxNum = max(len(df_0), len(df_1), len(df_2))

    if len(df_0) < maxNum:
        df = df.append(df_0.sample(n=maxNum - len(df_0), replace=True))
    if len(df_1) < maxNum:
        df = df.append(df_1.sample(n=maxNum - len(df_1), replace=True))
    if len(df_2) < maxNum:
        df = df.append(df_2.sample(n=maxNum - len(df_2), replace=True))
    df_out = df
    print('Now we have {} training samples.'.format(df_out.shape[0]))
    return df_out


if __name__ == '__main__':
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
        train_df = balance_data(train_df.iloc[train_index])
        train_df.to_csv("./data/data_{}/train.csv".format(i))
        train_df.iloc[dev_index].to_csv("./data/data_{}/dev.csv".format(i))
        test_df.to_csv("./data/data_{}/test.csv".format(i))
