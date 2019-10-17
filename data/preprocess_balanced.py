import pandas as pd
import os
import re
import random

if __name__ == '__main__':
    train_df = pd.read_csv("./data/Train_DataSet.csv")
    train_label_df = pd.read_csv("./data/Train_DataSet_Label.csv")
    test_df = pd.read_csv("./data/Test_DataSet.csv")
    train_df = train_df.merge(train_label_df, on='id', how='left')
    train_df['label'] = train_df['label'].fillna(-1)
    train_df = train_df[train_df['label'] != -1]
    train_df['label'] = train_df['label'].astype(int)
    test_df['label'] = 0

    test_df['content'] = test_df['content'].fillna('无 ')
    train_df['content'] = train_df['content'].fillna('无 ')
    test_df['title'] = test_df['title'].fillna('无 ')
    train_df['title'] = train_df['title'].fillna('无 ')

    train_0_df = train_df[train_df['label'] == 0]
    train_1_df = train_df[train_df['label'] == 1]
    train_2_df = train_df[train_df['label'] == 2]
    # print(len(train_0_df), len(train_1_df), len(train_2_df))  # 763 3646 2931
    maxNum = max(len(train_0_df), len(train_1_df), len(train_2_df))

    if len(train_0_df) < maxNum:
        train_df = train_df.append(train_0_df.sample(n=maxNum-len(train_0_df), replace=True))
    if len(train_1_df) < maxNum:
        train_df = train_df.append(train_1_df.sample(n=maxNum-len(train_1_df), replace=True))
    if len(train_2_df) < maxNum:
        train_df = train_df.append(train_2_df.sample(n=maxNum-len(train_2_df), replace=True))
    print('Now we have {} training samples.'.format(train_df.shape[0]))

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
