import pandas as pd
import os
import re
import random
import jieba


def replace(df, replace_dict):
    '''
    input:
        df: a pandas dataframe with columns ['id','title','content','label']
        replace_dict: a dict with similar word pairs shaped like {word1: word2}
    return:
        a pandas dataframe with same shape as df but may have title & content word replaced use replace_dict.
    '''
    df_title = df['title']
    df_content = df['content']

    tmp = []
    for line in df_title:
        line_cut = jieba.lcut(line)
        for i in range(len(line_cut)):
            if line_cut[i] in replace_dict:
                line_cut[i] = replace_dict.get(line_cut[i])
        tmp.append(''.join(line_cut))
    df['title'] = tmp

    tmp = []
    for line in df_content:
        line_cut = jieba.lcut(line)
        for i in range(len(line_cut)):
            if line_cut[i] in replace_dict:
                line_cut[i] = replace_dict.get(line_cut[i])
        tmp.append(''.join(line_cut))
    df['content'] = tmp

    add_df_ = df
    return add_df_


def balanced_data_v2(df):
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
        add_df = df_0.sample(n=maxNum - len(df_0), replace=True)
        df = df.append(replace(add_df, replace_dict=data_dict))
    if len(df_1) < maxNum:
        add_df = df_1.sample(n=maxNum - len(df_1), replace=True)
        df = df.append(replace(add_df, replace_dict=data_dict))
    if len(df_2) < maxNum:
        add_df = df_2.sample(n=maxNum - len(df_2), replace=True)
        df = df.append(replace(add_df, replace_dict=data_dict))
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

    cilin_dir = './data/cilin.txt'
    data_dict = {}
    with open('./data/cilin.txt', encoding='utf-8') as f:
        for line in f.readlines():
            line_split = line.split(' ')
            if len(line_split) > 2:
                if len(line_split[1]) > 1 and len(line_split[2]) > 1:
                    data_dict.update({line_split[1]: line_split[2].replace('\n', '')})
    # print(len(data_dict), data_dict)

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
        train_df_balanced = balanced_data_v2(train_df.iloc[train_index])
        train_df_balanced.to_csv("./data/data_{}/train.csv".format(i))
        train_df.iloc[dev_index].to_csv("./data/data_{}/dev.csv".format(i))
        test_df.to_csv("./data/data_{}/test.csv".format(i))
