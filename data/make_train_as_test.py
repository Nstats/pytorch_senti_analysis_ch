import pandas as pd
import os
import random

if __name__ == '__main__':
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

    train_df = pd.concat([train_df_v1, train_df_v2])
    train_df.to_csv('./data/whole_train_set.csv', index=False)
    for i in range(3):
        print("Fold", i)
        if os.path.exists('./data/data_{}'.format(i)):
            os.system("rm -rf ./data/data_{}".format(i))
        os.system("mkdir ./data/data_{}".format(i))
        train_df[['id', 'title', 'content']].to_csv("./data/data_{}/test.csv".format(i), index=False)
