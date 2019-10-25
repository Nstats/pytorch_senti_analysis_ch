import Levenshtein
import pandas as pd

train_df = pd.read_csv("./data/train.csv")
train_df['label'] = train_df['label'].fillna(-1)
train_df = train_df[train_df['label'] != -1]
train_df['label'] = train_df['label'].astype(int)

train_df['content'] = train_df['content'].fillna('无')
train_df['title'] = train_df['title'].fillna('无')

train_0_df = train_df[train_df['label'] == 0]
train_1_df = train_df[train_df['label'] == 1]
train_2_df = train_df[train_df['label'] == 2]
print(str(train_0_df.shape[0]))
print(str(train_1_df.shape[0]))
print(str(train_2_df.shape[0]))
fo = open("./data/wrong_data_id.txt", "w+")
for i in range(train_0_df.shape[0]):
    print(str(i))
    for j in range(train_1_df.shape[0]):
        len = 0
        len = len+Levenshtein.distance(train_0_df.iloc[[i],[-2]].values[0][0],train_1_df.iloc[[j], [-2]].values[0][0])
        len = len+Levenshtein.distance(train_0_df.iloc[[i],[-3]].values[0][0],train_1_df.iloc[[j], [-3]].values[0][0])
        if len<0.05*max(train_0_df.iloc[[i],[-2]].values[0][0].__len__()+train_0_df.iloc[[i],[-3]].values[0][0].__len__(),
                   train_1_df.iloc[[j],[-2]].values[0][0].__len__()+train_1_df.iloc[[j],[-3]].values[0][0].__len__()):
            fo.write(str(train_0_df.iloc[[i],[1]].values[0][0])+','+str(train_1_df.iloc[[j],[1]].values[0][0])+'\n')
    for j in range(train_2_df.shape[0]):
        len = 0
        len = len + Levenshtein.distance(train_0_df.iloc[[i],[-2]].values[0][0],train_2_df.iloc[[j], [-2]].values[0][0])
        len = len + Levenshtein.distance(train_0_df.iloc[[i],[-3]].values[0][0],train_2_df.iloc[[j], [-3]].values[0][0])
        if len < 0.05*max(train_0_df.iloc[[i],[-2]].values[0][0].__len__()+train_0_df.iloc[[i],[-3]].values[0][0].__len__(),
                   train_2_df.iloc[[j],[-2]].values[0][0].__len__()+train_2_df.iloc[[j],[-3]].values[0][0].__len__()):
            fo.write(str(train_0_df.iloc[[i], [1]].values[0][0])+','+str(train_2_df.iloc[[j], [1]].values[0][0])+'\n')

for i in range(train_1_df.shape[0]):
    print(str(i))
    for j in range(train_2_df.shape[0]):
        len = 0
        len = len + Levenshtein.distance(train_1_df.iloc[[i], [-2]].values[0][0],train_2_df.iloc[[j], [-2]].values[0][0])
        len = len + Levenshtein.distance(train_1_df.iloc[[i], [-3]].values[0][0],train_2_df.iloc[[j], [-3]].values[0][0])
        if len < 0.05*max(train_1_df.iloc[[i], [-2]].values[0][0].__len__() + train_1_df.iloc[[i], [-3]].values[0][0].__len__(),
                     train_2_df.iloc[[j], [-2]].values[0][0].__len__() + train_2_df.iloc[[j], [-3]].values[0][0].__len__()):
            fo.write(str(train_1_df.iloc[[i], [1]].values[0][0])+','+str(train_2_df.iloc[[j], [1]].values[0][0])+'\n')
fo.close()