import pandas as pd

train_dir = './data/Train_DataSet.csv'
test_dir = './data/Test_DataSet.csv'

target_dir = './data/data_for_lmfinetune.txt'

train_df = pd.DataFrame(pd.read_csv(train_dir))
train_content = train_df['content'].fillna(' ')
test_df = pd.DataFrame(pd.read_csv(test_dir))
test_content = test_df['content'].fillna(' ')

with open(target_dir, 'a') as fin:
    for ele in train_content:
        fin.write(str(ele)+'\n')
    for ele in test_content:
        fin.write(str(ele)+'\n')
