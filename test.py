import pandas as pd

data_dir = './data/Second_TestDataSet.csv'

df = pd.DataFrame(pd.read_csv(data_dir))
df_id = df['id']
df_content = df['content']
size = df.shape[0]
for i in range(size):
    if not len(df_id[i]) == 32:
        print(i, df_id[i])
