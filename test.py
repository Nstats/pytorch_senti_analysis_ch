import pandas as pd
import numpy as np

df = pd.DataFrame(np.arange(12).reshape(3, 4), index=['a', 'b', 'c'])
l = ['c', 'b', 'a']
df['st'] = df.index
print(df, '\n')
df['st'] = df['st'].astype('category')
df['st'].cat.reorder_categories(l, inplace=True)
print(df, '\n')
df.sort_values('st', inplace=True)
print(df)
