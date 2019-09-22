import pandas as pd
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_prefix", default='./output_RoBERTa_large_10epo_3split/fold_', type=str, required=True)
args = parser.parse_args()

k=5
df=pd.read_csv('data/submit_example.csv')
df['0']=0
df['1']=0
df['2']=0
for i in range(k):
    temp=pd.read_csv('{}{}/sub.csv'.format(args.model_prefix, i))
    df['0']+=temp['label_0']/k
    df['1']+=temp['label_1']/k
    df['2']+=temp['label_2']/k

df[['id', '0', '1', '2']].to_csv('./sub_probs.csv', index=False)
df['label'] = np.argmax(df[['0', '1', '2']].values, -1)
df[['id', 'label']].to_csv('./sub.csv', index=False)
