'''
This scrips aims to determine the best ensemble strategy when giving several single model results which look like this:
sub.csv:
    id,label
    00005a3efe934a19adc0b69b05faeae7,0
sub_probs.csv:
    id,0,1,2
    00005a3efe934a19adc0b69b05faeae7,0.9073851039999999,0.09224746980000001,0.000367439931

At the same time, the ground truth file is given like this: id,title,content,label

'''

import tensorflow as tf
import pandas as pd
import os

ground_truth_dir = './data/train.csv'
train_df = pd.read_csv(ground_truth_dir)
train_df_label = train_df['label']

sub_probs_parent_dir = './combine_data_windows'
sub_results_list = []
for score in [8132, 8146, 8161]:
    sub_probs_dir = sub_probs_parent_dir+'/{}_probs.csv'.format(score)
    sub_results_list.append(pd.read_csv(sub_probs_dir))
