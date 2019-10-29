'''
This scrips aims to determine the best ensemble strategy when giving several single model results which look like this:
sub.csv:
    id,label
    00005a3efe934a19adc0b69b05faeae7,0
sub_probs.csv:
    id,0,1,2
    00005a3efe934a19adc0b69b05faeae7,0.9073851039999999,0.09224746980000001,0.000367439931

At the same time, the ground truth file is given: id,title,content,label

'''

import tensorflow as tf
