import pandas as pd
import os
import re
import random
import jieba
from langconv import *


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

def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence


def clean_data(line):
    line = line.replace('\n', '').replace('\\n', '').replace(' ', '')
    text_list = re.findall("[^\u4E00-\u9FFF]{3,}", line)
    for item in text_list:
        test = re.sub("[0-9.]*", "", item)
        if test and len(test) > 2:
            line = line.replace(item, " ")
    simplified_sentence = Traditional2Simplified(line)

    # 匹配中文字符、中英文标点、英文字符
    re_keywords1 = re.compile('[^\u0020-\u007f\u4E00-\u9FA5\u2000-\u206f\u3000-\u303f\uff00-\uffef]+')
    newContext = re.sub(re_keywords1, ' ', simplified_sentence)

    # 删除“想爆料？”后面语句
    re_keywords4 = re.compile('(想爆料？)[\S\s]+')
    newContext = re.sub(re_keywords4, ' ', newContext)
    re_keywords5 = re.compile('[?]{2,1000}')
    newContext = re.sub(re_keywords5, ' ', newContext)

    # 删除“微信公众号：”后面语句
    re_keywords6 = re.compile('(遵义生活网微信公众号：)[\S\s]+杨某某，')
    newContext = re.sub(re_keywords6, '杨某某，', newContext)
    re_keywords7 = re.compile('(上海大学材料学院研究生微信公众号)')
    newContext = re.sub(re_keywords7, '上海大学材料学院研究生公众号', newContext)
    re_keywords8 = re.compile('微信公众号[\S\s]+！2月11日，')
    newContext = re.sub(re_keywords8, '2月11日，', newContext)
    re_keywords8 = re.compile('微信公众号：西安老板群')
    newContext = re.sub(re_keywords8, ' ', newContext)
    re_keywords8 = re.compile('微信公众号：平安江干')
    newContext = re.sub(re_keywords8, '公众号：平安江干', newContext)
    re_keywords9 = re.compile('微信公众号：[\S\s]+')
    newContext = re.sub(re_keywords9, '   ', newContext)

    # 匹配长串的无意义标签
    re_keywords3 = re.compile('[>]?[a-z]+[=\'\S\s]+blank">')
    newContext = re.sub(re_keywords3, ' ', newContext)
    re_keywords3 = re.compile('=0&&[\S\s]+none"">')
    newContext = re.sub(re_keywords3, ' ', newContext)
    re_keywords3 = re.compile('style=""[\S\s]+1.5em;')
    newContext = re.sub(re_keywords3, ' ', newContext)
    # 删除<!--xxxx-->的内容
    re_keywords3 = re.compile('<!--[\S\s]+-->')
    newContext = re.sub(re_keywords3, ' ', newContext)

    # 删除“责任编辑: ”后面语句

    re_keywords6 = re.compile('(\[责任编辑:王丽媛\]20)')
    newContext = re.sub(re_keywords6, ' ', newContext)

    re_keywords7 = re.compile('\(责任编辑:df333\)}，')
    newContext = re.sub(re_keywords7, ' ', newContext)
    re_keywords8 = re.compile('责任编辑[\S\s]+播放数:年，')
    newContext = re.sub(re_keywords8, ' ', newContext)
    re_keywords7 = re.compile('(责任编辑:张迪)')
    newContext = re.sub(re_keywords7, ' ', newContext)
    re_keywords6 = re.compile('(责任编辑: 汽车抵押贷款)')
    newContext = re.sub(re_keywords6, '汽车抵押贷款', newContext)

    re_keywords8 = re.compile('责任编辑往往是业内令人尊敬的大牛')
    newContext = re.sub(re_keywords8, '编辑往往是业内令人尊敬的大牛', newContext)

    re_keywords10 = re.compile('责任编辑[\S\s]+')
    newContext = re.sub(re_keywords10, ' ', newContext)

    re_keywords2 = re.compile('<chsdate[\S\s]+""2015""')
    newContext = re.sub(re_keywords2, ' ', newContext)

    re_keywords2 = re.compile('--[/diy]-->[\S\s]+important;"">"')
    newContext = re.sub(re_keywords2, ' ', newContext)

    re_keywords10 = re.compile('[? ]{2,}|</chsdate>')
    newContext = re.sub(re_keywords10, ' ', newContext)

    # 1、network addr by mijie
    re_netAddr = re.compile('(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%?=~_|!:,.;]+')
    newContext = re.sub(re_netAddr, ' ', newContext)
    # 2、email by mijie
    re_mail = re.compile('[a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+)')
    newContext = re.sub(re_mail, ' ', newContext)
    return newContext


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

    train_0_df = train_df[train_df['label'] == 0]
    train_1_df = train_df[train_df['label'] == 1]
    train_2_df = train_df[train_df['label'] == 2]

    cilin_dir = './data/cilin.txt'
    data_dict = {}
    with open('./data/cilin.txt', encoding='utf-8') as f:
        for line in f.readlines():
            line_split = line.split(' ')
            if len(line_split) > 2:
                if len(line_split[1]) > 1 and len(line_split[2]) > 1:
                    data_dict.update({line_split[1]: line_split[2].replace('\n', '')})
                    data_dict.update({line_split[2].replace('\n', ''): line_split[1]})
    # print(len(data_dict), data_dict)

    maxNum = max(len(train_0_df), len(train_1_df), len(train_2_df))

    if len(train_0_df) < maxNum:
        add_df = train_0_df.sample(n=maxNum - len(train_0_df), replace=True)
        add_df_ = replace(add_df, data_dict)
        train_df = train_df.append(add_df_)
    if len(train_1_df) < maxNum:
        add_df = train_1_df.sample(n=maxNum - len(train_1_df), replace=True)
        add_df_ = replace(add_df, data_dict)
        train_df = train_df.append(add_df_)
    if len(train_2_df) < maxNum:
        add_df = train_2_df.sample(n=maxNum - len(train_2_df), replace=True)
        add_df_ = replace(add_df, data_dict)
        train_df = train_df.append(add_df_)
    print('Now we have {} training samples.'.format(train_df.shape[0]))

    train_new_line = []
    for line in train_df['content']:
        train_new_line.append(clean_data(line))
    train_df['content'] = train_new_line

    test_new_line = []
    for line in test_df['content']:
        test_new_line.append(clean_data(line))
    test_df['content'] = test_new_line

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
