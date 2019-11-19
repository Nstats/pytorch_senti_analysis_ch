import pandas as pd
import Levenshtein as L


def most_similiar(target_str, str_list):
    '''
    find the most similar str with target str among str_list.
    :param target_str: a str
    :param str_list: [str_1, str_2,...,str_n]
    :return: one str in str_list
    '''
    L_distance = 1e10
    most_similiar_index = 0
    for i, str_i in enumerate(str_list):
        tmp = L.distance(target_str, str_i[0])
        if tmp < L_distance:
            L_distance = tmp
            most_similiar_index = i
    return str_list[most_similiar_index][0]


if __name__ == '__main__':
    data_dir = 'C:\\Users\\17801\\OneDrive\\桌面\\ocr\\IDcardRectify\\chinaaddress2018grade4.csv'
    df_values = pd.DataFrame(pd.read_csv(data_dir)).values
    print(most_similiar('广东省广市良口镇', df_values))
