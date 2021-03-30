import sys
import os

import numpy as np
import pandas as pd
import sys
import os
from PIL import Image

def is_anagram(string_a,string_b):
    """returns True if the strings are anagrams of each other
    str, list -> boolean"""
    if len(string_a) != len(string_b):
        return False
    for i in range(len(string_a)):
        if string_a[i] != string_b[i]:
            return False
    return True

def data_label(data):
    y = data.columns.values.tolist()
    # print(y)

    name = y.copy()
    name = np.array(name)
    name = name[1:]
    # print(name)
    name_sort = np.sort(name)
    name_argsort = np.argsort(name)
    # print(name_sort)
    for i in range(len(name_sort)):
        name_sort[i] = name_sort[i][3:]
    # print(name_sort)
    for i in range(len(name_sort)):
        for j in range(len(name_sort[i])):
            if name_sort[i][j] == '_':
                name_sort[i] = name_sort[i][:j]
                break
    # print(name_sort)
    label = np.zeros(len(name_sort))
    label[0] = 0
    for i in range(1, len(name_sort)):
        if is_anagram(name_sort[i], name_sort[i - 1]):
            label[i] = label[i - 1]
        else:
            label[i] = label[i - 1] + 1

    data_np = np.array(data)
    data_np_T = data_np.T
    data_np_T = data_np_T[:][1:]
    data_np_T = data_np_T[name_argsort, :]
    # print(data_np_T.shape)
    return data_np_T,label
def cellstandardization(data): # 使每个细胞的基因表达总量都相等
    sum = data.sum(axis=1)  # 每个细胞的基因表达总量
    # print(sum)
    #print(sum.shape)
    data0 = data[0]
    median = np.median(sum)
    # print(median)
    sum = sum / median
    # print(sum)
    for i in range(len(data)):
        for j in range(len(data.T)):
            data[i][j] = data[i][j] / sum[i]
    return data

def data_10000(data):
    # 计算每一列的方差
    deviation = np.var(data, axis=0)
    # print(deviation.shape)
    # print(deviation)
    # 前21807个方差不为0
    deviation_sort = np.sort(-deviation)
    # print(deviation_sort)
    # 方差从大到小排序
    deviation_argsort = np.argsort(-deviation)
    print(deviation_argsort)

    data_sort = data[:, deviation_argsort]
    data_10000 = data_sort[:, :10000]
    return data_10000
def genecorrelation(data):
    # 将相关性较大的基因分组
    data_10000 = data.astype(float)
    data_corr = np.corrcoef(data_10000.T)
    data_corr = data_corr * 0.5 + 0.5
    from sklearn.cluster import SpectralClustering
    SC = SpectralClustering(affinity='precomputed', assign_labels='discretize', random_state=100)
    label = SC.fit_predict(data_corr)
    label_argsort = np.argsort(label)
    data_10000 = data_10000[:, label_argsort]
    return data_10000


data = pd.read_csv('pollen.csv')
data, label = data_label(data)

print(data)
data = cellstandardization(data)
print(data)
data_10000 = data_10000(data)
print(data_10000.shape)
data_10000 = genecorrelation(data_10000)
print(data_10000)








name = label

#ToImg
for i in range(len(name)):
    if i==0:
            os.makedirs('data./' + str(int(name[i])))
    elif int(name[i]) > int(name[i-1]):
        os.makedirs('data./' + str(int(name[i])))
for i in range(len(data_10000)):

    #os.makedirs('D:\Siamese-pytorch-master - 副本\experment\data./' + name[i])
    path = 'data./' + str(int(name[i]))
    cell = data_10000[i].copy()
    cell = cell.reshape(100,100)
    im = Image.fromarray(cell)
    im.convert('L').save(path+'/'+str(i)+'.jpg', format='jpeg')