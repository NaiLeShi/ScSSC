import os
import torch
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import community

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

pred = pd.read_csv('pred.csv')
pred = np.array(pred)
#print(pred)

# for i in range(len(pred.T)):
#     if pred[-2,i] > 0.7:
#         print(i,pred[-2,i])

cell = pred[:-2,0]
#print(cell)
cell_sort = np.sort(cell)
cell_argsort = np.argsort(cell)

#print(cell_sort)
#print(cell_argsort)
pred = pred[:-2,:]
pred = pred[cell_argsort,:]

#pd.DataFrame(pred).to_csv("pred_sort.csv", index=False, sep=',')

# 对排序好的数据进行计算

y = pred
result = torch.zeros(len(y),len(y)).numpy()

for i in range(40,65):
    a = y.T[i]
    for j in range(len(y)):
        for k in range(j + 1, len(y)):
            if a[j] == a[k]:
                result[j][k] += 1
print(result.shape)
#pd.DataFrame(result).to_csv("result40-65.csv", index=False, sep=',')


#data = pd.read_csv('result40-65.csv')
#y = np.array(data)
#print(y.shape)

y = result
G = nx.Graph()
node = []
edge = []
for i in range(len(y)):
    node.append(str(i))
for i in range(len(y)):
    for j in range(i,len(y)):
        if y[i][j] >= 15 :
            edge.append((str(i),str(j),int(y[i][j])))

G.add_nodes_from(node)
G.add_weighted_edges_from(edge)
partition = community.best_partition(G)
value = np.array(list(partition.values()))
print(value)
#drawing
size = float(len(set(partition.values())))
pos = nx.spring_layout(G)
count = 0.
for com in set(partition.values()) :
    count = count + 1.
    list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
    nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                node_color = str(count / size))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()


for i in range(len(y)):
    for j in range(i):
        if i != j :
            y[i][j] = y[j][i]
for i in range(len(y)):
    for j in range(len(y)):
        if value[i] != value[j]:
            y[i][j] = 0

du = []
for i in range(len(y)):
    sum = 0
    num = 0
    for j in range(len(y)):
        if y[j][i] > 0:
            sum += y[i][j]
            num += 1
    if num == 0:
        du.append(round(0, 2))
    else:
        du.append(round(float(sum)/num,2))
print(du)

threshold = np.sort(du)
threshold = threshold[int(len(du)*0.015)]


for i in range(len(du)):
    if du[i] <= threshold:
        value[i] = -1
        print(du[i])
print(value)

pd.DataFrame(value).to_csv("value.csv", index=False, sep=',')



