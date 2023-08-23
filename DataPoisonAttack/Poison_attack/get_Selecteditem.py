import random
import tensorflow as tf # 2.3版本
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from collections import defaultdict
from collections import Counter
from re import compile,findall,split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


with open("data/aktrain0.04.txt") as f:
    ratings = f.readlines()

trainingData = defaultdict(dict)
trainingSet_i = defaultdict(dict)
# threshold = 3
# targets_number = 20

for lineNo, line in enumerate(ratings):
    items = split(' |,|\t', line.strip())
    userId = items[0]
    itemId = items[1]
    rating = items[2]
    trainingData[userId][itemId] = float(rating)

dataset = [[0] * 5825 for i in range(339)]
for i,user in enumerate(trainingData):
    for item in trainingData[user]:
        trainingSet_i[item][user] = trainingData[user][item]
        dataset[int(user)][int(item)] = trainingData[user][item]

# Itemdict = {x:{} for x in range(5825)}

Itemcount = [0 for i in range(5825)]
Itemkey = [i for i in range(5825)]
# print(len(dataset))
for user in range(len(dataset)):
    for item in range(len(dataset[user])):
        if dataset[user][item] != 0:
            Itemcount[item] += 1

Itemdict = dict(zip(Itemkey,Itemcount))
sortedItemdict = sorted(Itemdict.items(), key=lambda x:x[1], reverse=True)
print(sortedItemdict)
sortedItemlist = []
for i in range(len(sortedItemdict)):
    a = sortedItemdict[i]
    sortedItemlist.append(a[0])
print(sortedItemlist)
selectItemlist = []
for i in range(int(0.04*len(sortedItemlist))):
    selectItemlist.append(sortedItemlist[i])
print(selectItemlist)
print(len(selectItemlist))

print(Itemcount)
print(len(Itemcount))


count_Itemcount_X = []
count_Itemcount_Y = []
count_Itemcount = Counter(Itemcount)
print(count_Itemcount)
count_Itemcount = dict(count_Itemcount)
count_Itemcount = sorted(count_Itemcount.items(), key=lambda x: x[0], reverse=False)



# for k,v in count_Itemcount.items():
#     count_Itemcount_X.append(k)
#     count_Itemcount_Y.append(v)
for i in range(len(count_Itemcount)):
    a = count_Itemcount[i]
    count_Itemcount_X.append(a[0])
    count_Itemcount_Y.append(a[1])
print(count_Itemcount_X)
print(count_Itemcount_Y)
plt.bar(count_Itemcount_X, count_Itemcount_Y, width = 0.8,color="#87CEFA")
plt.show()

# user_mean = []
# user_std = []
# dataset_normalization = []
# for i in range(len(dataset)):
#     user_mean[i] = np.mean(dataset[i])
#     user_std[i] = np.std(dataset[i],ddof=1)
#     dataset_normalization.append([])
#     for j in range(len(dataset[i])):
#         dataset_normalization[i].append((dataset[i][j]-user_mean[i])/user_std[i])
#
# print()



