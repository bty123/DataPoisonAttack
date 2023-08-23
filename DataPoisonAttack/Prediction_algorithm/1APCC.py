import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import time
start = time.time()

features=['uid','sid']
attacksize = 0.0
#读取攻击文件
data_dir_attack= "data/wsdata/paper/RT_Attack/"
adata=pd.read_csv(data_dir_attack+'RT_0.04_bandwagon_0.1_20.txt',sep="\t",names=['uid','sid','rt'])
features_sizes=[adata[f].nunique() for f in features]#每个特征不同值的统计个数
print(features_sizes)
print("攻击文件的行数：",adata.shape)
print(adata.dtypes)

#读取正常文件
data_dir="data/wsdata/RT/"
data=pd.read_csv(data_dir+'RT.base',sep='\t',names=['uid','sid','rt'])
print(data.dtypes)

print("分割前训练集行数：   ", data.shape)

# ----------密度------
density = 0.04
# ----------密度--------
X_train, X_test, y_train, y_test = train_test_split(data[features], data['rt'], train_size=density, random_state=99)

print("训练集行数：  ",X_train.shape)
print("分割后训练集格式：   ",type(X_train))
# X_train = X_train.append(adata)
# X_train = pd.concat([X_train,adata[['uid','sid']]],ignore_index=True)
print("注入攻击后的训练集X行数：  ",X_train.shape)
print("注入攻击前y行数",y_train.shape)
# y_train = y_train.append(adata['rt'])
# y_train = pd.concat([y_train,adata['rt']],ignore_index=True)
print("注入攻击后的训练集y行数：  ",y_train.shape)
print(X_train)
print(y_train)

train = []
for i in range(5825):
    train.append([])
    for j in range(int(339*(1.0 + attacksize))):
        train[i].append(0)
# print(train)

for i in range(len(X_train)):
    print(i)
    train[X_train.iloc[i]['sid']][X_train.iloc[i]['uid']] = y_train.iloc[i]
# print(train)

pcc = []
for i in range(len(train)):
    pcc.append([])
    print(i)
    for j in range(len(train)):
        if j == i:
            pcc[i].append(0)
            continue
        pcc[i].append(abs(pearsonr(train[i], train[j])[0]))
print(pcc)

sortedUserlist = []
pcc_dic = {}
temp = []
item = [i for i in range(5825)]
for i in range(len(pcc)):
    print(i)
    pcc_dic = dict(zip(item,pcc[i]))
    sortedUserdict = sorted(pcc_dic.items(), key=lambda x: x[1], reverse=True)
    for j in range(len(sortedUserdict)):
        a = sortedUserdict[j]
        sortedUserlist.append(a[0])

    for user in range(len(train[i])):
        # print(user)
        if train[i][user] == 0:
            for k in range(20):
                if train[sortedUserlist[k]][user] != 0:
                    temp.append(train[sortedUserlist[k]][user])
            if len(temp) == 0:
                train[i][user] = 0.5
                continue
            train[i][user] = np.mean(temp)
        del temp[:]
    del sortedUserlist[:]

print(train)


y_pred = []
for i in range(len(X_test)):
    print(i)
    y_pred.append(train[X_test.iloc[i]['sid']][X_test.iloc[i]['uid']])

mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))

print('MAE:',mae)
print('RMSE:',rmse)

end = time.time()
print("程序的运行时间为：{}".format(end-start))
