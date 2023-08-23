import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import time
start = time.time()
np.random.seed(99)
features=['uid','sid']
attacksize = 0.1
#读取攻击文件
data_dir_attack= "data/wsdata/paper/RT_Attack/"
adata=pd.read_csv(data_dir_attack+'RT_0.04_average_0.1_20.txt',sep="\t",names=['uid','sid','rt'])
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
X_train = pd.concat([X_train,adata[['uid','sid']]],ignore_index=True)
print("注入攻击后的训练集X行数：  ",X_train.shape)
print("注入攻击前y行数",y_train.shape)
# y_train = y_train.append(adata['rt'])
y_train = pd.concat([y_train,adata['rt']],ignore_index=True)
print("注入攻击后的训练集y行数：  ",y_train.shape)
print(X_train)
print(y_train)

train = []
for i in range(int(339*(1.0 + attacksize))):
    train.append([])
    for j in range(5825):
        train[i].append(0)
# print(train)

for i in range(len(X_train)):
    if i % 10000 == 0:
        print(i)
    train[X_train.iloc[i]['uid']][X_train.iloc[i]['sid']] = y_train.iloc[i]
# print(train)

count_user = [0 for i in range(len(train))]
for i in range(len(train)):
    for j in range(len(train[i])):
        if train[i][j] != 0:
            count_user[i] += 1

BRASim = []
# mutual = []
for i in range(len(train)):
    # mutual.append([])
    BRASim.append([])
    if i % 10 == 0:
        print(i)
    for j in range(len(train)):
        if j == i:
            BRASim[i].append(0)
            continue
        count = 0
        sum = 0
        for k in range(len(train[i])):
            if train[i][k] != 0 and train[j][k] != 0:
                if train[i][k] > train[j][k]:
                    v = train[j][k] / train[i][k]
                else:
                    v = train[i][k] / train[j][k]
                sum += v
                count += 1
        # mutual[i].append(count)
        BRASim[i].append(sum/(count_user[i] + count_user[j] - count))
print(len(BRASim))
print(len(BRASim[0]))

sortedUserlist = []
pcc_dic = {}
temp = []
user = [i for i in range(int(339*(1.0 + attacksize)))]
for i in range(len(BRASim)):
    if i % 10 == 0:
        print(i)
    pcc_dic = dict(zip(user,BRASim[i]))
    sortedUserdict = sorted(pcc_dic.items(), key=lambda x: x[1], reverse=True)
    for j in range(len(sortedUserdict)):
        a = sortedUserdict[j]
        sortedUserlist.append(a[0])

    for item in range(len(train[i])):
        # print(item)
        if train[i][item] == 0:
            for k in range(20):
                if train[sortedUserlist[k]][item] != 0:
                    temp.append(train[sortedUserlist[k]][item])
            if len(temp) == 0:
                train[i][item] = 0.5
                continue
            train[i][item] = np.mean(temp)
        del temp[:]
    del sortedUserlist[:]

# print(train)

# 提取目标项目
target_item = []
with open(data_dir_attack + 'target_item_20.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        target_item.append(int(line))

print('target_item:', target_item)

# 提起与目标项目数量一致的随机项目
random_item = np.random.randint(0,5825,size=20)
random_item = list(random_item)
print('random_item:', random_item)

# 提取随机项目和目标项目的y_test
y_test_target = []
y_test_random = []
X_test_target = []
X_test_random = []

y_pred = []
for i in range(len(X_test)):
    if i % 10000 == 0:
        print(i)
    # 提取目标项目评分
    if X_test.iloc[i]["sid"] in target_item:
        X_test_target.append(np.array(X_test.iloc[i]))
        y_test_target.append(y_test.iloc[i])
    # 提取随机项目评分
    if X_test.iloc[i]["sid"] in random_item:
        X_test_random.append(np.array(X_test.iloc[i]))
        y_test_random.append(y_test.iloc[i])
    y_pred.append(train[X_test.iloc[i]['uid']][X_test.iloc[i]['sid']])
# print(y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
print('MAE:',mae)
print('RMSE:',rmse)

X_test_target = np.array(X_test_target)
y_test_target = np.array(y_test_target)
X_test_random = np.array(X_test_random)
y_test_random = np.array(y_test_random)
# y_test_target = pd.DataFrame(y_test_target)
print('len(y_test_target):', len(y_test_target))
print('len(y_test_random):', len(y_test_random))
# print(X_test_target)

# 计算随机项目的评价指标
y_pred_random = []
for i in range(len(y_test_random)):
    y_pred_random.append(train[X_test_random[i][0]][X_test_random[i][1]])
mae_random = mean_absolute_error(y_test_random, y_pred_random)
rmse_random = math.sqrt(mean_squared_error(y_test_random, y_pred_random))
print('MAE_random:',mae_random)
print('RMSE_random:',rmse_random)
# 计算目标项目的评价指标
y_pred_target = []
for i in range(len(y_test_target)):
    y_pred_target.append(train[X_test_target[i][0]][X_test_target[i][1]])
mae_target = mean_absolute_error(y_test_target, y_pred_target)
rmse_target = math.sqrt(mean_squared_error(y_test_target, y_pred_target))
print('MAE_target:',mae_target)
print('RMSE_target:',rmse_target)

end = time.time()
print("程序的运行时间为：{}".format(end-start))

