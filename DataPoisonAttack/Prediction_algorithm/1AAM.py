import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import time
start = time.time()
np.random.seed(99)
features=['uid','sid']

#读取攻击文件
targetfilename = 'invoked_RT_target_item_20.txt'
data_dir_attack = "data/wsdata/paper/RT_Attack/"
adata = pd.read_csv(data_dir_attack+'A_RT_0.04_diffusion_0.1_20.txt', sep="\t", names=['uid','sid','rt'])
features_sizes = [adata[f].nunique() for f in features] # 每个特征不同值的统计个数
print(features_sizes)
print("攻击文件的行数：",adata.shape)
print(adata.dtypes)

#读取正常文件
data_dir = "data/wsdata/RT/"
data = pd.read_csv(data_dir+'RT.base', sep='\t', names=['uid','sid','rt'])
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

average_item = []
# for i in range(len(X_train)):
average = []
for i in range(5825):
    average.append([])
for j in range(len(X_train)):
    if j % 10000 == 0:
        print(j)
    average[X_train.iloc[j]['sid']].append(y_train.iloc[j])

# print(average)
for i in range(len(average)):
    if len(average[i]) == 0:
        average_item.append(0.5)
        continue
    average_item.append(sum(average[i])/len(np.nonzero(average[i])[0]))
print(len(average_item))
print(average_item)

# 提取目标项目
target_item = []
target_count = 20
with open(data_dir_attack + targetfilename, 'r') as f:
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
    # if X_test.iloc[i]["sid"] in target_item:
    #     X_test_target.append(np.array(X_test.iloc[i]))
    #     y_test_target.append(y_test.iloc[i])
    # # 提取随机项目评分
    # if X_test.iloc[i]["sid"] in random_item:
    #     X_test_random.append(np.array(X_test.iloc[i]))
    #     y_test_random.append(y_test.iloc[i])
    # 提取整体项目评分
    y_pred.append(average_item[X_test.iloc[i]['sid']])

# X_test_target = np.array(X_test_target)
# y_test_target = np.array(y_test_target)
# X_test_random = np.array(X_test_random)
# y_test_random = np.array(y_test_random)
# # y_test_target = pd.DataFrame(y_test_target)
# print('len(y_test_target):', len(y_test_target))
# print('len(y_test_random):', len(y_test_random))
# print(X_test_target)

mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))

# 计算随机项目的评价指标
# y_pred_random = []
# for i in range(len(y_test_random)):
#     y_pred_random.append(average_item[X_test_random[i][1]])
# mae_random = mean_absolute_error(y_test_random, y_pred_random)
# rmse_random = math.sqrt(mean_squared_error(y_test_random, y_pred_random))
#
# # 计算目标项目的评价指标
# y_pred_target = []
# for i in range(len(y_test_target)):
#     y_pred_target.append(average_item[X_test_target[i][1]])
# mae_target = mean_absolute_error(y_test_target, y_pred_target)
# rmse_target = math.sqrt(mean_squared_error(y_test_target, y_pred_target))

print('MAE:',format(mae, '.4f'))
# print('MAE_random:',format(mae_random, '.4f'))
# print('MAE_target:',format(mae_target, '.4f'))
print('RMSE:',format(rmse, '.4f'))
# print('RMSE_random:',format(rmse_random, '.4f'))
# print('RMSE_target:',format(rmse_target, '.4f'))

# Top_K评价指标
K = 1000
sortedtargetlist = []
hit_user_count = [0 for i in range(K)]
pre_dataset = [[0] * 5825 for i in range(339)]
api = [i for i in range(5825)]
for i in range(len(X_test)):
    if i % 10000 == 0:
        print(i)
    pre_dataset[X_test.iloc[i]["uid"]][X_test.iloc[i]["sid"]] = y_pred[i]

# print(pre_dataset)
for i in range(len(pre_dataset)):
    pre_dataset_dic = dict(zip(api, pre_dataset[i]))
    sortedtargetdict = sorted(pre_dataset_dic.items(), key=lambda x: x[1], reverse=True)
    for j in range(len(sortedtargetdict)):
        a = sortedtargetdict[j]
        sortedtargetlist.append(a[0])
    print(sortedtargetlist)
    for j in range(target_count):
        if target_item[j] in sortedtargetlist[:K]:
            hit_user_count[i] += 1
    del sortedtargetlist[:]
print('hit_user_count:',hit_user_count)
print('len(hit_user_count):',len(hit_user_count))
Top_k = sum(hit_user_count)/(339*target_count)
print('Top_k:',Top_k)





end = time.time()
print("程序的运行时间为：{}".format(end-start))