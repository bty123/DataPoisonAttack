import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import time
import random
import csv
random.seed(2022)

start = time.time()
np.random.seed(99)
features=['uid','sid']

attacksize = 0.1
#读取攻击文件
targetfilename = 'invoked_RT_target_item_20.txt'
data_dir_attack= "data/wsdata/paper/RT_Attack/"
adata=pd.read_csv(data_dir_attack+'A_RT_0.04_diffusion_0.1_20.txt',sep="\t",names=['uid','sid','rt'])
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


train_U = []
for i in range(int(339*(1.0 + attacksize))):
    train_U.append([])
    for j in range(5825):
        train_U[i].append(0)
# print(train)

for i in range(len(X_train)):
    if i % 10000 == 0:
        print(i)
    train_U[X_train.iloc[i]['uid']][X_train.iloc[i]['sid']] = y_train.iloc[i]
# print(train)

pcc_U = []
for i in range(len(train_U)):
    pcc_U.append([])
    if i % 10 == 0:
        print(i)
    for j in range(len(train_U)):
        if j == i:
            pcc_U[i].append(0)
            continue
        pcc_U[i].append(abs(pearsonr(train_U[i], train_U[j])[0]))
# print(pcc)

sortedUserlist_U = []
pcc_dic_U = {}
temp_U = []
user = [i for i in range(int(339*(1.0 + attacksize)))]
for i in range(len(pcc_U)):
    if i % 10 == 0:
        print(i)
    pcc_dic_U = dict(zip(user,pcc_U[i]))
    sortedUserdict = sorted(pcc_dic_U.items(), key=lambda x: x[1], reverse=True)
    for j in range(len(sortedUserdict)):
        a = sortedUserdict[j]
        sortedUserlist_U.append(a[0])

    for item in range(len(train_U[i])):
        # print(item)
        if train_U[i][item] == 0:
            count = 0
            for k in range(len(sortedUserlist_U)):
                if train_U[sortedUserlist_U[k]][item] != 0:
                    temp_U.append(train_U[sortedUserlist_U[k]][item])
                    count += 1
                    if count > 20:
                        break
            if len(temp_U) == 0:
                train_U[i][item] = 0.5
                continue
            train_U[i][item] = np.mean(temp_U)
        del temp_U[:]
    del sortedUserlist_U[:]

# print(train_U)


train_A = []
for i in range(5825):
    train_A.append([])
    for j in range(int(339*(1.0 + attacksize))):
        train_A[i].append(0)
# print(train)

for i in range(len(X_train)):
    if i % 10000 == 0:
        print(i)
    train_A[X_train.iloc[i]['sid']][X_train.iloc[i]['uid']] = y_train.iloc[i]
# print(train)

pcc_A = []
for i in range(len(train_A)):
    pcc_A.append([])
    if i % 10 == 0:
        print(i)
    for j in range(len(train_A)):
        if j == i:
            pcc_A[i].append(0)
            continue
        pcc_A[i].append(abs(pearsonr(train_A[i], train_A[j])[0]))
# print(pcc)

sortedUserlist_A = []
pcc_dic_A = {}
temp_A = []
item = [i for i in range(5825)]
for i in range(len(pcc_A)):
    if i % 10 == 0:
        print(i)
    pcc_dic_A = dict(zip(item,pcc_A[i]))
    sortedUserdict = sorted(pcc_dic_A.items(), key=lambda x: x[1], reverse=True)
    for j in range(len(sortedUserdict)):
        a = sortedUserdict[j]
        sortedUserlist_A.append(a[0])

    for user in range(len(train_A[i])):
        # print(item)
        if train_A[i][user] == 0:
            count = 0
            for k in range(len(sortedUserlist_A)):
                if train_A[sortedUserlist_A[k]][user] != 0:
                    temp_A.append(train_A[sortedUserlist_A[k]][user])
                    count += 1
                    if count > 20:
                        break
            if len(temp_A) == 0:
                train_A[i][user] = 0.5
                continue
            train_A[i][user] = np.mean(temp_A)
        del temp_A[:]
    del sortedUserlist_A[:]

# print(train_A)

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

y_pred_U = []
y_pred_A = []
y_pred_WSRec = []
for i in range(len(X_test)):
    if i % 10000 == 0:
        print(i)
    #
    # 提取目标项目评分
    # if X_test.iloc[i]["sid"] in target_item:
    #     X_test_target.append(np.array(X_test.iloc[i]))
    #     y_test_target.append(y_test.iloc[i])
    # # 提取随机项目评分
    # if X_test.iloc[i]["sid"] in random_item:
    #     X_test_random.append(np.array(X_test.iloc[i]))
    #     y_test_random.append(y_test.iloc[i])

    y_pred_U.append(train_U[X_test.iloc[i]['uid']][X_test.iloc[i]['sid']])
    y_pred_A.append(train_A[X_test.iloc[i]['sid']][X_test.iloc[i]['uid']])
    y_pred_WSRec.append(train_U[X_test.iloc[i]['uid']][X_test.iloc[i]['sid']] + train_A[X_test.iloc[i]['sid']][X_test.iloc[i]['uid']])

for i in range(len(y_pred_WSRec)):
    y_pred_WSRec[i] = y_pred_WSRec[i]*0.5

mae_U = mean_absolute_error(y_test, y_pred_U)
rmse_U = math.sqrt(mean_squared_error(y_test, y_pred_U))
mae_A = mean_absolute_error(y_test, y_pred_A)
rmse_A = math.sqrt(mean_squared_error(y_test, y_pred_A))
mae_WSRec = mean_absolute_error(y_test, y_pred_WSRec)
rmse_WSRec = math.sqrt(mean_squared_error(y_test, y_pred_WSRec))

# X_test_target = np.array(X_test_target)
# y_test_target = np.array(y_test_target)
# X_test_random = np.array(X_test_random)
# y_test_random = np.array(y_test_random)
# # y_test_target = pd.DataFrame(y_test_target)
# print('len(y_test_target):', len(y_test_target))
# print('len(y_test_random):', len(y_test_random))
# # print(X_test_target)

# 计算随机项目的评价指标
# y_pred_random_U = []
# y_pred_random_A = []
# y_pred_random_WSRec = []
# for i in range(len(y_test_random)):
#     y_pred_random_U.append(train_U[X_test_random[i][0]][X_test_random[i][1]])
#     y_pred_random_A.append(train_A[X_test_random[i][1]][X_test_random[i][0]])
#     y_pred_random_WSRec.append(train_U[X_test_random[i][0]][X_test_random[i][1]] + train_A[X_test_random[i][1]][X_test_random[i][0]])
# for i in range(len(y_pred_random_WSRec)):
#     y_pred_random_WSRec[i] = y_pred_random_WSRec[i]*0.5
# mae_random_U = mean_absolute_error(y_test_random, y_pred_random_U)
# rmse_random_U = math.sqrt(mean_squared_error(y_test_random, y_pred_random_U))
# mae_random_A = mean_absolute_error(y_test_random, y_pred_random_A)
# rmse_random_A = math.sqrt(mean_squared_error(y_test_random, y_pred_random_A))
# mae_random_WSRec = mean_absolute_error(y_test_random, y_pred_random_WSRec)
# rmse_random_WSRec = math.sqrt(mean_squared_error(y_test_random, y_pred_random_WSRec))
#
# # 计算目标项目的评价指标
# y_pred_target_U = []
# y_pred_target_A = []
# y_pred_target_WSRec = []
# for i in range(len(y_test_target)):
#     y_pred_target_U.append(train_U[X_test_target[i][0]][X_test_target[i][1]])
#     y_pred_target_A.append(train_A[X_test_target[i][1]][X_test_target[i][0]])
#     y_pred_target_WSRec.append(train_U[X_test_target[i][0]][X_test_target[i][1]] + train_A[X_test_target[i][1]][X_test_target[i][0]])
# for i in range(len(y_pred_target_WSRec)):
#     y_pred_target_WSRec[i] = y_pred_target_WSRec[i]*0.5
# mae_target_U = mean_absolute_error(y_test_target, y_pred_target_U)
# rmse_target_U = math.sqrt(mean_squared_error(y_test_target, y_pred_target_U))
# mae_target_A = mean_absolute_error(y_test_target, y_pred_target_A)
# rmse_target_A = math.sqrt(mean_squared_error(y_test_target, y_pred_target_A))
# mae_target_WSRec = mean_absolute_error(y_test_target, y_pred_target_WSRec)
# rmse_target_WSRec = math.sqrt(mean_squared_error(y_test_target, y_pred_target_WSRec))

print('-------------------------------------------------')
print('MAE_UPCC:',format(mae_U, '.4f'))
# print('MAE_random_UPCC:',format(mae_random_U, '.4f'))
# print('MAE_target_UPCC:',format(mae_target_U, '.4f'))
print('RMSE_UPCC:',format(rmse_U, '.4f'))
# print('RMSE_random_UPCC:',format(rmse_random_U, '.4f'))
# print('RMSE_target_UPCC:',format(rmse_target_U, '.4f'))
print('-------------------------------------------------')
print('MAE_APCC:',format(mae_A, '.4f'))
# print('MAE_random_APCC:',format(mae_random_A, '.4f'))
# print('MAE_target_APCC:',format(mae_target_A, '.4f'))
print('RMSE_APCC:',format(rmse_A, '.4f'))
# print('RMSE_random_APCC:',format(rmse_random_A, '.4f'))
# print('RMSE_target_APCC:',format(rmse_target_A, '.4f'))
print('-------------------------------------------------')
print('MAE_WSRec:',format(mae_WSRec, '.4f'))
# print('MAE_random_WSRec:',format(mae_random_WSRec, '.4f'))
# print('MAE_target_WSRec:',format(mae_target_WSRec, '.4f'))
print('RMSE_WSRec:',format(rmse_WSRec, '.4f'))
# print('RMSE_random_WSRec:',format(rmse_random_WSRec, '.4f'))
# print('RMSE_target_WSRec:',format(rmse_target_WSRec, '.4f'))
print('-------------------------------------------------')

# Top_K评价指标
K = 1000
sortedtargetlist_U = []
sortedtargetlist_A = []
sortedtargetlist_WSRec = []
hit_user_count_U = [0 for i in range(K)]
hit_user_count_A = [0 for i in range(K)]
hit_user_count_WSRec = [0 for i in range(K)]
pre_dataset_U = [[0] * 5825 for i in range(339)]
pre_dataset_A = [[0] * 5825 for i in range(339)]
pre_dataset_WSRec = [[0] * 5825 for i in range(339)]
api_U = [i for i in range(5825)]
api_A = [i for i in range(5825)]
api_WSRec = [i for i in range(5825)]
for i in range(len(X_test)):
    # if i % 10000 == 0:
    #     print(i)
    pre_dataset_U[X_test.iloc[i]["uid"]][X_test.iloc[i]["sid"]] = y_pred_U[i]
    pre_dataset_A[X_test.iloc[i]["uid"]][X_test.iloc[i]["sid"]] = y_pred_A[i]
    pre_dataset_WSRec[X_test.iloc[i]["uid"]][X_test.iloc[i]["sid"]] = y_pred_WSRec[i]

# print(pre_dataset)
for i in range(len(pre_dataset_U)):
    # UPCC
    pre_dataset_dic_U = dict(zip(api_U, pre_dataset_U[i]))
    sortedtargetdict_U = sorted(pre_dataset_dic_U.items(), key=lambda x: x[1], reverse=True)
    for j in range(len(sortedtargetdict_U)):
        a = sortedtargetdict_U[j]
        sortedtargetlist_U.append(a[0])
    # print(sortedtargetlist_U)
    for k in range(target_count):
        if target_item[k] in sortedtargetlist_U[:K]:
            hit_user_count_U[i] += 1
    del sortedtargetlist_U[:]
    # APCC
    pre_dataset_dic_A = dict(zip(api_A, pre_dataset_A[i]))
    sortedtargetdict_A = sorted(pre_dataset_dic_A.items(), key=lambda x: x[1], reverse=True)
    for j in range(len(sortedtargetdict_A)):
        a = sortedtargetdict_A[j]
        sortedtargetlist_A.append(a[0])
    # print(sortedtargetlist_A)
    for k in range(target_count):
        if target_item[k] in sortedtargetlist_A[:K]:
            hit_user_count_A[i] += 1
    del sortedtargetlist_A[:]
    # WSRec
    pre_dataset_dic_WSRec = dict(zip(api_WSRec, pre_dataset_WSRec[i]))
    sortedtargetdict_WSRec = sorted(pre_dataset_dic_WSRec.items(), key=lambda x: x[1], reverse=True)
    for j in range(len(sortedtargetdict_WSRec)):
        a = sortedtargetdict_WSRec[j]
        sortedtargetlist_WSRec.append(a[0])
    # print(sortedtargetlist_WSRec)
    for k in range(target_count):
        if target_item[k] in sortedtargetlist_WSRec[:K]:
            hit_user_count_WSRec[i] += 1
    del sortedtargetlist_WSRec[:]
print('hit_user_count_U:',hit_user_count_U)
print('hit_user_count_A:',hit_user_count_A)
print('hit_user_count_WSRec:',hit_user_count_WSRec)
Top_k_U = sum(hit_user_count_U)/(339*target_count)
Top_k_A = sum(hit_user_count_A)/(339*target_count)
Top_k_WSRec = sum(hit_user_count_WSRec)/(339*target_count)
print('Top_k_U:',Top_k_U)
print('Top_k_A:',Top_k_A)
print('Top_k_WSRec:',Top_k_WSRec)





# 画图
# row = []
# result = []
# count = 0
# with open('data/wsdata/picture/WSRec_average.csv', 'w') as f:
#     # result = random.sample(range(len(y_pred)), 100)
#     while (count < 40):
#         a = random.randint(0, len(y_pred_WSRec))
#         if y_test[a] > 0.1 and y_test[a] < 0.8:
#             result.append(a)
#             count += 1
#     count = 0
#     while (count < 40):
#         a = random.randint(0, len(y_pred_WSRec))
#         if y_test[a] > 0.8 and y_test[a] < 1.4:
#             result.append(a)
#             count += 1
#     count = 0
#     while (count < 40):
#         a = random.randint(0, len(y_pred_WSRec))
#         if y_test[a] > 1.4 and y_test[a] < 2.0:
#             result.append(a)
#             count += 1
#
#     print('result:', result)
#     print(len(result))
#     for i in range(120):
#         if abs(y_test[result[i]] - y_pred_WSRec[result[i]]) < 0.4:
#             row.append(0)
#             row.append(round(y_test[result[i]][0], 5))
#             row.append(round(y_pred_WSRec[result[i]], 5))
#         else:
#             row.append(1)
#             row.append(round(y_test[result[i]][0], 5))
#             row.append(round(y_pred_WSRec[result[i]], 5))
#         # print(row)
#         writer = csv.writer(f)
#         writer.writerow(row)
#         row.clear()



end = time.time()
print("程序的运行时间为：{}".format(end-start))