import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import time
import random
import csv
random.seed(2022)

start = time.time()
np.random.seed(99)
features=['uid','sid']
attacksize = 0.1
#读取攻击文件
data_dir_attack= "data/wsdata/paper/RT_Attack/"
adata=pd.read_csv(data_dir_attack+'RT_0.04_diffusion_0.1_20.txt',sep="\t",names=['uid','sid','rt'])
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

train_set = pd.concat([X_train,y_train],axis=1)
test_set = pd.concat([X_test,y_test],axis=1)

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

# 提取随机项目和目标项目
y_test_target = []
y_test_random = []
X_test_target = []
X_test_random = []
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
    # if i > 10000:
    #     break

X_test_target = pd.DataFrame(X_test_target, columns=['uid', 'sid'])
y_test_target = pd.DataFrame(y_test_target, columns=['rt'])
X_test_random = pd.DataFrame(X_test_random, columns=['uid', 'sid'])
y_test_random = pd.DataFrame(y_test_random, columns=['rt'])

test_set_target = pd.concat([X_test_target,y_test_target],axis=1)
test_set_random = pd.concat([X_test_random,y_test_random],axis=1)


class SVDpp(object):
    def __init__(self, n_epochs, n_users, n_items, n_factors, lr, reg_rate, random_seed=0, early_stopping_rounds=2):
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg_rate = reg_rate
        self.n_factors = n_factors
        np.random.seed(random_seed)
        # self.pu = np.random.randn(n_users, n_factors) / np.sqrt(n_factors)  # 参数初始化不能太大
        # self.qi = np.random.randn(n_items, n_factors) / np.sqrt(n_factors)
        # self.yj = np.random.randn(n_items, n_factors) / np.sqrt(n_factors)
        self.pu = np.random.randn(n_users, n_factors) / 100  # 参数初始化不能太大
        self.qi = np.random.randn(n_items, n_factors) / 100
        self.yj = np.random.randn(n_items, n_factors) / 100
        self.bu = np.zeros(n_users, np.double)
        self.bi = np.zeros(n_items, np.double)
        self.global_bias = 0
        self.Iu = dict()
        self.early_stopping_rounds = early_stopping_rounds

    def reg_sum_yj(self, u, i):
        sum_yj = np.zeros(self.n_factors, np.double)
        for j in self.Iu[u]:
            sum_yj += self.yj[j]
        return sum_yj / np.sqrt(len(self.Iu[u]))


    def predict(self, u, i, feedback_vec_reg):
        return self.global_bias + self.bu[u] + self.bi[i] + np.dot(self.qi[i], self.pu[u] + feedback_vec_reg)

    def fit(self, train_set, verbose=True):
        temp = 1000
        best_rounds = 0

        self.global_bias = np.mean(train_set.rt)
        # 将用户打过分的记录到Iu字典中，key为uid，value为打过分的sid的list
        g = train_set.groupby(['uid'])
        for uid, df_uid in g:
            self.Iu[uid] = list(df_uid.sid)

        for epoch in range(self.n_epochs):
            mse = 0
            for index, row in train_set.iterrows():
                u, i, r = row.uid, row.sid, row.rt
                u = int(u)
                i = int(i)
                r = float(r)
                feedback_vec_reg = self.reg_sum_yj(u, i)
                error = r - self.predict(u, i, feedback_vec_reg)
                mse += error ** 2
                self.bu[u] += self.lr * (error - self.reg_rate * self.bu[u])
                self.bi[i] += self.lr * (error - self.reg_rate * self.bi[i])
                tmp_pu = self.pu[u]
                tmp_qi = self.qi[i]
                self.pu[u] += self.lr * (error * self.qi[i] - self.reg_rate * self.pu[u])
                self.qi[i] += self.lr * (error * (tmp_pu + feedback_vec_reg) - self.reg_rate * self.qi[i])
                for j in self.Iu[u]:
                    self.yj[j] += self.lr * (error / np.sqrt(len(self.Iu[u])) * tmp_qi - self.reg_rate * self.yj[j])
            if verbose == True:
                # rmse = np.sqrt(mse / len(train_set))
                predictions = test_set.apply(
                    lambda x: self.predict(int(x.uid), int(x.sid), self.reg_sum_yj(x.uid, x.sid)), axis=1)
                rmse = np.sqrt(np.sum((test_set.rt - predictions) ** 2) / len(test_set))
                mae = np.sum(abs(test_set.rt - predictions)) / len(test_set)
                if rmse < temp:
                    temp = rmse
                    best_rounds = epoch + 1
                print('epoch: %d, mae: %.4f, rmse: %.4f' % (epoch, mae, rmse))
                if epoch + 1 - best_rounds >= self.early_stopping_rounds:
                    print('Early Stop')
                    return self
        return self

    def test(self, test_set):
        predictions = test_set.apply(lambda x: self.predict(int(x.uid), int(x.sid), self.reg_sum_yj(x.uid, x.sid)), axis=1)
        rmse = np.sqrt(np.sum((test_set.rt - predictions) ** 2) / len(test_set))
        mae = np.sum(abs(test_set.rt - predictions)) / len(test_set)

        # row = []
        # result = []
        # count = 0
        # with open('data/wsdata/picture/SVD_bandwagon.csv', 'w') as f:
        #     # result = random.sample(range(len(predictions)), 100)
        #
        #     while (count < 40):
        #         a = random.randint(0, len(predictions))
        #         if test_set.rt.iloc[a] > 0.1 and test_set.rt.iloc[a] < 0.8:
        #             result.append(a)
        #             count += 1
        #     count = 0
        #     while (count < 40):
        #         a = random.randint(0, len(predictions))
        #         if test_set.rt.iloc[a] > 0.8 and test_set.rt.iloc[a] < 1.4:
        #             result.append(a)
        #             count += 1
        #     count = 0
        #     while (count < 40):
        #         a = random.randint(0, len(predictions))
        #         if test_set.rt.iloc[a] > 1.4 and test_set.rt.iloc[a] < 2.0:
        #             result.append(a)
        #             count += 1
        #
        #     print('result:', result)
        #     print(len(result))
        #     for i in range(120):
        #         if abs(test_set.rt.iloc[result[i]] - predictions.iloc[result[i]]) < 0.4:
        #             row.append(0)
        #             row.append(round(test_set.rt.iloc[result[i]], 5))
        #             row.append(round(predictions.iloc[result[i]], 5))
        #         else:
        #             row.append(1)
        #             row.append(round(test_set.rt.iloc[result[i]], 5))
        #             row.append(round(predictions.iloc[result[i]], 5))
        #         # print(row)
        #         writer = csv.writer(f)
        #         writer.writerow(row)
        #         row.clear()

        return rmse,mae

# train = []
# for i in range(int(339*(1.0 + attacksize))):
#     train.append([])
#     for j in range(5825):
#         train[i].append(0)
# # print(train)
#
# for i in range(len(X_train)):
#     print(i)
#     train[X_train.iloc[i]['uid']][X_train.iloc[i]['sid']] = y_train.iloc[i]
# # print(train)
#
# train_array = np.array(train)

svdpp = SVDpp(n_epochs=30, n_users=int(339*(1.0 + attacksize)), n_items=5825, n_factors=35, lr=0.01, reg_rate=0.02)
svdpp.fit(train_set, verbose=True)
rmse, mae = svdpp.test(test_set)
rmse_target,mae_target = svdpp.test(test_set_target)
rmse_random,mae_random = svdpp.test(test_set_random)
print('MAE:', format(mae, '.4f'))
print('MAE_random:',format(mae_random, '.4f'))
print('MAE_target:',format(mae_target, '.4f'))
print('RMSE:', format(rmse, '.4f'))
print('RMSE_random:',format(rmse_random, '.4f'))
print('RMSE_target:',format(rmse_target, '.4f'))
end = time.time()
print("程序的运行时间为：{}".format(end-start))
