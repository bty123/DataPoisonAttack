import pandas as pd
import tensorflow as tf
import numpy as np
import math
import os
from models import LR,MLR,FM, AFM, CFM, NFM, BiFM, DeepBiFM, FiBiFM, FiBiNet, MLP, DCN, WideAndDeep, FMAndDeep, DeepFM, DeepAFM, AutoInt,DeepAutoInt,AutoFIS
import PermutationImportance
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import random
import csv

random.seed(2022)
if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    np.random.seed(99)
    #features_neighbor=[]

    #features=['uid','sid','ucountry','uas','ulatulong','sprovider','scountry','sas','slatslong']
    features=['uid','sid']

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

    print("分割前训练集行数：   ",data.shape)

    #----------密度------
    density=0.04
    #----------密度--------
    X_train, X_test, y_train, y_test = train_test_split(data[features], data['rt'], train_size = density, random_state = 99)
    #
    # print(X_test)
    # print(y_test)



    print("训练集行数：  ",X_train.shape)
    print("分割后训练集格式：   ",type(X_train))
    # X_train = X_train.append(adata)
    X_train = pd.concat([X_train, adata[['uid', 'sid']]], ignore_index=True)
    print("注入攻击后的训练集X行数：  ",X_train.shape)
    print("注入攻击前y行数",y_train.shape)
    # y_train = y_train.append(adata['rt'])
    y_train = pd.concat([y_train, adata['rt']], ignore_index=True)
    print("注入攻击后的训练集y行数：  ",y_train.shape)
    print(X_train)
    print(y_train)
    # X_train = np.array(X_train)
    # y_train = np.array(y_train)

    # 提取目标项目
    target_item = []
    with open(data_dir_attack + 'target_item_20.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            target_item.append(int(line))

    print('target_item:', target_item)

    # 提起与目标项目数量一致的随机项目
    random_item = np.random.randint(0, 5825, size=20)
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

    test_set_target = pd.concat([X_test_target, y_test_target], axis=1)
    test_set_random = pd.concat([X_test_random, y_test_random], axis=1)
   

    y_train=y_train.values.reshape((-1,1))#负数表示模糊控制，改变数组形状为不定行，1列
    y_test=y_test.values.reshape((-1,1))# --------------修改过------------------------------------------
    print(y_train)
    print(y_test)

    #------统计个数，为了编码---------------
    X_data = X_train.append(X_test)
    features_sizes=[X_data[f].nunique() for f in features]#每个特征不同值的统计个数

    #------数据集标签编码--------------------------------------------
    from sklearn.preprocessing import LabelEncoder
    lbl=LabelEncoder()
    for c in features:
        lbl.fit(list(X_train[c])+list(X_test[c]))  #训练LabelEncoder
        X_train[c]=lbl.transform(list(X_train[c]))  #使用训练好的LabelEncoder对标签进行编码
        X_test[c]=lbl.transform(list(X_test[c]))
    #------数据集标签编码--------------------------------------------


    lossls=[]
    maels=[]
    rmsels=[]
    grda_c=0.0015#
    grda_mu=0.7#
    batch_size=64
    deep_layers=(16,16,16)
    lr=0.0004#
    lr2=0.01#
    k=60
    LR_Weight=0.1
    FM_Weight=0.4
    MLP_Weight=0.5
    dropout_keeprate=0.5

    #for i in range(1):
    for i in range(1):
        tf.reset_default_graph()
        #batch_size=batch_size+100    
        
                    #1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36
        #comb_mask = [1,1,1,1,0,1,1,1,1,1, 1  ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1]#ȥ��һ��������#grda_c = 0.003,grda_mu = 0.7    
        
        
        #model=FM(features_sizes,dense_features_size=0,loss_type='rmse',k=k,deep_layers=deep_layers,activation=tf.nn.relu,FM_ignore_interaction=None,dropout_keeprate=dropout_keeprate,grda_c = grda_c,grda_mu =grda_mu,retrain_stage = 0,comb_mask=None,metric_type=None)#best grda_c = 0.0015,grda_mu = 0.7  
        #best_score=model.fit(X_train[features],X_test[features],y_train,y_test,lr=lr,lr2=lr2,N_EPOCH=100,batch_size=batch_size,early_stopping_rounds=5,LR_Weight=LR_Weight,FM_Weight=FM_Weight,MLP_Weight=MLP_Weight)     
#FM加攻击
        model=FM(features_sizes,loss_type='rmse',k=k,FM_ignore_interaction=None,dropout_keeprate=dropout_keeprate,metric_type='mae')#best grda_c = 0.0015,grda_mu = 0.7
        best_score=model.fit(X_train[features],X_test[features],y_train,y_test,lr=lr2,N_EPOCH=1000,batch_size=batch_size,early_stopping_rounds=5)
        lossls.append(best_score)

        # model=MLP(features_sizes,dense_features_size=0,loss_type='rmse',deep_layers=deep_layers,activation=tf.nn.relu,k=k,dropout_keeprate=dropout_keeprate,metric_type=None)#best grda_c = 0.0015,grda_mu = 0.7  
        # best_score=model.fit(X_train[features],X_test[features],y_train,y_test,lr=lr,lr2=lr2,N_EPOCH=100,batch_size=batch_size,early_stopping_rounds=5,LR_Weight=LR_Weight,FM_Weight=FM_Weight,MLP_Weight=MLP_Weight)     
        # lossls.append(best_score)
#DCN加攻击
        # model=DCN(features_sizes,dense_features_size=0,loss_type='rmse',k=k,deep_layers=deep_layers,activation=tf.nn.relu,dropout_keeprate=dropout_keeprate,metric_type=None)
        # best_score=model.fit(X_train[features],X_test[features],y_train,y_test,N_EPOCH=1000,lr=lr,batch_size=batch_size,early_stopping_rounds=20)     
        # lossls.append(best_score)
#LR加攻击
        # model=LR(features_sizes,loss_type='rmse',hash_size=None,metric_type=None)
        # best_score=model.fit(X_train[features],X_test[features],y_train,y_test,lr=lr,N_EPOCH=1000,batch_size=batch_size,early_stopping_rounds=5)
        # lossls.append(best_score)

#MLR模型加攻击    
        # model=MLR(features_sizes,loss_type='rmse',hash_size=None,metric_type=None)
        # best_score=model.fit(X_train[features],X_test[features],y_train,y_test,lr=lr,N_EPOCH=3000,batch_size=500,early_stopping_rounds=50)     
        # lossls.append(best_score)
# MLP模型未加攻击
#         model = MLP(features_sizes, dense_features_size=0, loss_type='rmse', k=k, deep_layers=deep_layers,
#                     activation=tf.nn.relu, dropout_keeprate=dropout_keeprate, metric_type="mae")
#         best_score = model.fit(X_train[features], X_test[features], y_train, y_test, lr=lr, N_EPOCH=500,
#                                batch_size=batch_size, early_stopping_rounds=7)
#         lossls.append(best_score)
#AFM模型加攻击
        # model=AFM(features_sizes,loss_type='rmse',k=k,FM_ignore_interaction=None,dropout_keeprate=dropout_keeprate,metric_type="mae")#best grda_c = 0.0015,grda_mu = 0.7
        # best_score=model.fit(X_train[features],X_test[features],y_train,y_test,lr=lr,N_EPOCH=500,batch_size=batch_size,early_stopping_rounds=7)
        # lossls.append(best_score)
#CFM模型加攻击
        # model = CFM(features_sizes, loss_type='rmse', k=k, FM_ignore_interaction=None,dropout_keeprate=dropout_keeprate, metric_type="mae")  # best grda_c = 0.0015,grda_mu = 0.7
        # best_score = model.fit(X_train[features], X_test[features], y_train, y_test, lr=lr, N_EPOCH=100,batch_size=batch_size, early_stopping_rounds=5)
        # lossls.append(best_score)

# DeepFM模型未加攻击
#         model = DeepFM(features_sizes, dense_features_size=0, loss_type='rmse', k=k, deep_layers=deep_layers,
#                        activation=tf.nn.relu, FM_ignore_interaction=None, dropout_keeprate=dropout_keeprate,
#                        metric_type="mae")
#         best_score = model.fit(X_train[features], X_test[features], y_train, y_test, lr=lr, N_EPOCH=500,
#                                batch_size=batch_size, early_stopping_rounds=7)
#         lossls.append(best_score)

        y_pred = model.predict(X_test[features])
        y_pred_random = model.predict(X_test_random[features])
        y_pred_target = model.predict(X_test_target[features])

        mae = mean_absolute_error(y_test, y_pred)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        mae_random = mean_absolute_error(y_test_random, y_pred_random)
        rmse_random = math.sqrt(mean_squared_error(y_test_random, y_pred_random))
        mae_target = mean_absolute_error(y_test_target, y_pred_target)
        rmse_target = math.sqrt(mean_squared_error(y_test_target, y_pred_target))

        print('MAE:', format(mae, '.4f'))
        print('MAE_random:', format(mae_random, '.4f'))
        print('MAE_target:', format(mae_target, '.4f'))
        print('RMSE:', format(rmse, '.4f'))
        print('RMSE_random:', format(rmse_random, '.4f'))
        print('RMSE_target:', format(rmse_target, '.4f'))
 
        # maels.append(mae)
        # rmsels.append(rmse)
        # print(model)
        # print("Loss \t %.4f  \t %.4f \t %s"% (pd.Series(lossls).mean(),pd.Series(lossls).min(),str(lossls)))
        # print("MAE \t %.4f   \t %.4f \t %s"% (pd.Series(maels).mean(),pd.Series(maels).min(),str(maels)))
        # print("RMSE \t %.4f  \t %.4f \t %s"% (pd.Series(rmsels).mean(),pd.Series(rmsels).min(),str(rmsels)))
        #
        # #with open('result\\batch_size_0.02.txt', 'a',encoding='utf-8') as writer:
        # #    writer.write("Loss:" + str(lossls) + "\t" + "MAE:" + str(maels)+ "\t" +"RMSE:"+ str(rmsels) \
        # #        + "\t"+"Density:"+str(density) +"\t"+"grda_c:"+str(grda_c)+"\t"+"grda_mu:"+str(grda_mu)+"batch_size:"+str(batch_size)+'\n')
        # lossls=[]
        # maels=[]
        # rmsels=[]

        # row = []
        # # result = []
        # count = 0
        # with open('data/wsdata/picture/1test_FM_random.csv', 'w') as f:
        #     result = random.sample(range(len(y_pred_target)), 90)
            # while (count < 20):
            #     a = random.randint(0, len(y_pred_target))
            #     if y_test_target.iloc[a] > 0.1 and y_test_target.iloc[a] < 0.8:
            #         result.append(a)
            #         count += 1
            # count = 0
            # while (count < 40):
            #     a = random.randint(0, len(y_pred_target))
            #     if y_test_target.iloc[a] > 0.8 and y_test_target.iloc[a] < 1.4:
            #         result.append(a)
            #         count += 1
            # count = 0
            # while (count < 30):
            #     a = random.randint(0, len(y_pred_target))
            #     if y_test_target.iloc[a] > 1.4 and y_test_target.iloc[a] < 2.0:
            #         result.append(a)
            #         count += 1

            # print('result:', result)
            # print(len(result))
            # for i in range(90):
            #     if abs(float(y_test_target.iloc[result[i]]) - y_pred_target[result[i]][0]) < 1.0:
            #         row.append(0)
            #         row.append(round(float(y_test_target.iloc[result[i]]), 5))
            #         row.append(round(y_pred_target[result[i]][0], 5))
            #     else:
            #         row.append(1)
            #         row.append(round(float(y_test_target.iloc[result[i]]), 5))
            #         row.append(round(y_pred_target[result[i]][0], 5))
            #     # print(row)
            #     writer = csv.writer(f)
            #     writer.writerow(row)
            #     row.clear()