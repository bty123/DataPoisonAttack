import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve
import torch
from collections import defaultdict
from re import compile,findall,split
import random

targetCount = 5
attackSize = 0.03
attackfilename = 'paper/RT_0.04_10_small/RT_0.04_diffusion_0.03_5.txt'
# attackfilename1 = 'diffusion_test/B_RT_0.04_diffusion_0.2_20.txt'
targetfilename = 'target_item_5.txt'

sparsity = 0.04
# 数据集
with open("data/RTaktrain0.04.txt") as f:
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

dataset = [[0] * 5825 for i in range(340)]
dataset_remove0 = [] # 正常用户去0后评分列表
users_count = [] # 正常用户评分数量列表

for i,user in enumerate(trainingData):
    for item in trainingData[user]:
        trainingSet_i[item][user] = trainingData[user][item]
        dataset[int(user)][int(item)] = trainingData[user][item]
        # print(user)
        # print(trainingData[user][item])
# print(dataset)
print("shape of data :",np.shape(dataset))

# targetitem = np.loadtxt(targetfilename)
# targetitem = list(targetitem)
# print("targetitem:",targetitem)
# for i in range(len(dataset)):
#     for j in targetitem:
#         dataset[i][int(j)] = round(random.uniform(10.0, 20.0), 3)

dataset = torch.Tensor(dataset).float() # shape of data : (339, 5825)

# 改！！！


# TODO 确定超参数的值
num_steps = 200 # 可以由beta alpha 分布 均值 标准差 进行估算

# 学习的超参数 动态的在（0，1）之间逐渐增大
betas = torch.linspace(-1,1,num_steps)
betas = torch.sigmoid(betas)* (0.2e-2 - 1e-5) + 1e-5

# 计算 alpha , alpha_prod , alpha_prod_previous , alpha_bar_sqrt 等变量的值
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas ,dim=0 ) # 累积连乘  https://pytorch.org/docs/stable/generated/torch.cumprod.html
alphas_prod_p = torch.cat([torch.tensor([1]).float() ,alphas_prod[:-1]],0) # p means previous
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1-alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1-alphas_prod)

assert  alphas_prod.shape == alphas_prod.shape == alphas_prod_p.shape \
        == alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape \
        == one_minus_alphas_bar_sqrt.shape
print("all the same shape:",betas.shape)  #


# TODO 确定扩散过程中任意时刻的采样值
def q_x(x_0 ,t):
    noise = torch.randn_like(x_0) # noise 是从正太分布中生成的随机噪声
    alphas_t = alphas_bar_sqrt[t] ## 均值 \sqrt{\bar \alpha_t}
    alphas_l_m_t = one_minus_alphas_bar_sqrt[t] ## 标准差  \sqrt{ 1 - \bar \alpha_t}
    # alphas_t = extract(alphas_bar_sqrt , t, x_0) # 得到sqrt(alphas_bar[t]) ,x_0的作用是传入shape
    # alphas_l_m_t = extract(one_minus_alphas_bar_sqrt , t, x_0) # 得到sqrt(1-alphas_bart[t])
    return (alphas_t * x_0 + alphas_l_m_t * noise)

# TODO 演示原始数据分布加噪100步后的效果
# num_shows = 20
# fig, axs = plt.subplots(2,10,figsize=(28,3))
# plt.rc('text',color='blue')
# # 共有10000个点，每个点包含两个坐标
# # 生成100步以内每隔5步加噪声后的图像
# for i in range(num_shows):
#     j = i // 10
#     k = i % 10
#     t = i*num_steps//num_shows # t=i*5
#     q_i = q_x(dataset ,torch.tensor( [t] )) # 使用刚才定义的扩散函数，生成t时刻的采样数据  x_0为dataset
#     axs[j,k].scatter(q_i[:,0],q_i[:,1],color='red',edgecolor='white')
#
#     axs[j,k].set_axis_off()
#     axs[j,k].set_title('$q(\mathbf{x}_{'+str(i*num_steps//num_shows)+'})$')
# plt.show()

# TODO 编写拟合逆扩散过程 高斯分布 的模型
# \varepsilon_\theta(x_0,t)
import torch
import torch.nn as nn
class MLPDiffusion(nn.Module):
    def __init__(self,n_steps,num_units=1024,num_units1=512):
        super(MLPDiffusion,self).__init__()
        self.linears = nn.ModuleList([
            nn.Linear(5825,num_units),
            nn.LeakyReLU(),
            nn.Linear(num_units, num_units1),
            nn.LeakyReLU(),
            nn.Linear(num_units1, num_units),
            nn.LeakyReLU(),
            nn.Linear(num_units, 5825),
        ])
        self.step_embeddings = nn.ModuleList([
            nn.Embedding(n_steps, num_units),
            nn.Embedding(n_steps, num_units1),
            nn.Embedding(n_steps, num_units)
        ])
    def forward(self,x,t):
        for idx,embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2*idx](x)
            x += t_embedding
            x = self.linears[2*idx +1](x)

        x = self.linears[-1](x)
        return x

# TODO loss　使用最简单的　loss
def diffusion_loss_fn(model,x_0,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,n_steps):# n_steps 用于随机生成t
    '''对任意时刻t进行采样计算loss'''
    batch_size = x_0.shape[0]

    # 随机采样一个时刻t,为了体检训练效率，需确保t不重复
    # weights = torch.ones(n_steps).expand(batch_size,-1)
    # t = torch.multinomial(weights,num_samples=1,replacement=False) # [barch_size, 1]
    t = torch.randint(0,n_steps,size=(batch_size//2,)) # 先生成一半
    t = torch.cat([t,n_steps-1-t],dim=0) # 【batchsize,1】
    t = t.unsqueeze(-1)# batchsieze
    # print(t.shape)

    # x0的系数
    a = alphas_bar_sqrt[t]
    # 生成的随机噪音eps
    e = torch.randn_like(x_0)
    # eps的系数
    aml = one_minus_alphas_bar_sqrt[t]
    # 构造模型的输入
    x = x_0 * a + e * aml
    # 送入模型，得到t时刻的随机噪声预测值
    output = model(x,t.squeeze(-1))


    # 与真实噪声一起计算误差，求平均值

    return (e-output).pow(2).mean()


# TODO 编写逆扩散采样函数（inference过程）
def p_sample_loop(model ,shape ,n_steps,betas ,one_minus_alphas_bar_sqrt):
    '''从x[T]恢复x[T-1],x[T-2],……，x[0]'''

    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model,cur_x, i ,betas,one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq

def p_sample(model,x,t,betas,one_minus_alphas_bar_sqrt):
    '''从x[T]采样时刻t的重构值'''
    t = torch.tensor(t)
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x,t)
    mean = (1/(1-betas[t]).sqrt()) * (x-(coeff * eps_theta))
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * z
    return (sample)



# TODO 模型的训练
seed = 99
class EMA():
    '''构建一个参数平滑器'''
    def __init__(self,mu = 0.01):
        self.mu =mu
        self.shadow = {}
    def register(self,name,val):
        self.shadow[name] = val.clone()

    def __call__(self, name, x): # call函数？
        assert name in self.shadow
        new_average = self.mu * x +(1.0 -self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average

print('Training model ……')
'''
'''
batch_size = 32
dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle = True,drop_last=False)
num_epoch = 80
# plt.rc('text',color='blue')

model = MLPDiffusion(num_steps) # 输出维度是2 输入是x 和 step
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-4)

for t in range(num_epoch):
    for idx,batch_x in enumerate(dataloader):
        loss = diffusion_loss_fn(model,batch_x,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,num_steps)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(),1.) #
        optimizer.step()
        # for name ,param in model.named_parameters():
        #   if params.requires_grad:
        #       param.data = ems(name,param.data)

    # print loss
    if (t% 10 == 0):
        print(loss)
        x_seq = p_sample_loop(model,dataset.shape,num_steps,betas,one_minus_alphas_bar_sqrt)# 共有100个元素
        # print(len(x_seq))
        # print(len(x_seq[0]))
        # print(len(x_seq[0][0]))

        # fig, axs = plt.subplots(1,10,figsize=(28,3))
        # for i in range(1,11):
        #     cur_x = x_seq[i*10].detach()
        #     axs[i-1].scatter(cur_x[:,0],cur_x[:,1],color='red',edgecolor='white');
        #     axs[i-1].set_axis_off()
        #     axs[i-1].set_title('$q(\mathbf{x}_{'+str(i*10)+'})$')
        # plt.show()
count = [0 for i in range(340)]
pre_attack = []
with open('diffusion.txt', 'w') as f:
    for i in range(len(x_seq[num_steps])):
        pre_attack.append([])
        for j in range(len(x_seq[num_steps][i])):
            # if float(pre_attack[i][j]) != 0:
            #     count += 1
            # pre_attack[i] = x_seq[num_steps][i][j]
            # if j in targetitem:
            #     pre_attack[i].append(x_seq[num_steps][i][j])
            #     continue
            if x_seq[num_steps][i][j] < 0:
                pre_attack[i].append(0.0)
                f.write(str(0.0) + ',\t')
            else:
                if random.random() < sparsity*2:
                    pre_attack[i].append(x_seq[num_steps-5][i][j])
                    f.write(str(round(float(x_seq[num_steps][i][j]), 3)) + ',\t')
                    count[i] += 1
                else:
                    pre_attack[i].append(0.0)
                    continue
        f.write('\n')
    f.write('\n')
print(count)

# print(pre_attack)
# print(len(pre_attack))
# print(len(pre_attack[0]))

nums = np.random.randint(1, 340, int(attackSize*339))
print("nums:",nums)

targetitem = np.loadtxt(targetfilename)
print("targetitem:",targetitem)

with open(attackfilename, 'w') as f:
    for i in range(len(nums)):
        for j in range(len(pre_attack[i])):
            string = '\0'
            if float(pre_attack[i][j]) != 0 and float(pre_attack[i][j]) > 0.001 and float(pre_attack[i][j]) < 20.0:
                # if float(pre_attack[i][j]) != 0 and float(pre_attack[i][j]) > 0.001 and float(pre_attack[i][j]) < 1000.0:
                string = str(i + 339) + '\t' + str(j) + '\t' + str(round(float(pre_attack[i][j]), 3))
                # f.write(string + '\n')
                # akusers.append(string)
            if j in targetitem:
                string = str(i + 339) + '\t' + str(j) + '\t' + str(round(random.uniform(10.0,20.0),3))
            if string != '\0':
                f.write(string + '\n')

# target_mu = 7
# target_signa = 9
# with open(attackfilename, 'w') as f:
#     for i in range(len(nums)):
#         for j in range(len(pre_attack[i])):
#             string = '\0'
#             if float(pre_attack[i][j]) != 0 and float(pre_attack[i][j]) > 0.001 and float(pre_attack[i][j]) < 20.0:
#                 # if float(pre_attack[i][j]) != 0 and float(pre_attack[i][j]) > 0.001 and float(pre_attack[i][j]) < 1000.0:
#                 string = str(i + 339) + '\t' + str(j) + '\t' + str(round(float(pre_attack[i][j]), 3))
#                 # f.write(string + '\n')
#                 # akusers.append(string)
#             if j in targetitem:
#                 while True:
#                     a = np.random.normal(target_mu, target_signa)
#                     if a < target_mu:
#                         a = 2*target_mu - a
#                     if a < 20:
#                         break
#                 string = str(i + 339) + '\t' + str(j) + '\t' + str(round(a, 3))
#             if string != '\0':
#                 f.write(string + '\n')




    # for i in range(len(dataset_selectItem)):
    #     for j in range(len(dataset_selectItem[i])):
    #         f.write(str(round(float(dataset_selectItem[i][j]), 3)) + ',\t')
    #     f.write('\n')








