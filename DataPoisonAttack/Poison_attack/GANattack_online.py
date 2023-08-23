import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from collections import defaultdict
from re import compile,findall,split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

BATCH_SIZE = 20
BUFFER_SIZE = 339

EPOCHS = 1000
noise_dim = 100

targetCount = 20
# targetScore = 17.0
attackSize = 0.1
attackfilename = 'paper/RT_0.04_20/RT_0.04_gan_0.1_20.txt'
targetfilename = 'RT_target_item_20.txt'
np.random.seed(2021)
lambd = 0.25
# 加载数据
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

dataset = [[0] * 5825 for i in range(339)]
for i,user in enumerate(trainingData):
    for item in trainingData[user]:
        trainingSet_i[item][user] = trainingData[user][item]
        dataset[int(user)][int(item)] = trainingData[user][item]
        # print(user)
        # print(trainingData[user][item])
# print(dataset)

# dataset = np.array(trainingData)
# print(dataset)

datasets = tf.data.Dataset.from_tensor_slices(dataset)
# <TensorSliceDataset shapes: (5825,), types: tf.float32>
datasets = datasets.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# <BatchDataset shapes: (None, 5825), types: tf.float32>
# print(datasets)

for item in datasets:
    print(item.shape)
    print(type(item))
    break

def generator_model():
    generator = keras.models.Sequential([
        keras.layers.Input(shape=(100,)),  # 输入为长度100点随机向量
        keras.layers.Dense(256),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.BatchNormalization(momentum=0.8),
        keras.layers.Dense(512),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.BatchNormalization(momentum=0.8),
        keras.layers.Dense(256),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.BatchNormalization(momentum=0.8),
        keras.layers.Dense(5825, activation='relu'),
        # keras.layers.Reshape((28, 28, 1))  # 将向量重塑shape为（28，28，1），输出图片
    ])
    return generator

def discriminator_model():
    discriminator = keras.models.Sequential([
        keras.layers.Flatten(),  # 将输入的多维数据展平为一维
        keras.layers.Dense(512),
        keras.layers.ReLU(),
        keras.layers.Dense(256),
        keras.layers.ReLU(),
        keras.layers.Dense(128),
        keras.layers.ReLU(),
        keras.layers.Dense(1)
    ])
    return discriminator

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_out,fake_out):
    real_loss = cross_entropy(tf.ones_like(real_out),real_out) # 真实图片的输出与1比较
    fake_loss = cross_entropy(tf.zeros_like(fake_out),fake_out) # 生成图片的输出与0比较
    return real_loss + fake_loss
def generator_loss(fake_out):
    fake_loss = cross_entropy(tf.ones_like(fake_out),fake_out)
    return fake_loss
def sample_loss(real,fake):
    for i in range(len(real)):
        if real[i] == 0:
            fake[i] = 0
    sample_loss = cross_entropy(real,fake)
    return sample_loss
# 优化器
generator_opt = tf.keras.optimizers.Adam(2e-4,0.5)
discriminator_opt = tf.keras.optimizers.Adam(2e-4,0.5)

generator = generator_model() # 获取生成器模型
discriminator = discriminator_model() # 获取判别器模型

@tf.function
def train_step(rating):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        real_out = discriminator(rating, training=True)
        gen_rating = generator(noise, training=True)
        fake_out = discriminator(gen_rating, training=True)
        gen_loss = generator_loss(fake_out)
        # alpha = tf.random.uniform(shape=rating.get_shape(), minval=0., maxval=1.)
        # differences = gen_rating - rating  # This is different from MAGAN
        # interpolates = rating + (alpha * differences)
        # D_inter= discriminator(interpolates, training=True)
        # gradients = tf.gradients(D_inter, [interpolates])[0]
        # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        # gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        # d_loss = lambd * gradient_penalty
        disc_loss = discriminator_loss(real_out, fake_out)
        # print("g_loss:", gen_loss)
        # print("d_loss:", disc_loss)
    gradient_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradient_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_opt.apply_gradients(zip(gradient_gen, generator.trainable_variables))
    discriminator_opt.apply_gradients(zip(gradient_disc, discriminator.trainable_variables))
    return gen_loss,disc_loss

num_example_to_generate = int(attackSize*339)
seed = np.random.normal(0,1,(num_example_to_generate,noise_dim))
# seed = np.random.exponential(0.5,(num_example_to_generate,noise_dim))

def generate_attack(test_noise):
    pre_attack = generator(test_noise, training=False)
    # print(pre_attack.shape)

    # 随机生成target项目
    # targetitem = random.sample(range(0,5825), int(targetCount))
    # targetitem = np.random.randint(0, 5825, size=int(targetCount))
    # print(targetitem)

    # with open('RT_target_item_20.txt', 'r') as f:
    #     line = f.readlines()
    #     print(line)
    targetitem = np.loadtxt(targetfilename)
    print(targetitem)

    # akusers = []
    target_mu = 5
    target_signa = 7
    with open(attackfilename, 'w') as f:
        for i in range(len(pre_attack)):
            for j in range(len(pre_attack[i])):
                string = '\0'
                if float(pre_attack[i][j]) != 0 and float(pre_attack[i][j]) > 0.001 and float(pre_attack[i][j]) < 20.0:
                # if float(pre_attack[i][j]) != 0 and float(pre_attack[i][j]) > 0.001 and float(pre_attack[i][j]) < 1000.0:
                    string = str(i+339) + '\t' + str(j) + '\t' + str(round(float(pre_attack[i][j]),3))
                    # f.write(string + '\n')
                    # akusers.append(string)
                if j in targetitem:
                    while True:
                        a = np.random.normal(target_mu, target_signa)
                        if a < target_mu:
                            a = 2 * target_mu - a
                        if a < 20:
                            break
                    string = str(i+339) + '\t' + str(j) + '\t' + str(round(a,3))
                if string != '\0':
                    f.write(string + '\n')
                    # akusers.append(string)
    # for i in akusers:
    #     f.write(i + '\n')

    # count = 0
    # with open('targets.txt', 'w') as f:
    #     for i in range(len(pre_attack)):
    #         for j in range(len(pre_attack[i])):
    #             if float(pre_attack[i][j]) != 0:
    #                 count += 1
    #             f.write(str(round(float(pre_attack[i][j]),3)) + ' , ')
    #         f.write('\n' + str(count) +'\n')
    #         count = 0

def train(dataset,epochs):
    G_loss = []
    D_loss = []
    for epoch in range(1,epochs+1):
        print("epoch:",epoch)
        for rating_batch in dataset:
            gen_loss,disc_loss = train_step(rating_batch)
            G_loss.append(float(gen_loss))
            D_loss.append(float(disc_loss))
            # print(".",end="")
            # print('G_loss:', float(gen_loss))
            # print('D_loss:', float(disc_loss))
            # print(".",end="")
        print('G_loss:', np.mean(G_loss))
        print('D_loss:', np.mean(D_loss))
    generate_attack(seed)


train(datasets,EPOCHS)

