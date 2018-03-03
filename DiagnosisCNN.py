# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 20:21:03 2017

@author: shanpo
"""

import numpy as np
import tensorflow as tf

nSampleSize = 24000     # 总样本数
nSig_dim = 576          # 单个样本维度
nLab_dim = 4            # 类别维度

def getdata(nSampSize=24000):
    # 读取float型二进制数据
    signal = np.fromfile('DLdata90singal.raw', dtype=np.float32) 
    labels = np.fromfile('DLdata90labels.raw', dtype=np.float32)
    mat_sig = np.reshape(signal,[-1, nSampSize]) #由于matlab 矩阵写入文件是按照【列】优先, 需要按行读取
    mat_lab = np.reshape(labels,[-1, nSampSize])
    mat_sig = mat_sig.T # 转换成正常样式 【样本序号，样本维度】
    mat_lab = mat_lab.T
    return mat_sig, mat_lab

def zscore(xx):
    # 样本归一化到【-1，1】，逐条对每个样本进行自归一化处理
    max1 = np.max(xx,axis=1)    #按行或者每个样本，并求出单个样本的最大值
    max1 = np.reshape(max1,[-1,1])  # 行向量 ->> 列向量
    min1 = np.min(xx,axis=1)    #按行或者每个样本，并求出单个样本的最小值
    min1 = np.reshape(min1,[-1,1])  # 行向量 ->> 列向量
    xx = (xx-min1)/(max1-min1)*2-1
    return xx

def NextBatch(iLen, n_batchsize):
    # iLen: 样本总数
    # n_batchsize: 批处理大小
    # 返回n_batchsize个随机样本（序号）
    ar = np.arange(iLen)    # 生成0到iLen-1，步长为1的序列
    np.random.shuffle(ar)   # 打断顺序
    return ar[0:n_batchsize]

def weight_variable(shape):
    # 定义权重
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # 定义偏置
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # 卷积层
    # stride [1, x_movement:1, y_movement:1, 1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # 池化层
    # stride [1, x_movement:2, y_movement:2, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# 为NN定义placeholder
xs = tf.placeholder(tf.float32,[None,nSig_dim])
ys = tf.placeholder(tf.float32,[None,nLab_dim])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(xs, [-1, 24, 24, 1])   # 24*24

W_conv1 = weight_variable([5, 5, 1, 32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 32@24*24
h_pool1 = max_pool_2x2(h_conv1)  # size 32@12*12

## conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64 
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 64@12*12
h_pool2 = max_pool_2x2(h_conv2)     # output size 64@6*6

## fc1 layer ##
W_fc1 = weight_variable([6*6*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 6, 6, 64] ->> [n_samples, 6*6*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 6*6*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, nLab_dim])
b_fc2 = bias_variable([nLab_dim])
y_pre = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(y_pre),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 正确率
correct_pred = tf.equal(tf.argmax(y_pre, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

mydata = getdata()
iTrainSetSize =  np.floor(nSampleSize*3/4).astype(int) # 训练样本个数
iIndex = np.arange(nSampleSize) # 按照顺序，然后划分训练样本、测试样本
train_index = iIndex[0:iTrainSetSize]
test_index = iIndex[iTrainSetSize:nSampleSize]

train_data = mydata[0][train_index]     # 训练数据
train_y = mydata[1][train_index]        # 训练标签
test_data = mydata[0][test_index]       # 测试数据
test_y = mydata[1][test_index]          # 测试标签

train_x = zscore(train_data)        # 对训练数据进行归一化
test_x = zscore(test_data)          # 对测试数据进行归一化

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(2000):
        intervals = NextBatch(iTrainSetSize, 100) # 每次从所有样本中随机取100个样本（序号）
        xx = train_x[intervals]
        yy = train_y[intervals]
        sess.run(train_step, feed_dict={xs:xx, ys:yy, keep_prob:0.9})
        if step%100 == 0:
            acc = sess.run(accuracy,feed_dict={xs:xx, ys:yy, keep_prob:1.0})
            print("step: " + "{0:4d}".format(step) + ",  train acc:" + "{:.4f}".format(acc))
    test_acc = sess.run(accuracy,feed_dict={xs:test_x, ys:test_y, keep_prob:1.0})
    print("test acc:" + "{:.4f}".format(test_acc))

# 结果   大概98.4%
