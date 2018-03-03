# DiagnosisDL2TF
使用TensorFlow建立简单的轴承故障诊断模型
## 1、数据来源及介绍
轴承、轴、齿轮是旋转机械重要组成部分，为了验证深度学习在旋转装备故障分类识别的有效性，本文选取凯斯西储大学轴承数据库(Case Western Reserve University, CWRU)[9]为验证数据。CWRU实验装置如图 4‑1所示。轴承通过电火花加工设置成四种尺寸的故障直径，分别为0.007、0.014、0.021、0.028英寸。实验中使用加速度传感器采集振动信号，传感器分别被放置在电机驱动端与风扇端。由于驱动端采集到的振动信号数据全面，并且收到其他部件和环境噪声的干扰较少，因此本文选取驱动端采集的振动信号作为实验数据。实验数据包括4种轴承状态下采集到的振动信号，分别为正常状态（Normal，N）、滚珠故障状态（Ball Fault，BF）、外圈故障状态（Outer Race Fault，ORF）以及内圈故障状态（Inner Race Fault，IRF），每种状态下采集到的信号又按照故障直径与负载的大小进行分类，其中故障直径分别为0.007、0.014、0.021、0.028英寸，负载大小从0Hp-3Hp（1Hp=746W），对应转速为1797rpm、1772rpm、1750rpm、1730rpm。负载为0Hp，故障直径为0.007英寸时四种工况下采集的振动信号样本时域波形如图 4‑2所示。本文选取CWRU数据集中采样频率为12k Hz的各个状态的样本，通过深度学习建立故障诊断模型，对电机轴承的四种故障进行分类识别。
由于负载的不同，转速不恒定，但采集的转速都在1800rpm左右，采样频率为12kHz，转轴转一圈，约采集400（60/1800*12000 = 400）个数据点。由于采用原始数据切分方式，通常取稍微大于一个周期的点数比较合适，为了便于多层CNN网络的输入，本文以24*24=576点作为输入长度，分别测试全连接神经网络、CNN、LSTM等。 
## 2、实验过程及结果分析
以576点步长对原始振动数据进行切分，采取数据重叠度为0，可以得208段数据，继而得到数据样本2496个，以3:1划分出训练样本和测试样本数，但发现误差非常的大。考虑到深度学习对样本的数量是有一定要求的，通常样本量越大越好，但试验数据时长是固定的，只能通过调节数据重叠度来增加样本量，这样也在一定概率上弥补了输入点数设置误差。随着数据量的增加，训练参数也需要进行相应的更改。分别设置重叠度为50%、70%、80%、90%，对应的样本量由原来的2496分别变成了4492、9360、12480、24960。由于说明数据重叠取样对于深度神经网络提升的效果，在此，只列举数据重叠度90%一种情况。
```matlab
clear all;
clc;
load DataSet;
[iType, iCondition] = size(A);
iExtSize = 24*24;
iSampleRate = 12000;
iTime = 10;
iOverlap = floor(iExtSize * 0.9);
iUCover = iExtSize - iOverlap;
iGetDataLen = iSampleRate*iTime + iExtSize;
iLen2 = floor((iGetDataLen-iExtSize)/iUCover) + 1;
iLen1 =  floor(iLen2/100)*100;
iGetDataLen = iLen1*iUCover + iExtSize;
fExtSamp = zeros(iType, iGetDataLen);
 
tmp = 0;
for jCnt = 1: iType
    str1 = sprintf('%03d',A(jCnt,1));
    szValName = strcat('X', str1, '_DE_time');
    eval(strcat('tmp=',szValName,';'));
    fExtSamp(jCnt,:) = tmp(1:iGetDataLen);
end
iLen = iLen1;
iSampSize = iLen * iType;
fData = zeros(iSampSize, iExtSize);
fLabel = zeros(iSampSize, 4);
 
for iCnt = 1:1:iLen
    iInterval = (iCnt -1)*iUCover + (1:1:iExtSize);
    for jCnt =1:1:iType
        fData((iCnt - 1)*iType + jCnt,:) = fExtSamp(jCnt, iInterval);
        if (jCnt ==1)
            fLabel((iCnt - 1)*iType + jCnt,:) = [1 0 0 0];
        end
        if (jCnt >=2 && jCnt<=5)
            fLabel((iCnt - 1)*iType + jCnt,:) = [0 1 0 0];
        end
        if (jCnt >=6 && jCnt<=9)
            fLabel((iCnt - 1)*iType + jCnt,:) = [0 0 1 0];
        end
        if (jCnt >=10)
            fLabel((iCnt - 1)*iType + jCnt,:) = [0 0 0 1];
        end
    end
end
save('DL_Data90.mat','fData', 'fLabel');  
```
## 3、深度学习模型在故障分别上的应用
### 3.1  全连接DNN
```python
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 15:09:43 2017
 
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
    mat_sig = np.reshape(signal,[-1, nSampSize]) #由于matlab 矩阵写入文件是按照列优先,
    mat_lab = np.reshape(labels,[-1, nSampSize])
    mat_sig = mat_sig.T # 转换成正常样式 【样本序号，样本维度】
    mat_lab = mat_lab.T
    return mat_sig, mat_lab
 
def layer(inputs, in_size, out_size, keep_prob, activation_function=None):
    # 构建全连接神经网络层
    # inputs：输入
    # in_size：输入维度
    # out_size：输出维度
    # keep_prob：dropout的保持率
    # activation_function 激活函数
    weights = tf.Variable(tf.random_normal([in_size,out_size],stddev=0.1))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.01)
    WxPb = tf.matmul(inputs,weights) + biases
    WxPb = tf.nn.dropout(WxPb,keep_prob)
    if activation_function==None:
        outputs = WxPb
    else:
        outputs = activation_function(WxPb)
    return outputs
 
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
   
xs = tf.placeholder(tf.float32,[None, nSig_dim])        # 样本输入
ys = tf.placeholder(tf.float32,[None, nLab_dim])        # 样本标签
keep_pro = tf.placeholder(tf.float32)                   # 1-dropout概率
 
#FCN 模型 [nSig_dim:576, 200, 50, nLab_dim:10]
Layer1 = layer(xs, nSig_dim, 200, keep_pro, activation_function=tf.nn.relu)
hiddenL = layer(Layer1, 200, 50, keep_pro, activation_function=tf.nn.relu)
y_pre  = layer(hiddenL, 50, nLab_dim, keep_pro, activation_function=tf.nn.softmax)
 
loss = tf.reduce_mean(tf.reduce_sum(-ys*tf.log(y_pre),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
 
correct_pred = tf.equal(tf.argmax(y_pre,1),tf.argmax(ys,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32)) #计算正确率
 
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
    for step in range(10000):
        intervals = NextBatch(iTrainSetSize, 100) # 每次从所有样本中随机取100个样本（序号）
        xx = train_x[intervals]
        yy = train_y[intervals]
        sess.run(train_step, feed_dict={xs:xx, ys:yy, keep_pro:0.95})
        if step%100 == 0:
            acc = sess.run(accuracy,feed_dict={xs:xx, ys:yy, keep_pro:1.0})
            print("step: " + "{0:4d}".format(step) + ",  train acc:" + "{:.4f}".format(acc))
    print(sess.run(accuracy,feed_dict={xs:test_x, ys:test_y, keep_pro:1.0})) 
```
准确率：
```python
准确率大概 97.8%左右。
```

### 3.2  CNN
```python
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

# 结果   大概98.4%
```
### 3.3 LSTM
```python
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 15:26:21 2017
 
@author: shanpo
"""
 
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
 
nSampleSize = 24000     # 总样本数
nSig_dim = 576          # 单个样本维度
nLab_dim = 4            # 类别维度
 
learning_rate = 1e-3
batch_size = tf.placeholder(tf.int32, [])   # 在训练和测试，用不同的 batch_size
input_size = 24     # 每个时刻的输入维数为 24
timestep_size = 24  # 时序长度为24
hidden_size = 128   # 每个隐含层的节点数
layer_num = 3       # LSTM layer 的层数
class_num = nLab_dim       # 类别维数
 
def getdata(nSampSize=24000):
    # 读取float型二进制数据
    signal = np.fromfile('DLdata90singal.raw', dtype=np.float32)
    labels = np.fromfile('DLdata90labels.raw', dtype=np.float32)
    #由于matlab 矩阵写入文件是按照【列】优先, 需要按行读取
    mat_sig = np.reshape(signal,[-1, nSampSize])
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
    np.random.shuffle(ar)   # 打乱顺序
    return ar[0:n_batchsize]
 
xs = tf.placeholder(tf.float32, [None, nSig_dim])
ys = tf.placeholder(tf.float32, [None, class_num])
keep_prob = tf.placeholder(tf.float32)
 
x_input = tf.reshape(xs, [-1, 24, 24])
 
# 搭建LSTM 模型
def unit_LSTM():
    # 定义一层 LSTM_cell，只需要说明 hidden_size
    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    #添加 dropout layer, 一般只设置 output_keep_prob
    lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    return lstm_cell
 
#调用 MultiRNNCell 来实现多层 LSTM
mLSTM_cell = rnn.MultiRNNCell([unit_LSTM() for icnt in range(layer_num)], state_is_tuple=True)
 
#用全零来初始化state
init_state = mLSTM_cell.zero_state(batch_size, dtype=tf.float32)
outputs, state = tf.nn.dynamic_rnn(mLSTM_cell, inputs=x_input,
                                   initial_state=init_state, time_major=False)
h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]
 
# 设置 loss function 和 优化器
W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)
# 损失和评估函数
cross_entropy = tf.reduce_mean(tf.reduce_sum(-ys*tf.log(y_pre),reduction_indices=[1]))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(ys,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
 
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
# 开始训练。
with tf.Session() as sess:
    sess.run(init)
    for icnt in range(1000):
        _batch_size = 100
        intervals = NextBatch(iTrainSetSize, _batch_size) # 每次从所有样本中随机取100个样本（序号）
        xx = train_x[intervals]
        yy = train_y[intervals]   
        if (icnt+1)%100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                xs:xx, ys: yy, keep_prob: 1.0, batch_size: _batch_size})
            print("step: " + "{0:4d}".format(icnt+1) + ",  train acc:" + "{:.4f}".format(train_accuracy))
        sess.run(train_op, feed_dict={ xs:xx, ys: yy, keep_prob: 0.9, batch_size: _batch_size})
    bsize = test_x.shape[0]
    test_acc = sess.run(accuracy,feed_dict={xs:test_x, ys:test_y, keep_prob: 1.0, batch_size:bsize})
    print("test acc:" + "{:.4f}".format(test_acc))
    # 结果
    # test acc:0.9868
    # test acc:0.9907 
```

```
