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