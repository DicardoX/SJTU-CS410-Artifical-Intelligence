# Logistic regression: 预测函数为sigmoid(WX + b)

import tensorflow as tf # Tensorflow
import numpy as np # 用于后续计算
import pandas as pd # 用于读取数据文件
import matplotlib.pyplot as plt # 用于画图

lr = 0.00001 # 学习率
EPOCH = 100000 # 迭代次数

# 读取数据文件
readIn = pd.read_csv("assignment 2-supp.csv", header=0) # 将csv文件中的数据读入data变量，将第一行作为名称，不读入
data = readIn.values # 将DataFrame转化为二维数组

# 分离数据和标签，并获取数据维数
train_X = data[:,:-1] # 将特征放在变量train_X中（8维）
train_Y = data[:,-1:] # 将标签放在变量train_Y中（1维）
feature_num = len(train_X[0])
sample_num = len(train_X)

#print ("Size of train_X: {}x{}".format(sample_num, feature_num))
#print ("Size of train_Y: {}x{}".format(len(train_Y), len(train_Y[0])))
#print (train_X)

# 数据集（占位符创建），占位符意味着变量的值未指定，直到开始训练模型时才需要将给定的数据赋给变量
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 训练目标
W = tf.Variable(tf.zeros([feature_num, 1])) # W为权值矩阵，将W的初始值设为了feature_num维的零向量，类型为Variable，意味着变量将在训练迭代过程中不断变化，最终得到我们需要的值
b = tf.Variable([-0.9]) # b为偏差bias，将b的初始值设置为-0.9

# 损失函数
db = tf.matmul(X, tf.reshape(W, [-1, 1])) + b # 矩阵变换后相乘
hyp = tf.sigmoid(db) # 使用的sigmoid预测函数

cost0 = Y * tf.log(hyp)
cost1 = (1 - Y) * tf.log(1 - hyp)
cost = (cost0 + cost1) / -sample_num # 逻辑回归的损失函数定义
loss = tf.reduce_sum(cost) # 损失函数求和后表征代价函数

# 优化
optimizer = tf.train.GradientDescentOptimizer(lr) # 使用gradient descent的优化器进行优化
train = optimizer.minimize(loss) # 设定优化目标，即最小化loss

# 模型训练
init = tf.global_variables_initializer() # 将之前定义的Variable初始化
sess = tf.Session() # 任务执行的主体，之前定义的只是一个模型为了得到结果需要的执行步骤和框架，类似于流程图，需要主体来实际运行
sess.run(init)

#train_X.astype(np.float32)
#train_Y.astype(np.float32)
feed_dict = {X:train_X, Y:train_Y} # 将传入数据存放在一个变量中

for epoch in range(EPOCH):
    sess.run(train, feed_dict) # 参数传入sess.run()，train为之前定义的优化目标
    if epoch % 100 == 0:
        print("Epoch: ", epoch, sess.run(W).flatten(), sess.run(b).flatten())









