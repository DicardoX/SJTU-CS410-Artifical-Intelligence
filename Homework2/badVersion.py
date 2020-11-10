# Logistic regression: 预测函数为sigmoid(WX + b)

import tensorflow as tf # Tensorflow
import numpy as np # 用于后续计算
import pandas as pd # 用于读取数据文件
import matplotlib.pyplot as plt # 用于画图

LR = 0.00001 # 学习率
EPOCH = 1000 # 迭代次数

# 数组分割，每隔n分割一个新数组
def list_split(items, n):
    return [items[i:i+n] for i in range(0, len(items), n)]

# 读取数据文件
readIn = pd.read_csv("assignment 2-supp.csv", header=0) # 将csv文件中的数据读入data变量，将第一行作为名称，不读入
data = readIn.values # 将DataFrame转化为二维数组

# 分离数据和标签，并获取数据维数
data_X = data[:,:-1] # 将特征放在变量train_X中（8维）
data_Y = data[:,-1:] # 将标签放在变量train_Y中（1维）
feature_num = len(data_X[0])
sample_num = len(data_X)
#print ("Size of train_X: {}x{}".format(sample_num, feature_num))
#print ("Size of train_Y: {}x{}".format(len(data_Y), len(data_Y[0])))

# 数据分割，构建训练集和验证集
tmpArray_X = np.vsplit(data_X, 3)
tmpArray_Y = np.vsplit(data_Y, 3)
train_X = np.append(tmpArray_X[0], tmpArray_X[1])
train_Y = np.append(tmpArray_Y[0], tmpArray_Y[1])
train_X = np.reshape(train_X, [-1, 8])           # 取数据的前2/3作为训练集
train_Y = np.reshape(train_Y, [-1, 1])
validation_X = tmpArray_X[2]                     # 数据的后1/3作为验证集
validation_Y = tmpArray_Y[2]

def train_model(lr = LR):
    # 数据集（占位符创建），占位符意味着变量的值未指定，直到开始训练模型时才需要将给定的数据赋给变量
    input = tf.placeholder(tf.float32, name="input")
    label = tf.placeholder(tf.float32, name="label")

    # 训练目标
    W = tf.Variable(tf.zeros([feature_num, 1])) # W为权值矩阵，将W的初始值设为了feature_num维的零向量，类型为Variable，意味着变量将在训练迭代过程中不断变化，最终得到我们需要的值
    b = tf.Variable([0.1]) # b为偏差bias，将b的初始值设置为0.1

    # 损失函数
    db = tf.matmul(input, tf.reshape(W, [-1, 1])) + b # 矩阵变换后相乘
    prediction = tf.sigmoid(db, name="prediction") # 使用的sigmoid预测函数
    cost0 = label * tf.log(prediction)
    cost1 = (1 - label) * tf.log(1 - prediction)
    cost = (cost0 + cost1) / -sample_num # 逻辑回归的损失函数定义
    loss = tf.reduce_sum(cost) # 损失函数求和后表征代价函数
    #accuracy = tf.metrics.accuracy(labels=tf.argmax(label, axis=1), predictions=tf.argmax(prediction, axis=1), )[1]
    #accuracy = tf.metrics.accuracy(labels=input, predictions=prediction, )[1]

    # 优化
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss) # 设定优化目标，使用gradient descent的优化器进行优化

    # 模型训练
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess: # 任务执行的主体，之前定义的只是一个模型为了得到结果需要的执行步骤和框架，类似于流程图，需要主体来实际运行
        sess.run(init)
        for epoch in range(EPOCH):
            avgLoss = 0
            for i in range(0, len(train_X), int(len(train_X) / 2)):
                feed_dict = {"input:0": train_X[i: i + int(len(train_X) / 2)], "label:0": train_Y[i: i + int(len(train_X) / 2)]}  # 将传入数据存放在一个变量中
                _, c = sess.run([optimizer, loss], feed_dict) # 参数传入sess.run()，train为之前定义的优化目标
                avgLoss += c
            if epoch % 100 == 0:
                print ("Epoch: ", epoch, "|", "Average Loss: " , avgLoss / 100)
                avgLoss = 0

    # 模型验证
    #X_ = tf.placeholder(tf.float32, (256, 8))
    #Y_ = tf.placeholder(tf.float32, (256, 1))
    #init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #sess.run(init)
    #accuracy_ = 0
    #feed_dict2 = {'input:0': validation_X, 'label:0': validation_Y}
    #accuracy_ = sess.run(accuracy, feed_dict2)
    #print(accuracy_)
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({'input:0': validation_X, 'label:0': validation_Y}))

train_model(LR)









