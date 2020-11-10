# Logistic regression: 预测函数为sigmoid(WX + b)

import tensorflow as tf # Tensorflow
import numpy as np # 用于后续计算
import pandas as pd # 用于读取数据文件
import matplotlib.pyplot as plt # 用于画图

LR = 0.0003 # 学习率
EPOCH = 1000 # 迭代次数

# 读取数据文件
readIn = pd.read_csv("assignment 2-supp.csv", header=0) # 将csv文件中的数据读入data变量，将第一行作为名称，不读入
data = readIn.values # 将DataFrame转化为二维数组

# 分离数据和标签，并获取数据维数
train_X = data[:,:-1] # 将特征放在变量train_X中（8维）
train_Y = data[:,-1:] # 将标签放在变量train_Y中（1维）
feature_num = len(train_X[0]) # 特征数
sample_num = len(train_X) # 样本数

# 绘图
accuracyArr = [] # 准确率数组
lrArr = [] # 学习率数组

# 数据集（占位符创建），占位符意味着变量的值未指定，直到开始训练模型时才需要将给定的数据赋给变量
input = tf.placeholder(tf.float32, name="input")
label = tf.placeholder(tf.float32, name="label")

def train_model(lr = LR):
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

    # 优化
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss) # 设定优化目标，使用gradient descent的优化器进行优化

    # 模型训练
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess: # 任务执行的主体，之前定义的只是一个模型为了得到结果需要的执行步骤和框架，类似于流程图，需要主体来实际运行
        sess.run(init)
        avgLoss = 0
        avgAccuracy = 0
        for epoch in range(EPOCH):
            feed_dict = {"input:0": train_X, "label:0": train_Y} # 将传入数据存放在一个变量中
            _, tmpLoss = sess.run([optimizer, loss], feed_dict)  # 参数传入sess.run()，train为之前定义的优化目标
            avgLoss += tmpLoss

            # 计算accuracy
            tmpAmount = 0
            pred = sess.run(prediction, feed_dict)
            for i in range(0, sample_num, 1):
                if((pred[i] < 0.5 and train_Y[i] == 0) or (pred[i] >= 0.5 and train_Y[i] == 1)):
                    tmpAmount = tmpAmount + 1
            avgAccuracy += float(tmpAmount / sample_num)

            if epoch % 100 == 0:
                #print("Epoch: ", epoch, "|", "Average Loss: %.4f" % float(avgLoss / 100), "|", "Average Accuracy: %.4f" % float(avgAccuracy / 100))
                avgLoss = 0
                avgAccuracy = 0

        return float(avgAccuracy / 100)

# 迭代运行模型，记录accuracy和learning rate的关系
for i in range(1, 5000, 50):
    lr = (i - 1) * 0.0000001
    if(i == 1):
        lr = i * 0.0000001  # 第一个learning rate设置为0.0000001

    accuracy_ = train_model(lr)
    accuracyArr.append(accuracy_)  # 将平均准确率数据保存到accuracy数组中
    lrArr.append(lr) # 将学习率数据保存到lr数组中

    print("Epoch: ", int((i - 1) / 50), "|", "Learning Rate = %.7f" % lr, "|", "Accuracy = %.4f" % accuracy_)

plt.plot(lrArr, accuracyArr)
plt.grid(True) # 增加网格点
plt.title('Accuracy - Learning Rate Graph') # 设置图表标题
plt.xlabel('Learning Rate') # 横坐标标题
plt.ylabel('Accuracy') # 纵坐标标题
plt.axis('tight') # 坐标轴紧密排布
plt.xticks(np.arange(0, 0.0005, 0.0001)) # 设置x轴刻度
plt.show()


