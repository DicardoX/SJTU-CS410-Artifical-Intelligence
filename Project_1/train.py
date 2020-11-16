import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import time

LR = 0.001 # Learning Rate
EPOCH = 10 # epoch
BATCHSIZE = 96 # batch size
aucArr = [] # 存储AUC Score的数组

## Data Loading ##
def load_data(filefolder):
    filefolder = './data/' + filefolder
    data = np.load(os.path.abspath(filefolder + '/names_onehots.npy'), allow_pickle=True).item()  # allow_pickle: 可选，布尔值，允许使用 Python pickles 保存对象数组；data为dict字典类型
    data = data['onehots'] # 查询规格：data.shape[0 ~ 2]，分别是高度（sampleNum）、宽度（特征数）、通道数
    label = pd.read_csv(os.path.abspath(filefolder + '/names_labels.txt'), sep=',')
    label = label['Label'].values
    return data, label

## Data Set ##
train_X, train_Y = load_data('train') # 训练集，[8169，73, 398]
valid_X, valid_Y = load_data('validation') # 验证集，[272, 73, 398]

## Neuron Network Building ##
class Network(tf.keras.Model):
    def __init__(self):
        super(Network, self).__init__() # 对继承的父类对象进行初始化，将MyModel类型的对象self转化成tf.keras.Model类型的对象，然后调用其__init__()函数
        self.conv1 = tf.keras.layers.Conv2D(32, 5, 1, padding='same', activation=tf.nn.relu)  # 卷积层一
        self.pool1 = tf.keras.layers.MaxPool2D(2, 2) # 池化层一
        #self.conv2 = tf.keras.layers.Conv2D(32, 3, (1, 2), padding='same', activation=tf.nn.relu) # 卷积层二
        self.conv2 = tf.keras.layers.Conv2D(32, 3, (1, 2), padding='same', activation=tf.nn.relu)  # 卷积层二
        self.pool2 = tf.keras.layers.MaxPool2D(2, 2) #池化层二
        self.fc = tf.keras.layers.Dense(2) #全连接层

    def call(self, inputs, training=None, mask=None):
        conv1Res = self.conv1(inputs)
        pool1Res = self.pool1(conv1Res)
        conv2Res = self.conv2(pool1Res)
        pool2Res = self.pool2(conv2Res)
        flat = tf.reshape(pool2Res, [-1, 18*50*32])
        output = self.fc(flat)
        output = tf.nn.softmax(output)
        return output

## Model Building ##
myModel = Network()
onehots_shape = list(train_X.shape[1:]) # 将除第一个元素外的其他元素转化为列表（list）
#input、label占位符定义及reshape
input = tf.placeholder(tf.float32, [None] + onehots_shape, name='input') # 将onehots_shape列表的首元素加为None
input = tf.reshape(input, [-1] + onehots_shape + [1]) # 将input规范成[73, 398, 1]的形式，与data的宽度及通道数对应
label = tf.placeholder(tf.int32, [None], name='label') # Tensor("label:0", shape=(?,), dtype=int32)
label2D = tf.one_hot(label, 2) # 将label矩阵转化成one_hot编码类型的label2D，维数为2，及有两种不同的编码，分别对应label的0和1，Tensor("one_hot:0", shape=(?,2), dtype=float32)

output = myModel(input) # 利用input获取output
myModel.summary() # 展示模型结构

#bce = tf.keras.losses.BinaryCrossentropy()
#loss = bce(label, output)
loss = tf.keras.losses.BinaryCrossentropy()(label2D, output) # 计算损失函数，使用二维one_hot类型的label
optimizer = tf.train.AdamOptimizer(LR).minimize(loss) # 定义优化器
prediction = tf.placeholder(tf.float64, [None], name='prediction') # 定义预测占位符
_,aucScore = tf.metrics.auc(labels=label, predictions=prediction) # 计算AUC score，使用shape为(?, )，int32类型的label

## Running ##
def run_model():
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init)

    saver = tf.train.Saver() # 创建三个数据文件和一个检查点（Checkpoint）文件，保存模型
    sampleNum = train_X.shape[0]  # 样本数
    bestAuc = 0 # 记录最佳AUC Score
    print("Model Training...")
    for epoch in range(EPOCH):
        for i in range(0, sampleNum, BATCHSIZE):
            b_X, b_Y = train_X[i: i + BATCHSIZE], train_Y[i: i + BATCHSIZE]
            feed_dict = {'input:0': b_X, 'label:0': b_Y}
            _, loss_ = sess.run([optimizer, loss], feed_dict=feed_dict)

        if epoch % 1 == 0:
            valPrediction = sess.run(output, {'input:0': valid_X})
            valPrediction = valPrediction[:, 1] # 在全部数组（维）中取第1个数据，及所有集合的第1个数据
            feed_dict = {prediction: valPrediction, label: valid_Y}
            aucScore_ = sess.run(aucScore, feed_dict=feed_dict) # 运行计算auc score
            aucArr.append(aucScore_) # 将当前auc score存入数组
            print("Epoch:", epoch, '|', "AUC Score:", aucScore_)
            if aucScore_ > bestAuc:
                saver.save(sess, "./weights/model")
                bestAuc = aucScore_

        if epoch % 2 == 0:
            epoch /= 1.2


run_model()

plt.plot(aucArr)
plt.grid(True)  # 增加网格点
plt.title('AUC Curve')  # 设置图表标题
plt.xlabel('Epoch')  # 横坐标标题
plt.ylabel('AUC Score')  # 纵坐标标题
plt.axis('tight')  # 坐标轴紧密排布
plt.show()

