import os
import tensorflow as tf
import numpy as np
import pandas as pd
import random
dropout_prob = 0.1
BatchSize = 64              # 对预测结果无影响

def load_test_data(filefolder):
    filefolder = './data/' + filefolder
    data = np.load(os.path.abspath(filefolder + '/names_onehots.npy'), allow_pickle=True).item()
    onehots = data['onehots']
    name = data['names']
    return onehots, name

## Neuron Network Building ##
class Network(tf.keras.Model):
    def __init__(self):
        #regularization = tf.contrib.layers.l2_regularizer(regularization_coeff)
        super(Network, self).__init__() # 对继承的父类对象进行初始化，将MyModel类型的对象self转化成tf.keras.Model类型的对象，然后调用其__init__()函数
        self.conv1 = tf.keras.layers.Conv2D(24, 5, 1, padding='same', activation=tf.nn.relu)  # 卷积层一
        self.pool1 = tf.keras.layers.MaxPool2D(2, 2) # 池化层一
        self.conv2 = tf.keras.layers.Conv2D(32, 3, (1, 2), padding='same', activation=tf.nn.relu)  # 卷积层二
        self.pool2 = tf.keras.layers.MaxPool2D(2, 2) #池化层二
        self.conv3 = tf.keras.layers.Conv2D(48, 3, 1, padding='same', activation=tf.nn.relu)
        self.pool3 = tf.keras.layers.MaxPool2D(2, 2)
        self.fc = tf.keras.layers.Dense(2) #全连接层

    def call(self, inputs, training=None, mask=None):
        conv1Res = self.conv1(inputs)   # input是[72*398]
        #bnRes = tf.layers.batch_normalization(inputs=conv1Res, training=True)  # Batch Normalization层
        pool1Res = self.pool1(conv1Res)
        conv2Res = self.conv2(pool1Res)
        pool2Res = self.pool2(conv2Res)
        conv3Res = self.conv3(pool2Res)
        pool3Res = self.pool3(conv3Res)
        flat = tf.reshape(pool3Res, [-1, 9*25*48])
        output = self.fc(flat)
        #output = tf.keras.layers.Dropout(rate=dropout_rate)(output)
        output = tf.nn.softmax(output)
        return output

# data
test_data, test_name = load_test_data("test")

# model
model = Network()
input = tf.placeholder(tf.float32, [None] + list(test_data.shape[1:]), name='input')
input = tf.reshape(input, [-1] + list(test_data.shape[1:]) + [1])
output = model(input)

# Predict on the test set
def predict():
    sampleNum = test_data.shape[0] # 样本数
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, os.path.abspath('./weights/model')) # 从最佳权重中恢复，并进行prediction
    prediction = []

    for i in range(0, sampleNum, BatchSize):
        testOutput = sess.run(output, {'input:0': test_data[i:i + BatchSize]})
        pred = testOutput[:, 1]
        prediction.extend(list(pred))
    sess.close()
    f = open('output_518021910698.csv', 'w')
    f.write('Chemical,Label\n')
    for i, v in enumerate(prediction):
        f.write(test_name[i] + ',%f\n' % v)
    f.close()

predict()
