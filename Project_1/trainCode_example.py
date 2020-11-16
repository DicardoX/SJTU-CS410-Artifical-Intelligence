import os
import numpy as np
import tensorflow as tf
import time
import pandas as pd
import matplotlib.pyplot as plt

## Data Loading ##
def load_data(filefolder):
    filefolder = './data/' + filefolder
    data = np.load(os.path.abspath(filefolder + '/names_onehots.npy'), allow_pickle=True).item()  # allow_pickle: 可选，布尔值，允许使用 Python pickles 保存对象数组；data为dict字典类型
    label = pd.read_csv(os.path.abspath(filefolder + '/names_labels.txt'), sep=',')
    data = data['onehots']
    label = label['Label'].values
    return data, label

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__() # 对继承的父类对象进行初始化，将MyModel类型的对象self转化成tf.keras.Model类型的对象，然后调用其__init__()函数
        self.conv1_layer = tf.keras.layers.Conv2D(32, 5, 1, 'same', activation=tf.nn.relu) #
        self.pool1_layer = tf.keras.layers.MaxPool2D(2, 2)
        self.conv2_layer = tf.keras.layers.Conv2D(32, 3, (1, 2), 'same', activation=tf.nn.relu)
        self.pool2_layer = tf.keras.layers.MaxPool2D(2, 2)
        # flat
        self.FCN = tf.keras.layers.Dense(2)
        # softmax

    def call(self, inputs):
        x = self.conv1_layer(inputs)
        x = self.pool1_layer(x)
        x = self.conv2_layer(x)
        x = self.pool2_layer(x)
        flat = tf.reshape(x, [-1, 18*50*32])
        output = self.FCN(flat)
        output_with_sm = tf.nn.softmax(output)
        return output, output_with_sm

if __name__ == '__main__':
    # parameters
    LR = 0.001
    BatchSize = 128
    EPOCH = 10

    # data
    train_x, train_y = load_data('train')
    valid_x, valid_y = load_data('validation')

    # model & input and output of model
    model = MyModel()

    onehots_shape = list(train_x.shape[1:])
    input_place_holder = tf.placeholder(tf.float32, [None] + onehots_shape, name='input')
    input_place_holder_reshaped = tf.reshape(input_place_holder, [-1] + onehots_shape + [1])
    label_place_holder = tf.placeholder(tf.int32, [None], name='label')
    label_place_holder_2d = tf.one_hot(label_place_holder, 2)
    output, output_with_sm = model(input_place_holder_reshaped)
    model.summary()  # show model's structure

    # loss
    bce = tf.keras.losses.BinaryCrossentropy()  # compute cost
    loss = bce(label_place_holder_2d, output_with_sm)

    # Optimizer
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)

    # auc
    prediction_place_holder = tf.placeholder(tf.float64, [None], name='pred')
    auc, update_op = tf.metrics.auc(labels=label_place_holder, predictions=prediction_place_holder)

    aucArr = []

    # run
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        saver = tf.train.Saver()

        train_size = train_x.shape[0]
        best_val_auc = 0
        for epoch in range(EPOCH):
            print("Training for Epoch", epoch, "...")
            for i in range(0, train_size, BatchSize):
                b_x, b_y = train_x[i:i + BatchSize], train_y[i:i + BatchSize]
                _, loss_ = sess.run([train_op, loss], {'input:0': b_x, 'label:0': b_y})
                #print("Epoch {}: [{}/{}], training set loss: {:.4}".format(epoch, i, train_size, loss_))

            if epoch % 1 == 0:
                val_prediction = sess.run(output_with_sm, {'input:0': valid_x})
                val_prediction = val_prediction[:, 1]
                auc_value = sess.run(update_op, feed_dict={prediction_place_holder: val_prediction, label_place_holder: valid_y})
                aucArr.append(auc_value)
                print("auc_value", auc_value)
                if auc_value > best_val_auc:
                    saver.save(sess, './weights/model')

            if epoch % 2 == 0:
                epoch = epoch / 1.2

        plt.plot(aucArr)
        plt.grid(True)  # 增加网格点
        plt.title('AUC Curve')  # 设置图表标题
        plt.xlabel('Epoch')  # 横坐标标题
        plt.ylabel('AUC Score')  # 纵坐标标题
        plt.axis('tight')  # 坐标轴紧密排布
        plt.show()
