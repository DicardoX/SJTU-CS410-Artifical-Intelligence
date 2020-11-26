import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

lr = 3e-4 # Best learning rate for Adam optimizer
#lr = 0.001
#dropout_rate = 0.3                 # Dropout rate
dropout_rate = tf.placeholder_with_default(0.0, shape=())
regularization_coeff = 0.5          # Regularization coefficient
EPOCH = 1000 # epoch
BATCHSIZE = 32 # batch size
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
        channelNum = 32
        regularization = tf.contrib.layers.l2_regularizer(regularization_coeff)
        super(Network, self).__init__() # 对继承的父类对象进行初始化，将MyModel类型的对象self转化成tf.keras.Model类型的对象，然后调用其__init__()函数
        #self.dropout = tf.keras.layers.Dropout # 随机丢弃层
        self.conv1 = tf.keras.layers.Conv2D(channelNum, 5, 1, padding='same', activation=tf.nn.relu, kernel_regularizer=regularization)  # 卷积层一
        self.pool1 = tf.keras.layers.MaxPool2D(2, 2) # 池化层一
        self.conv2 = tf.keras.layers.Conv2D(channelNum, 3, (1, 2), padding='same', activation=tf.nn.relu, kernel_regularizer=regularization)  # 卷积层二
        self.pool2 = tf.keras.layers.MaxPool2D(2, 2) #池化层二
        self.fc = tf.keras.layers.Dense(2) #全连接层

    def call(self, inputs, training=None, mask=None):
        conv1Res = self.conv1(inputs)
        #dropoutRes = self.dropout(conv1Res, training=True)
        #pool1Res = self.pool1(dropoutRes)
        #bnRes = tf.layers.batch_normalization(inputs=conv1Res, training=True)  # Batch Normalization层
        pool1Res = self.pool1(conv1Res)
        conv2Res = self.conv2(pool1Res)
        pool2Res = self.pool2(conv2Res)
        flat = tf.reshape(pool2Res, [-1, 18*50*32])
        output = self.fc(flat)
        output = tf.keras.layers.Dropout(rate=dropout_rate)(output)
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
#myModel.summary() # 展示模型结构

loss = tf.keras.losses.BinaryCrossentropy()(label2D, output) # 计算损失函数，使用二维one_hot类型的label

#update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#with tf.control_dependencies(update_ops):
optimizer = tf.train.AdamOptimizer(lr).minimize(loss) # 定义优化器

prediction = tf.placeholder(tf.float64, [None], name='prediction') # 定义预测占位符
_,aucScore = tf.metrics.auc(labels=label, predictions=prediction) # 计算AUC score，使用shape为(?, )，int32类型的label

## Running ##
def run_model():
    global lr, loss_, dropout_rate
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init)

    saver = tf.train.Saver() # 创建三个数据文件和一个检查点（Checkpoint）文件，保存模型
    sampleNum = train_X.shape[0]  # 样本数
    #bestAuc = 0 # 记录最佳AUC Score
    worseCount = 0
    bestValidLoss = 1 # 记录最低验证集loss
    print("Model Training...")
    for epoch in range(EPOCH):
        for i in range(0, sampleNum, BATCHSIZE):
            b_X, b_Y = train_X[i: i + BATCHSIZE], train_Y[i: i + BATCHSIZE]
            feed_dict = {'input:0': b_X, 'label:0': b_Y, dropout_rate: 0.3}
            _, loss_ = sess.run([optimizer, loss], feed_dict=feed_dict)
        print("Epoch:", epoch, '|', "Train loss:", loss_)

        #if epoch % 2 == 0:                      Adam优化器可以自动进行学习率的衰减更新，不需要人为更新
        #    lr /= 2

        if epoch % 1 == 0:
            valPrediction = sess.run(output, {'input:0': valid_X, dropout_rate: 0.0})
            valPrediction = valPrediction[:, 1] # 在全部数组（维）中取第1个数据，及所有集合的第1个数据
            feed_dict = {prediction: valPrediction, label: valid_Y, dropout_rate: 0.0}
            aucScore_ = sess.run(aucScore, feed_dict=feed_dict) # 运行计算auc score
            feed_dict = {'input:0': valid_X, 'label:0': valid_Y, dropout_rate: 0.0}
            valid_loss_ = sess.run(loss, feed_dict=feed_dict)
            aucArr.append(aucScore_) # 将当前auc score存入数组
            print("------------------------------------------------------------")
            print("Valid loss", valid_loss_, '|', "AUC Score in validation:", aucScore_)
            print("------------------------------------------------------------")

            if valid_loss_ > bestValidLoss:
                worseCount = worseCount + 1
            else:
                worseCount = 0

            if worseCount >= 5:         # 验证集loss下降一次，学习率减半
                lr = lr / 2

            if worseCount >= 20:         # 验证集loss连续下降两次，early stop
                return

            #if aucScore_ > bestAuc:
            #    saver.save(sess, "./weights/model")
            #    bestAuc = aucScore_
            if valid_loss_ < bestValidLoss:
                bestValidLoss = valid_loss_ # 更新最低验证集loss
                #bestAuc = aucScore_ # 更新最佳auc score
                saver.save(sess, "./weights/model") # 保证保存的结果是过拟合发生之前

            #if aucScore_ > bestAuc:
            #    bestAuc = aucScore_  # 更新最佳auc score
            #    saver.save(sess, "./weights2/model")


run_model()

plt.plot(aucArr)
plt.grid(True)  # 增加网格点
plt.title('AUC Curve')  # 设置图表标题
plt.xlabel('Epoch')  # 横坐标标题
plt.ylabel('AUC Score')  # 纵坐标标题
plt.axis('tight')  # 坐标轴紧密排布
plt.show()
