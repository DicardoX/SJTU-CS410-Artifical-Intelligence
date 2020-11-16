import os
import numpy as np
import tensorflow as tf
import time
import pandas as pd
import matplotlib.pyplot as plt

LR = 0.0001
BATCHSIZE = 64
EPOCH = 20

## Data Loading ##
def load_data(filefolder):
    filefolder = './data/' + filefolder
    data = np.load(os.path.abspath(filefolder + '/names_onehots.npy'), allow_pickle=True).item()  # allow_pickle: 可选，布尔值，允许使用 Python pickles 保存对象数组；data为dict字典类型
    data = data['onehots']
    label = pd.read_csv(os.path.abspath(filefolder + '/names_labels.txt'), sep=',')
    label = label['Label'].values
    return data, label

## Network Building ##
def net(onehots_shape, lamda=0.5):  # [73,398]
    onehots_shape = list(onehots_shape)
    input = tf.placeholder(tf.float32, [None] + onehots_shape, name='input')  # tf.placeholder(): TensorFlow中的占位符，用于传入外部数据
    input = tf.reshape(input, [-1] + onehots_shape + [1])  # tf.reshape(tensor, shape, name=None): 函数的作用是将tensor变换为参数shape的形式。
    label = tf.placeholder(tf.int32, [None], name='label')
    label = tf.one_hot(label, 2)  # 该函数的功能主要是将indices（label）转换成one_hot类型的张量输出，2是depth，表示张量的尺寸，indices中元素默认不超过（depth-1），如果超过，输出为[0,0,···,0]

    CONV_NUM = 32
    conv1 = tf.keras.layers.Conv2D(CONV_NUM, 5, 1, padding='same', activation=tf.nn.relu)(input)  # 卷积层一，padding表示填充方式，activation表示激活函数，输入为input
    pool1 = tf.keras.layers.MaxPool2D(2, 2)(conv1)  # 池化层一，kernel_size = 2，表示窗口大小，stride = 2，表示步长（通常与kernel_size相等），输入为conv1
    conv2 = tf.keras.layers.Conv2D(CONV_NUM, 3, (1, 2), padding='same', activation=tf.nn.relu)(pool1)  # 卷积层二
    pool2 = tf.keras.layers.MaxPool2D(2, 2)(conv2)  # 池化层二
    #conv3 = tf.keras.layers.Conv2D(CONV_NUM, 3, 3, padding='same', activation=tf.nn.relu)(pool2)  # 卷积层三
    #conv4 = tf.keras.layers.Conv2D(CONV_NUM, 3, 3, padding='same', activation=tf.nn.relu)(conv3)  # 卷积层四
    #pool3 = tf.keras.layers.MaxPool2D(2, 2)(conv4)  # 池化层三

    flat = tf.reshape(pool2, [-1, 18 * 50 * CONV_NUM])  # 将pool2改变为每“行”18*50*32的shape，-1表示根据原大小推断行数
    #output1 = tf.keras.layers.Dense(16, name='output1')(flat)  # tf.keras.layers.Dense()相当于在全连接层中添加一个层
    #output2 = tf.keras.layers.Dense(16, name='output2')(output1)
    output = tf.keras.layers.Dense(2, name='output')(flat)

    #prediction = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=output)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=label, logits=output, loss_collection=tf.contrib.layers.l2_regularizer(lamda))
    optimizer = tf.train.AdamOptimizer(LR).minimize(loss)  # 建立网络中训练的节点并利用Adam算法进行最优化，即最小化loss，LR为learning rate
    accuracy = tf.metrics.accuracy(labels=tf.argmax(label, axis=1), predictions=tf.argmax(output, axis=1), )[1]  # 计算accuracy
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())  # tf.group()类似于批处理操作，将一个或多个语句变成操作

    return init, optimizer, loss, accuracy


## Model Training ##
def train_model(BatchSize=BATCHSIZE, lr_=LR):
    #tf.reset_default_graph()  # tf.reset_default_graph函数用于清除默认图形堆栈并重置全局默认图形。
    tf.reset_default_graph()

    train_X, train_Y = load_data('train')
    valid_X, valid_Y = load_data('validation')
    init, optimizer, loss, accuracy = net(train_X.shape[1:], 0.5)  # [1:]表示从第一位开始索引，默认到最后一位，否则需要指定
    sess = tf.Session()  # Session 是 Tensorflow 为了控制,和输出文件的执行的语句. 运行 session.run() 可以获得你要得知的运算结果, 或者是你所要运算的部分.
    sess.run(init)  # 运行传递来的init_op，初始化全局和局部变量
    saver = tf.train.Saver()

    print("Training...")

    train_size = train_X.shape[0]
    accuracyArr = []
    for epoch in range(EPOCH):
        time1 = 0
        if epoch < 1:
            time1 = time.time()

        for i in range(0, train_size, BatchSize):  # 即i从0开始，到train_size，每次增加BatchSize
            b_X, b_Y = train_X[i: i + BatchSize], train_Y[i: i + BatchSize]
            _, loss_ = sess.run([optimizer, loss], {'input:0': b_X, 'label:0': b_Y})

        if epoch % 1 == 0:
            accuracy_ = 0
            for i in range(0, valid_X.shape[0], BatchSize):
                b_X, b_Y = valid_X[i: i + BatchSize], valid_Y[i: i + BatchSize]
                accuracy_ += sess.run(accuracy, {'input:0': b_X, 'label:0': b_Y})
            accuracy_ = accuracy_ * BatchSize / (valid_X.shape[0] + BatchSize)  # 为了将train accuracy的均值限制在[0, 1]之间
            print('Epoch:', epoch, '| Train Loss: %.4f' % loss_, " | ",  'Current Validation Accuracy: %.6f' % accuracy_)
            accuracyArr.append(accuracy_)

        if epoch % 10 == 0:
            epoch = epoch / 1.2

        # 第一代输出预计时间
        if epoch < 1:
            dt = round(time.time() - time1, 2)
            print("Epoch Time(sec): ", dt)
            print("Estimated Time(min): ", round(dt * EPOCH / 60, 1))

    plt.plot(accuracyArr)
    plt.grid(True)  # 增加网格点
    plt.title('Accuracy Graph')  # 设置图表标题
    plt.xlabel('Epoch')  # 横坐标标题
    plt.ylabel('Accuracy')  # 纵坐标标题
    plt.axis('tight')  # 坐标轴紧密排布
    plt.show()

    accuracy_ = 0
    b_data, b_label = valid_X, valid_Y
    accuracy_ = sess.run(accuracy, {'input:0': b_data, 'label:0': b_label})
    print("Validation Accuracy: %.8f" % accuracy_)

    saver.save(sess, './weights/model')
    sess.close()

    return accuracy_, accuracyArr


## Train Model ##
(accuracy, accuracyArr) = train_model(BATCHSIZE, LR)


# 以下部分是为了局部最优化BatchSize

'''
max_accuracy = 0
max_variance = -100000
curRound = 0
for size in range(64, 128, 16):
    curRound = curRound + 1
    print("------------------------")
    print('Round: ', curRound, '| BatchSize = ', size)
    (accuracy, accuracyArr) = train_model(size, LR)
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        size1 = size
    relativeAccuracySum = 0.00
    for i in range(0, accuracyArr.size() - 2, 1):
        if accuracyArr[i + 1] > accuracyArr[i]:
            relativeAccuracySum += pow(accuracyArr[i + 1] - accuracyArr[i], 2)
        else:
            relativeAccuracySum -= pow(accuracyArr[i] - accuracyArr[i + 1], 2)
    variance = float(relativeAccuracySum / float(accuracyArr.size()))
    if variance > min_variance:
        min_variance = variance
        size2 = size
print('Max Accuracy: ', max_accuracy, 'BatchSize: ', size1)
print('Max Variance: ', max_variance, 'BatchSize: ', size2)
'''
