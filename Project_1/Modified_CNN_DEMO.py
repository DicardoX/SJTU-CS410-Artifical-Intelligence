############################# Obtain File List ##########################################

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



############################# Environment Settings ##########################################

!pip install tensorflow==1.13.1 # 在使用pip指令前要加!

import os
import numpy as np
# import pandas as pd
import tensorflow as tf

LR = 0.01
BatchSize = 50
EPOCH = 2
print(tf.__version__)


############################# Load Data ##########################################

def load_data(filefolder):
    ori_filefolder = filefolder
    filefolder = '../input/cs410-2020-fall-ai-project-1/data/' + filefolder 
    
    data = np.load(os.path.abspath(filefolder + '/names_onehots.npy'), allow_pickle=True).item() # allow_pickle: 可选，布尔值，允许使用 Python pickles 保存对象数组；data为dict字典类型
    data = data['onehots']
    if ori_filefolder == 'test':
        label_filename = filefolder + '/output_sample.txt' 
        #label = pd.read_csv(os.path.abspath(filefolder + '/output_sample.csv'), sep=',')
    else:
        label_filename = filefolder + '/names_labels.txt'
        #label = pd.read_csv(os.path.abspath(filefolder + '/names_labels.csv'), sep=',')
    label = []
    
    with open(label_filename, 'r') as f: # with open() 用来打开本地文件
        header = f.readline().replace('\n', '').split(',')
        if header[0] == 'Label':
            label_index = 0
        else:
            label_index = 1
        for line in f.readlines():
            line = line.replace('\n', '').split(',')
            label.append((line[label_index])) # 本来是int(line[label_index])
    label = np.array(label)
    # label = label['Label'].values
    return data, label

  
########### Test array output ###############
data, label = load_data('test')
print(data.shape)



######################## Network Building #############################

def net(onehots_shape): #[73,398]
    if not isinstance(onehots_shape, list):
        onehots_shape = list(onehots_shape)
    input = tf.placeholder(tf.float32, [None] + oneshots_shape, name='input') # tf.placeholder(): TensorFlow中的占位符，用于传入外部数据
    input = tf.reshape(tensor=input, shape=[-1] + onehots_shape + [1]) # tf.reshape(tensor, shape, name=None): 函数的作用是将tensor变换为参数shape的形式。
    #input = tf.reshape(input, [None, 73, 398, 1])
    label = tf.placeholder(tf.int32, [None], name='label')
    label = tf.one_hot(label, 2) # 该函数的功能主要是将indices（label）转换成one_hot类型的张量输出，2是depth，表示张量的尺寸，indices中元素默认不超过（depth-1），如果超过，输出为[0,0,···,0]
    
    conv1 = tf.keras.layers.Conv2D(32, 5, 1, padding='same', activation=tf.nn.relu)(input) # 卷积层一，padding表示填充方式，activation表示激活函数，输入为input
    pool1 = tf.keras.layers.MaxPool2D(2, 2)(conv1) # 池化层一，kernel_size = 2，表示窗口大小，stride = 2，表示步长（通常与kernel_size相等），输入为conv1
    
    conv2 = tf.keras.layers.Conv2D(32, 3, (1, 2), padding='same', activation=tf.nn.relu)(pool1) # 卷积层二
    pool2 = tf.keras.layers.MaxPool2D(2, 2)(conv2) # 池化层二
    
    flat = tf.reshape(tensor=pool2, shape=[-1, 18*50*32]) # 将pool2改变为每“行”18*50*32的shape，-1表示根据原大小推断行数
    output = tf.keras.layers.Dense(2, name='Output')(flat) # tf.keras.layers.Dense()相当于在全连接层中添加一个层
    
    loss = tf.losses.softmax_cross_entropy(onehot_labels=label, logits=output) # 计算cost
    train_op = tf.train.AdamOptimizer(LR).minimize(loss) # 建立网络中训练的节点并利用Adam算法进行最优化，即最小化loss
    accuracy = tf.metrics.accuracy(labels=tf.argmax(label, axis=1), predictions=tf.argmax(output, axis=1), )[1]
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) #
    
    return init_op, train_op, loss, accuracy
        
        
        
        


    



