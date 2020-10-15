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





    



