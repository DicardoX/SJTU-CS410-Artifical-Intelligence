import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
import math

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from utils import accuracy, normalize, encode_onehot, load_data
from models import GCN


# 处理并加载数据集
print("Data loading and preprocessing...")
test_graphs, test_features = load_data('test')
filefolder = 'test'
filefolder = './data/' + filefolder
data = np.load(os.path.abspath(filefolder + '/names_onehots.npy'), allow_pickle=True).item()
test_names = data['names']


# 超参数定义
hiddenLayerNum = 512
BatchSize = 64

# 模型加载
model = th.load('./model/model.pkl')

# 模型预测
def predict():
    prediction = []
    result = []
    samepleNum = test_features.shape[0]
    for i in range (0, samepleNum, BatchSize):
        output = model(th.FloatTensor(test_features[i: i + BatchSize]), th.FloatTensor(test_graphs[i: i + BatchSize]))
        output_res = output.detach().numpy()[:, 1]
        output_res = np.array(output_res)
        #pred = output[:, 1]
        #print(output.shape)

        prediction.extend(list(output_res))
        #prediction.append(u)
        #result.append(np.exp(u))


    #for i in range(0, len(prediction), 1):
    #    prediction[i] = list(prediction[i])[1]
        #print(prediction[i])
    #    predNumber = prediction[i].item()
    #    predNumber = math.exp(predNumber)
    #    result.append(predNumber)
    #print(result)
    #print(prediction)
    print(prediction)
    f = open('output_518021910698.csv', 'w')
    f.write('Chemical,Label\n')
    for i, v in enumerate(prediction):
        f.write(test_names[i] + ',%f\n' % v)
    f.close()

predict()
