import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import numpy as np
import os
import pandas as pd
import networkx as nx
import pysmiles

## Step 2: 定义GCN的message函数和reduce函数
# 每个节点发送Embedding的时候不需要任何处理，所以通过内置的copy_src实现，out='m'表示发送到目标节点后目标节点的mailbox用'm'来标示这个消息是源节点的Embedding。
from pysmiles import read_smiles

gcn_msg = fn.copy_src(src='h', out='m')
# 目标节点的reduce函数很简单，因为按照GCN的数学定义，邻接矩阵和特征矩阵相乘，目标节点的reduce只需要通过sum将接收到的message相加
gcn_reduce = fn.sum(msg='m', out='h')

## Step 3: 定义一个应用于节点的node UDF(user defined function)，即定义一个全连接层来对中间节点表示h^{hat}_i进行线性变换，再利用非线性函数进行计算
class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}

## Step 4: 定义GCN的Embedding更新层，以实现在所有节点上进行消息传递，并利用NodeApplyModule对节点信息进行计算更新
class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

## Step 5: 定义一个包含两个GCN层的图神经网络分类器，通过向该分类器输入特征大小为1433的训练样本，以获得该样本所属的类别编号
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(1433, 16, F.relu)
        self.gcn2 = GCN(16, 7, F.relu)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x

def load_data(filefolder):
    ## 从names_smiles.txt文件中读入数据
    filefolder = './data/' + filefolder
    fopen = open(filefolder + "/names_smiles.txt", "r")
    List_row = fopen.readlines()
    data = []
    for i in range(1, len(List_row), 1):
        colume_list = List_row[i].strip().split(",")
        data.append(colume_list)

    ## 读取分子式并转化为邻接矩阵
    matrix_list = []
    for i in range(0, len(data), 1):
        mol = read_smiles(data[i][1])   # 读取分子式
        adjacency_matrix = nx.to_numpy_matrix(mol)  # 将分子式转化为邻接矩阵
        matrix_list.append(adjacency_matrix)    # 存储到list中



    #print(data)

    """
    data = np.load(os.path.abspath(filefolder + '/names_onehots.npy'),
                   allow_pickle=True).item()  # allow_pickle: 可选，布尔值，允许使用 Python pickles 保存对象数组；data为dict字典类型
    print(data)
    data = data['onehots']  # 查询规格：data.shape[0 ~ 2]，分别是高度（sampleNum）、宽度（特征数）、通道数
    #print(data)
    label = pd.read_csv(os.path.abspath(filefolder + '/names_labels.txt'), sep=',')
    label = label['Label'].values
    return data, label
    """

load_data('train')
