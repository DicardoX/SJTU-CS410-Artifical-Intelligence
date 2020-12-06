import time
import numpy as np
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import matplotlib.pyplot as plt
from smilesToGraph import load_data
import math
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from utils import accuracy, normalize, encode_onehot

gcn_msg = fn.copy_src(src="h", out="m")
gcn_reduce = fn.sum(msg="m", out="h")  # 聚合邻居节点的特征

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(th.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(th.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        #self.weight = th.tensor(self.weight, dtype=th.float32)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #input = th.tensor(input, dtype=th.float32)
        #input = np.asmatrix(input)
        support = th.matmul(input, self.weight)
        #output = th.spmm(adj, support)
        output = th.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, featureNum, hiddenLayerNum, classNum, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(featureNum, hiddenLayerNum)
        self.gc2 = GraphConvolution(hiddenLayerNum, classNum)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

'''
# 定义节点的UDF apply_nodes  他是一个完全连接层
class NodeApplyModule(nn.Module):
    # 初始化
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    # 前向传播
    def forward(self, node):
        h = self.linear(node.data["h"])
        if self.activation is not None:
            h = self.activation(h)
        return {"h": h}


# 定义GCN模块  GCN模块的本质是在所有节点上执行消息传递  然后再调用NOdeApplyModule全连接层
class GCN(nn.Module):
    # 初始化
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        # 调用全连接层模块
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    # 前向传播
    def forward(self, g, feature):
        g.ndata["h"] = feature  # feature应该对应的整个图的特征矩阵
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)  # 将更新操作应用到节点上

        return g.ndata.pop("h")


# 利用cora数据集搭建网络然后训练
class Net(nn.Module):
    # 初始化网络参数
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(1433, 16, F.relu)  # 第一层GCN
        self.gcn2 = GCN(16, 7, None)

    # 前向传播
    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x
'''

# 处理并加载数据集
train_graphs, train_features, train_labels = load_data('train')
valid_graphs, valid_features, valid_labels = load_data('validation')
train_labels = encode_onehot(train_labels)
valid_labels = encode_onehot(valid_labels)
#train_features = th.FloatTensor(train_features)
train_labels = th.LongTensor(train_labels)


#print(train_features.dtype)

#train_features = normalize(train_features)
#train_features = th.FloatTensor(train_features)
#valid_features = th.FloatTensor(valid_features)

#nx.draw(g.to_networkx(), node_size=50, with_labels=True)
#plt.show()

hiddenLayerNum = 16
classNum = 2
dropout_rate = 0.5
lr = 1e-3
weight_decay = 5e-4
EPOCH = 10
BatchSize = 1

# 模型构造
#net = Net()
#print(train_features[0].shape[1])
model = GCN(featureNum=train_features[0].shape[1],      # 132
            hiddenLayerNum=hiddenLayerNum,
            classNum=classNum,
            dropout=dropout_rate)
# 定义优化器
optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# 模型评估
def validation():
    model.eval()
    output = model(valid_features, valid_graphs)
    valid_loss = F.nll_loss(output, valid_labels)
    valid_acc = accuracy(output, valid_labels)
    #Print("Test set results:",
    #     "loss= {:.4f}".format(loss_test.item()),
    #      "accuracy= {:.4f}".format(acc_test.item()))
    print("Validation loss:", valid_loss.item(), '|', "Validation accuracy:", valid_acc.item())

'''
def evaluate(model, graphs, features, labels):
    model.eval()                        # 会通知所有图层您处于评估模式
    with th.no_grad():
        logits = model(graphs, features)
        #logits = logits[mask]
        #labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
'''

def train(epoch):
    avgTrainLoss = 0.0
    avgTrainAcc = 0.0
    samepleNum = train_features.shape[0]
    for i in range(0, samepleNum, BatchSize):
        #idx_train = range(int(i), int(i + BatchSize))
        #idx_train = range(i, i + BatchSize, 1)
        #idx_train = th.IntTensor(idx_train)
        #print(idx_train.dtype)
        #idx_train = th.cat(idx_train, dim=0)
        #b_X, b_Y, b_label= train_graphs[idx_train], train_features[idx_train], train_labels[idx_train]

        #output = model(train_features[i], train_graphs[i])
        #train_loss = F.nll_loss(output, train_labels[i])
        #train_acc = accuracy(output, train_labels[i])
        t = time.time()
        model.train()
        optimizer.zero_grad()
        myOutput = model(th.FloatTensor(train_features[i: i + BatchSize]), th.FloatTensor(train_graphs[i: i + BatchSize]))
        train_loss = F.nll_loss(myOutput, train_labels[i: i + BatchSize])
        train_acc = accuracy(myOutput, train_labels[i: i + BatchSize])
        train_loss.backward()
        optimizer.step()


    #idx_train = range(0, len(train_features))
    #output = model(train_features, train_graphs)
    #train_loss = F.nll_loss(output[idx_train], train_labels[idx_train])
    #train_acc = accuracy(output[idx_train], train_labels[idx_train])
    #train_loss.backward()
    #optimizer.step()
        avgTrainLoss += train_loss.item()
        avgTrainAcc += train_acc.item()

    avgTrainLoss /= train_features.shape[0]
    avgTrainAcc /= train_features.shape[0]
    print("Epoch:", epoch, '|', "Train loss:", avgTrainLoss, '|', "Train accuracy:", avgTrainAcc)
    #print("Epoch:", epoch, '|', "Train loss:", train_loss.item(), '|', "Train accuracy:", train_acc.item())

#dur = []  # 时间

for epoch in range(EPOCH):
    print("----------------------------------")
    train(epoch)
    #validation()
#    print(epoch)
#    if epoch >= 3:
#        t0 = time.time()
    #model.train()
    #print(type(train_graphs))
    #logits = model(train_graphs, train_features)
    #logp = F.log_softmax(logits, 1)
    #loss = F.nll_loss(logp, train_labels)

    #optimizer.zero_grad()
    #loss.backward()
    #optimizer.step()

#    if epoch >= 3:
#        dur.append(time.time() - t0)

    #acc = evaluate(net, valid_graphs, valid_features, valid_labels)
#    print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
#        epoch, loss.item(), acc, np.mean(dur)))
    #print("Epoch:", epoch, '|', "Train loss:", loss.item(), '|', "Validation accuracy:", acc)
