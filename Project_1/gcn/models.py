import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

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
        #print(stdv)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = th.matmul(input, self.weight)
        #support = th.mm(input, self.weight)
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
    def __init__(self, maxAtomNum, featureNum, hiddenLayerNum1, hiddenLayerNum2, classNum, dropout, BatchSize):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(featureNum, hiddenLayerNum1)
        self.gc2 = GraphConvolution(hiddenLayerNum1, hiddenLayerNum2)
        self.fc = th.nn.Linear(in_features=hiddenLayerNum2*maxAtomNum, out_features=classNum)
        self.dropout = dropout

    def forward(self, x, adj):
        #print(x.shape)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        #print(x.shape)
        x = th.reshape(x, shape=[-1, 256*132])
        # Flatten
        #print(x.shape)
        # Fully Connetc
        x = self.fc(x)
        #print(x.shape)
        return F.softmax(x, dim=1)

'''
class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # output layer
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layers in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layers(self.g, h)
        return h
'''
