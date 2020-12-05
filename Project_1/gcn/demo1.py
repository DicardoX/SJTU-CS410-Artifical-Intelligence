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

gcn_msg = fn.copy_src(src="h", out="m")
gcn_reduce = fn.sum(msg="m", out="h")  # 聚合邻居节点的特征

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


net = Net()

# 使用DGL内置模块加载cora数据集
from dgl.data import citation_graph as citegrh

train_graphs, train_features, train_labels = load_data('train')
valid_graphs, valid_features, valid_labels = load_data('validation')

#nx.draw(g.to_networkx(), node_size=50, with_labels=True)
#plt.show()


# 测试模型
def evaluate(model, graphs, features, labels):
    model.eval()                        # 会通知所有图层您处于评估模式
    with th.no_grad():
        logits = model(graphs, features)
        #logits = logits[mask]
        #labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

# 定义优化器
optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
#dur = []  # 时间
for epoch in range(100):
#    print(epoch)
#    if epoch >= 3:
#        t0 = time.time()
    net.train()
    print(type(train_graphs))
    logits = net(train_graphs, train_features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp, train_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#    if epoch >= 3:
#        dur.append(time.time() - t0)

    acc = evaluate(net, valid_graphs, valid_features, valid_labels)
#    print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
#        epoch, loss.item(), acc, np.mean(dur)))
    print("Epoch:", epoch, '|', "Train loss:", loss.item(), '|', "Validation accuracy:", acc)
