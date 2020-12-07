import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import roc_auc_score


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from utils import encode_onehot, load_data
from models import GCN

# 超参数定义
hiddenLayerNum1 = 64
hiddenLayerNum2 = 128
hiddenLayerNum3 = 128
hiddenLayerNum4 = 256
#layerNum = 3
dropout_rate = 0.5
#lr = 1e-3
lr = 3e-3
weight_decay = 5e-4
EPOCH = 1000
BatchSize = 128
seed = 123   # Random seed
#maxAtomNum = 132

np.random.seed(seed)
th.manual_seed(seed)

# 处理并加载数据集
print("Data loading and preprocessing...")
train_graphs, train_features, train_labels = load_data('train')
valid_graphs, valid_features, valid_labels = load_data('validation')

train_labels = encode_onehot(train_labels)
valid_labels = encode_onehot(valid_labels)
train_labels = th.LongTensor(train_labels)
valid_labels = th.LongTensor(valid_labels)

# 模型构造
model = GCN(maxAtomNum=train_features[0].shape[0],
            featureNum=train_features[0].shape[1],      # 59
            hiddenLayerNum1=hiddenLayerNum1,
            hiddenLayerNum2=hiddenLayerNum2,
            hiddenLayerNum3=hiddenLayerNum3,
            hiddenLayerNum4=hiddenLayerNum4,
            classNum=train_labels.max().item() + 1,
            dropout=dropout_rate)

# 定义优化器
optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# 模型评估
def validation(epoch):
    avgValidLoss = 0.0
    avgValidAuc = 0.0
    samepleNum = valid_features.shape[0]
    for i in range(0, samepleNum, BatchSize):

        model.eval()

        output = model(th.FloatTensor(valid_features[i: i + BatchSize]), th.FloatTensor(valid_graphs[i: i + BatchSize]))
        valid_loss = th.nn.BCELoss()(output, valid_labels[i: i + BatchSize].float())
        output_res = output.detach().numpy()[:, 1]
        output_res = np.array(output_res)
        labels = valid_labels[i: i + BatchSize].numpy()[:, 1]
        labels = np.array(labels)
        valid_auc = roc_auc_score(labels, output_res)

        avgValidLoss += valid_loss.item()
        avgValidAuc += valid_auc


    avgValidLoss = avgValidLoss * BatchSize / samepleNum
    avgValidAuc = avgValidAuc * BatchSize / samepleNum
    print("Epoch:", epoch, "|", "Validation loss:", avgValidLoss, '|', "Validation AUC Score:", avgValidAuc)

    return avgValidAuc, avgValidLoss

# 模型训练
def train(epoch):
    avgTrainLoss = 0.0
    avgTrainAuc = 0.0
    samepleNum = train_features.shape[0]
    #for i in range(0, samepleNum, BatchSize):
    for i in range(0, samepleNum, BatchSize):
    #    if samepleNum < (i + 1) * BatchSize:
    #        continue

        model.train()
        optimizer.zero_grad()
        output = model(th.FloatTensor(train_features[i: i + BatchSize]), th.FloatTensor(train_graphs[i: i + BatchSize]))
        train_loss = th.nn.BCELoss()(output, train_labels[i: i + BatchSize].float())
        output_res = output.detach().numpy()[:, 1]
        output_res = np.array(output_res)
        labels = train_labels[i: i + BatchSize].numpy()[:, 1]
        labels = np.array(labels)
        train_auc = roc_auc_score(labels, output_res)
        train_loss.backward()
        optimizer.step()

        avgTrainLoss += train_loss.item()
        avgTrainAuc += train_auc

    avgTrainLoss = avgTrainLoss * BatchSize /  samepleNum
    avgTrainAuc = avgTrainAuc * BatchSize / samepleNum
    print("Epoch:", epoch, '|', "Train loss:", avgTrainLoss, '|', "Train AUC Score:", avgTrainAuc)

# Current best validation loss and accuracy
bestValidLoss = 1000
bestValidAuc = 0
worseCount = 0

# 迭代训练
print("Model training...")
#aucArr = []
lossArr = []
for epoch in range(EPOCH):
    print("----------------------------------")
    train(epoch)
    validAuc, validLoss = validation(epoch)
    #aucArr.append(validAuc)
    lossArr.append(validLoss)

    if validLoss <= bestValidLoss:
        worseCount = 0
        bestValidLoss = validLoss
        th.save(model, './model/model_loss.pkl')
        #th.save(model.state_dict(), './model/parameter.pkl')
    else:
        worseCount += 1

    #if validAuc >= bestValidAuc:
    #    worseCount = 0
    #    bestValidAuc = validAuc
    #    th.save(model, './model/model_auc.pkl')
    #else:
    #    worseCount += 1

    if worseCount > 10:
        print("Constantly getting worse valid AUC score. Early stop!")
        break

    #print("Currently best validation loss:", bestValidLoss)
    print("Currently best validation loss:", bestValidLoss, "(for", worseCount, "epochs not improved)")

plt.plot(lossArr)
plt.grid(True)  # 增加网格点
plt.title('Valid Loss Curve')  # 设置图表标题
plt.xlabel('Epoch')  # 横坐标标题
plt.ylabel('Loss')  # 纵坐标标题
plt.axis('tight')  # 坐标轴紧密排布
plt.show()
