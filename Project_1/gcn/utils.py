import numpy as np
import torch
from rdkit import Chem


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def adj_k(adj, k):
    ret = adj
    for i in range(0, k - 1):
        ret = np.dot(ret, adj)

    return convertAdj(ret)


def convertAdj(adj):
    dim = len(adj)
    a = adj.flatten()
    b = np.zeros(dim * dim)
    c = (np.ones(dim * dim) - np.equal(a, b)).astype('float64')
    d = c.reshape((dim, dim))

    return d


def convertToGraph(smiles_list, k):
    global columnCount
    adj = []
    adj_norm = []
    features = []
    maxNumAtoms = 132
    for i in range(0, len(smiles_list), 1):
        iMol = Chem.MolFromSmiles(smiles_list[i])
        Chem.SanitizeMol(iMol)
        iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)
        if( iAdjTmp.shape[0] <= maxNumAtoms):
            iFeature = np.zeros((maxNumAtoms, 59))
            iFeatureTmp = []
            for atom in iMol.GetAtoms():
                iFeatureTmp.append( atom_feature(atom) ) ### atom features only
            iFeature[0:len(iFeatureTmp), 0:59] = iFeatureTmp ### 0 padding for feature-set
            features.append(iFeature)
            # Adj-preprocessing
            iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
            iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
            adj.append(adj_k(np.asarray(iAdj), k))
    features = np.asarray(features, dtype=np.float32)

    return adj, features


def atom_feature(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                                           'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                                           'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                                           'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])  # (40, 6, 5, 6, 1)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    # print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def load_data(filename):
    ## 从names_smiles.txt文件中读入数据
    filefolder = './data/' + filename
    fopen = open(filefolder + "/names_smiles.txt", "r")
    List_row = fopen.readlines()
    smiles_list = []
    for i in range(1, len(List_row), 1):
        smile = List_row[i].strip().split(",")[1]
        smiles_list.append(smile)

    adjGraphs, features = convertToGraph(smiles_list, 1)

    if(filename != 'test'):
        fopen = open(filefolder + "/names_labels.txt", "r")
        List_row = fopen.readlines()
        labels_list = []
        for i in range(1, len(List_row), 1):
            label = List_row[i].strip().split(",")[1]
            labels_list.append(label)

        return adjGraphs, features, labels_list
    else:
        return adjGraphs, features