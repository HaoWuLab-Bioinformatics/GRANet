import math

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class scRNADataset(Dataset):
    def __init__(self,train_set,num_gene,flag=False):
        super(scRNADataset, self).__init__()
        self.train_set = train_set
        self.num_gene = num_gene
        self.flag = flag


    def __getitem__(self, idx):
        train_data = self.train_set[:,:2]
        train_label = self.train_set[:,-1]

        if self.flag:
            train_len = len(train_label)
            train_tan = np.zeros([train_len,2])
            train_tan[:,0] = 1 - train_label
            train_tan[:,1] = train_label
            train_label = train_tan

        data = train_data[idx].astype(np.int64)
        label = train_label[idx].astype(np.float32)

        return data, label

    def __len__(self):
        return len(self.train_set)


    def Adj_Generate(self,TF_set,direction=False, loop=False):

        adj = sp.dok_matrix((self.num_gene, self.num_gene), dtype=np.float32)

        for pos in self.train_set:
            tf = pos[0]
            target = pos[1]

            if direction == False:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    adj[target, tf] = 1.0
            else:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    if target in TF_set:
                        adj[target, tf] = 1.0


        if loop:
            adj = adj + sp.identity(self.num_gene)

        adj = adj.todok()


        return adj



class load_data():
    def __init__(self, data, normalize=True):
        self.data = data
        self.normalize = normalize

    def data_normalize(self,data):
        standard = StandardScaler()
        epr = standard.fit_transform(data.T)

        return epr.T


    def exp_data(self):
        data_feature = self.data.values

        if self.normalize:
            data_feature = self.data_normalize(data_feature)

        data_feature = data_feature.astype(np.float32)

        return data_feature




def adj2saprse_tensor(adj):
    coo = adj.tocoo()
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.from_numpy(coo.data).float()

    adj_sp_tensor = torch.sparse_coo_tensor(i, v, coo.shape)
    return adj_sp_tensor



def normalize(expression):
    std = StandardScaler()
    epr = std.fit_transform(expression)

    return epr



def Network_Statistic(data_type,net_scale,net_type):


    if net_type == 'Non-Specific':

        dic = {'hESC500': 0.016, 'hESC1000': 0.014, 'hHEP500': 0.015, 'hHEP1000': 0.013, 'mDC500': 0.019,
               'mDC1000': 0.016, 'mESC500': 0.015, 'mESC1000': 0.013, 'mHSC-E500': 0.022, 'mHSC-E1000': 0.020,
               'mHSC-GM500': 0.030, 'mHSC-GM1000': 0.029, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.043}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Specific':
        dic = {'hESC500': 0.164, 'hESC1000': 0.165,'hHEP500': 0.379, 'hHEP1000': 0.377,'mDC500': 0.085,
               'mDC1000': 0.082,'mESC500': 0.345, 'mESC1000': 0.347,'mHSC-E500': 0.578, 'mHSC-E1000': 0.566,
               'mHSC-GM500': 0.543, 'mHSC-GM1000': 0.565,'mHSC-L500': 0.525, 'mHSC-L1000': 0.507}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'STRING':
        dic = {'hESC500': 0.024, 'hESC1000': 0.021, 'hHEP500': 0.028, 'hHEP1000': 0.024, 'mDC500': 0.038,
               'mDC1000': 0.032, 'mESC500': 0.024, 'mESC1000': 0.021, 'mHSC-E500': 0.029, 'mHSC-E1000': 0.027,
               'mHSC-GM500': 0.040, 'mHSC-GM1000': 0.037, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.045}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Lofgof':
        dic = {'mESC500': 0.158, 'mESC1000': 0.154}

        query = 'mESC' + str(net_scale)
        scale = dic[query]
        return scale

    else:
        raise ValueError




def random_walk_imp(matrix, rp):
    row, col = matrix.shape
    row_sum = np.sum(matrix, axis=1)
    for i in range(row_sum.shape[0]):
        if row_sum[i] == 0:
            row_sum[i] = 0.001
    nor_matrix = np.divide(matrix.T, row_sum).T
    Q = np.eye(row)
    I = np.eye(row)
    for i in range(30):
        # The random walk process can be represented by the following matrix operations.
        Q_new = rp * np.dot(Q, nor_matrix) + (1 - rp) * I
        delta = np.linalg.norm(Q - Q_new)
        Q = Q_new.copy()
        # When delta is less than threshold, restart random walk process converges and restart random walk process terminates.
        if delta < 1e-6:
            break
    return Q



def Feature_discretization_data(data, class_num):
    data = data.T

    gene_name = list(data)  # 使用 list() 函数将 DataFrame 转换为列表时，会返回 DataFrame 的列索引，而不是 DataFrame 中的数据。

    # 暂时没发现以下两个字典的作用
    new_data_dict = {}

    zero_index = np.where(data.values == 0)  # 会返回一个元组，其中每个元素都是一个一维数组。第一个数组包含所有值为0的元素的行索引，第二个数组包含对应的列索引。
    mask = np.ones_like(data.values)
    mask[zero_index] = 0
    for gene in gene_name:
        temp = data[gene]

        non_zero_element = np.log(temp[temp != 0.].values)

        if len(non_zero_element) == 0:
            new_data_dict[gene] = temp.apply(lambda x: 0)
            continue
        # 这里主要是将基因表达矩阵中每一列（即每一基因）的非0值提取出来进行log对数化。
        # 然后再将其计算出均值、最小最大值、标准差、等级的下界与上界、等级的宽度
        mean = np.mean(non_zero_element)
        tmin = np.min(non_zero_element)
        std = np.std(non_zero_element)
        tmax = np.max(non_zero_element)
        lower_bound = max(mean - 2 * std, tmin)
        upper_bound = min(mean + 2 * std, tmax)
        bucket_width = (upper_bound - lower_bound) / class_num
        mask_zero = np.ones_like(temp)
        mask_zero[temp == 0] = 0  # 列遮掩向量，非0即1

        np.seterr(divide='ignore', invalid='ignore')
        try:
            temp = temp.apply(lambda x: 0 if x == 0.0 else math.floor(
                (np.log(x) - lower_bound) / bucket_width))  # 将基因表达矩阵中每一列（即每一基因）的每一个表达值进行等级离散化
        except:
            temp = temp.apply(lambda x: 0 if x == 0.0 else 0)  # 目前似乎用不上

        temp[temp >= class_num] = class_num - 1  # 将大于上界的表达值限制到calss_num-1,但目前花不清楚为什么要减一（和下面的temp+1相对应）
        temp[(temp < 0)] = 0  # 将小于下界的表达值限制为0，注意此时稍微高于下界的正常表达值也为0

        temp = temp + 1  # 将所有离散值的等级+1，将原先非0但是低于等级一的表达值进行处理
        temp = temp * mask_zero  # 将离散化前的0值进行处理
        new_data_dict[gene] = temp  # 创建一个新的数据字典接受处理后的离散化表达值

    new_data = pd.DataFrame(new_data_dict)
    new_data = torch.tensor(new_data.T.values, dtype=torch.float32)

    return new_data


























