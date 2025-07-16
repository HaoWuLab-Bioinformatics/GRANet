import torch
from Model import DeepGeneNet
import pandas as pd
import numpy as np
from preprocessing import scRNADataset, load_data, adj2saprse_tensor, Feature_discretization_data
import os
import random
import warnings

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING']= '1'
warnings.filterwarnings(action='ignore', category=FutureWarning)

net_type = "Specific"
cell_type = "mHSC-GM"
num_pair = 25
gene_num = str(500)

exp_file = ('../Dataset/Benchmark Dataset/' +
            net_type + ' Dataset/' + cell_type + '/TFs+' + str(gene_num) + '/BL--ExpressionData.csv')

tf_file = '../Dataset/Benchmark Dataset/' + net_type + ' Dataset/' + cell_type + '/TFs+' + str(gene_num) + '/TF.csv'

target_file = '../Dataset/Benchmark Dataset/' + net_type + ' Dataset/' + cell_type + '/TFs+' + str(
    gene_num) + '/Target.csv'

train_file = '../Dataset/train/' + net_type + '/' + cell_type + ' ' + str(gene_num) + '/Train_set.csv'

data_input = pd.read_csv(exp_file,index_col=0)
loader = load_data(data_input)
feature = loader.exp_data()
discretization_fea = Feature_discretization_data(data_input, 20)
smoothed_fea = data_input.rolling(window=3, axis=1, min_periods=1).mean().to_numpy()
tf = pd.read_csv(tf_file,index_col=0)['index'].values.astype(np.int64)
target = pd.read_csv(target_file,index_col=0)['index'].values.astype(np.int64)
feature = torch.from_numpy(feature)
smoothed_fea = torch.from_numpy(smoothed_fea)
tf = torch.from_numpy(tf)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_feature = feature.to(device)
discretization_fea = discretization_fea.to(device)
smoothed_fea = smoothed_fea.to(torch.float32).to(device)
tf = tf.to(device)
train_data = pd.read_csv(train_file, index_col=0).values


train_load = scRNADataset(train_data, feature.shape[0])
adj = train_load.Adj_Generate(tf,loop=False)
adj = adj2saprse_tensor(adj).to('cuda:0')

#%%
model = DeepGeneNet(
        input_dim=feature.size()[1],
        hidden1_dim=128,
        hidden2_dim=64,
        hidden3_dim=32,
        output_dim=16,
        num_head1=3,
        alpha=0.2,
        device=torch.device('cuda:0'),
)
model = model.to(device)

model_path = '../Code/' + 'model/' + net_type + '/' + cell_type + ' ' + str(gene_num)
model.load_state_dict(torch.load(model_path + '/best_model.pkl'))

cs = pd.DataFrame(columns=['TF','Target','Label'])
cs.Target = pd.read_csv(target_file).index
cs.TF = 508
cs.Label = -1
cs_data = torch.tensor(cs.to_numpy()).to('cuda:0')

score_test = model(data_feature, smoothed_fea, discretization_fea, adj, cs_data)
score_test = torch.nn.functional.sigmoid(score_test)
score_test = score_test.cpu().detach().numpy().round(3)
index = score_test.reshape(-1).argsort()[-num_pair:]
result = cs.iloc[index, :]
result = result.reindex(['TF', 'Target', 'Value'], axis=1)
TF_csv = pd.read_csv(tf_file,index_col=0)
TF_name_index = TF_csv['index'] == result.TF.values[0]
TF_name = TF_csv[TF_name_index].TF.values[0]
Target_csv = pd.read_csv(target_file,index_col=0)
Target_name = Target_csv[Target_csv['index'].isin(result.Target.values)]
result.loc[:, 'TF'] = TF_name
result.loc[:, 'Target'] = Target_name.iloc[:, 0]
result.loc[:, 'Value'] = score_test[index]
result = result.iloc[::-1]
result.index = pd.Index(np.arange(num_pair))
result.to_csv(f"./Regulatory_relationship/{cell_type}_{TF_name}.csv")


