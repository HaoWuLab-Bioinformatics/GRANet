import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model import GRANet, vae_loss, VAE
from Tools import Evaluation, SavaBestModel
from preprocessing import scRNADataset, load_data, adj2saprse_tensor, Feature_discretization_data

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING']= '1'
warnings.filterwarnings(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=4e-4)
parser.add_argument('--epochs', type=int, default=95)
parser.add_argument('--num_head', type=list, default=3)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--hidden_dim', type=int, default=[128,64,32])
parser.add_argument('--output_dim', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--loop', type=bool, default=False, help='whether to use self-loop in adjacent matrix')
parser.add_argument('--seed', type=int, default=40)

args = parser.parse_args()
seed = args.seed
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False


def write_to_txt(net_type, cell_type, gene_num, auroc, auprc, filename='results.txt'):
    with open(filename, mode='a') as file:
        # 写入表头（如果文件为空）
        if file.tell() == 0:
            file.write("Experimental results record\n")
            file.write("=======================================")
        file.write(f"\n{net_type}\t{cell_type}\t{gene_num}\t{auroc:.4f}\t{auprc:.4f}")


def embed2file(tf_embed,tg_embed,gene_file,tf_path,target_path):
    tf_embed = tf_embed.cpu().detach().numpy()
    tg_embed = tg_embed.cpu().detach().numpy()

    gene_set = pd.read_csv(gene_file, index_col=0)

    tf_embed = pd.DataFrame(tf_embed,index=gene_set['Gene'].values)
    tg_embed = pd.DataFrame(tg_embed, index=gene_set['Gene'].values)

    tf_embed.to_csv(tf_path)
    tg_embed.to_csv(target_path)

def Running_Model(net_type, cell_type, gene_num, record_model):
    exp_file = '../Dataset/Benchmark Dataset/' + net_type + ' Dataset/' + cell_type + '/TFs+' + str(
        gene_num) + '/BL--ExpressionData.csv'
    tf_file = '../Dataset/Benchmark Dataset/' + net_type + ' Dataset/' + cell_type + '/TFs+' + str(gene_num) + '/TF.csv'
    target_file = '../Dataset/Benchmark Dataset/' + net_type + ' Dataset/' + cell_type + '/TFs+' + str(
        gene_num) + '/Target.csv'

    train_file = '../Dataset/train/' + net_type + '/' + cell_type + ' ' + str(gene_num) + '/Train_set.csv'
    val_file = '../Dataset/val/' + net_type + '/' + cell_type + ' ' + str(gene_num) + '/Validation_set.csv'
    test_file = '../Dataset/test/' + net_type + '/' + cell_type + ' ' + str(gene_num) + '/test_set.csv'

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
    # tf = tf.to(device)

    train_data = pd.read_csv(train_file, index_col=0).values
    validation_data = pd.read_csv(val_file, index_col=0).values
    test_date = pd.read_csv(test_file, index_col=0).values

    train_load = scRNADataset(train_data, feature.shape[0])
    adj = train_load.Adj_Generate(tf,loop=args.loop)

    adj = adj2saprse_tensor(adj)

    # train_data = torch.from_numpy(train_data)
    val_data = torch.from_numpy(validation_data)
    test_data = torch.from_numpy(test_date)
    vae = VAE(input_dim=feature.size()[1], hidden_dim=256, latent_dim=feature.size()[1])
    model = GRANet(
        input_dim=feature.size()[1],
        hidden1_dim=128,
        hidden2_dim=64,
        hidden3_dim=32,
        output_dim=16,
        num_head1=3,
        alpha=args.alpha,
        device=device,
    )
    #print(model)
    adj = adj.to(device)
    model = model.to(device)
    # train_data = train_data.to(device)
    validation_data = val_data.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.99)
    # early_stopping = EarlyStopping(save_dir='./',patience=15, verbose=True)
    loss_BCE = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))
    model_path = 'model/' + net_type + '/' + cell_type + ' ' + str(gene_num)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    metrics = []

    print(f"The cells selected were: {cell_type}, The regulatory network selected is: {net_type}, The number of genes selected is: {gene_num}")
    print("============================Start training============================")

    def train_vae(model, data, epochs=100, lr=0.001):
        """
        Training VAE
        param model
        param data
        param epochs
        param lr
        """
        print("Start reconstructing the gene expression matrix")
        optimizer = Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()

            # 前向传播
            reconstructed, mu, logvar = model(data)

            # 计算损失
            loss = vae_loss(reconstructed, data, mu, logvar)

            # 反向传播
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
        print("Complete gene expression matrix reconstruction")
    def train_GRNs():
        best_model = SavaBestModel(model_path)
        for epoch in range(args.epochs):
            running_loss = 0.0
            data_loader = DataLoader(train_load, batch_size=args.batch_size, shuffle=True, drop_last=False)
            loop = tqdm((data_loader), total=len(data_loader))
            # for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
            for train_x, train_y in loop:
                model.train()
                optimizer.zero_grad()

                train_y = train_y.to(device).view(-1, 1)

                pred = model(data_feature, smoothed_fea, discretization_fea, adj, train_x)
                loss = loss_BCE(pred, train_y)
                loss.backward()
                optimizer.step()

                running_loss += loss
                loop.set_description(f'Epoch [{epoch + 1}/{args.epochs}], loss:[{running_loss:.3f}]')
            scheduler.step()

            with torch.no_grad():

                model.eval()

                score_val = model(data_feature, smoothed_fea, discretization_fea, adj, validation_data)
                score_val = torch.nn.functional.sigmoid(score_val)
                AUROC, AUPRC = Evaluation(y_pred=score_val, y_true=validation_data[:, -1])
                #early_stopping(AUROC, model)
                metrics.append([AUROC, AUPRC])
                best_model(AUROC, AUPRC, model)
                # if early_stopping.early_stop:
                #     print("Early stopping")
                #     break;

    train_vae(vae, torch.tensor(data_input.to_numpy(), dtype=torch.float32), epochs=350, lr=0.0005)
    reconstructed_data, _, _ = vae(torch.tensor(data_input.to_numpy(), dtype=torch.float32))
    data_feature = StandardScaler().fit_transform(reconstructed_data.detach().numpy())

    discretization_fea = Feature_discretization_data(pd.DataFrame(reconstructed_data.detach().numpy()), 20)

    smoothed_fea = pd.DataFrame(reconstructed_data.detach().numpy()).rolling(window=3, axis=1, min_periods=1).mean()
    feature = torch.from_numpy(data_feature)
    smoothed_fea = torch.from_numpy(smoothed_fea.to_numpy())

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_feature = feature.to(device)
    discretization_fea = discretization_fea.to(device)
    smoothed_fea = smoothed_fea.to(torch.float32).to(device)
    train_GRNs()


    print("============================Validation============================\n")

    print(f"the best AUROC is: {np.array(metrics)[:, 0].max():.3f}, the best AUPRC is: {np.array(metrics)[:, 1].max():.3f}")

    print("\n============================Test============================\n")

    model.load_state_dict(torch.load(model_path + '/best_model.pkl'))
    score_test = model(data_feature, smoothed_fea, discretization_fea, adj, test_data)
    score_test = torch.nn.functional.sigmoid(score_test)
    AUROC, AUPRC= Evaluation(y_pred=score_test, y_true=test_data[:, -1])

    print(f"the AUROC is: {AUROC:.3f}, the AUPRC is: {AUPRC:.3f}")
    write_to_txt(net_type, cell_type, gene_num, AUROC, AUPRC)
    if record_model == True:
        print(model)


if __name__ == '__main__':

    net_types = ["String"]
    cell_types = ["hESC", "hHEP", "mDC", "mESC", "mHSC-E", "mHSC-GM", "mHSC-L"]
    #cell_types = ["hHEP"]
    gene_num =1000

    with open("results.txt", mode='a') as file:
        file.write("\n=======================================")

    for net_type in net_types:
        i = 0
        for cell_type in cell_types:
            i += 1
            record_model = True if i == 7 else False
            Running_Model(net_type, cell_type, gene_num, record_model)