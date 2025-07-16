import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False


class DeepGeneNet(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim, num_head1, alpha, device):
        super().__init__()
        self.input_dim = input_dim
        self.hidden1_dim = num_head1 * hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.num_head1 = num_head1
        self.device = device
        self.alpha = alpha
        self.dropout_rate = 0.2

        self.Multi_Head_Layer = nn.ModuleList([
            GAT_Layer(self.input_dim, self.hidden1_dim, self.alpha) for _ in range(self.num_head1)
        ])
        self.Mid_Layer = Middle_layer()
        self.Res_Layer2 = CoreModule(
            in_channels=3,
            out_channels=1,
            kernel_size=(1, (256 + 1 - self.hidden2_dim)),
            hidden_dim=self.hidden2_dim
        )

        self.tf_linear1 = nn.Linear(hidden2_dim, hidden3_dim)
        self.target_linear1 = nn.Linear(hidden2_dim, hidden3_dim)
        self.tf_linear2 = nn.Linear(hidden3_dim, output_dim)
        self.target_linear2 = nn.Linear(hidden3_dim, output_dim)
        self.embed_linear = nn.Linear(output_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        for attention in self.Multi_Head_Layer:
            attention.reset_parameters()

        nn.init.xavier_uniform_(self.tf_linear1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.target_linear1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.tf_linear2.weight, gain=1.414)
        nn.init.xavier_uniform_(self.target_linear2.weight, gain=1.414)
        nn.init.xavier_uniform_(self.embed_linear.weight, gain=1.414)

    def encode(self, x1, x2, x3, adj):
        data = torch.stack((x1, x2, x3), dim=0)
        output = [att(data[i], adj) for i, att in enumerate(self.Multi_Head_Layer)]
        x = torch.mean(torch.stack(output, dim=0), dim=0)
        x = F.elu(x)
        x = x.reshape((1, 1, x.shape[0], x.shape[1]))
        x = self.Mid_Layer(x)
        x = self.Res_Layer2(x, adj)
        return x

    def decode(self, tf_embed, target_embed):
        embed_vct = torch.mul(tf_embed, target_embed)
        # output = self.embed_linear(embed_vct)
        output = torch.sum(embed_vct, dim=1).view(-1, 1)

        return output

    def forward(self, x1, x2, x3, adj, train_sample):
        embed = self.encode(x1, x2, x3, adj)
        tf_embed = self.tf_linear1(embed)
        tf_embed = F.elu(tf_embed)
        tf_embed = F.dropout(tf_embed, p=0.01)
        tf_embed = self.tf_linear2(tf_embed)
        tf_embed = F.elu(tf_embed)
        target_embed = self.target_linear1(embed)
        target_embed = F.elu(target_embed)
        target_embed = F.dropout(target_embed, p=0.01)
        target_embed = self.target_linear2(target_embed)
        target_embed = F.elu(target_embed)

        self.tf_ouput = tf_embed
        self.target_output = target_embed
        train_tf = tf_embed[train_sample[:, 0]]
        train_target = target_embed[train_sample[:, 1]]
        pred = self.decode(train_tf, train_target)

        return pred

    def get_embedding(self):
        return self.tf_ouput, self.target_output


class GAT_Layer(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.2, bias=True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.weight = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.weight_interact = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.a = nn.Parameter(torch.zeros(size=(2 * self.output_dim, 1)))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.output_dim))
        else:
            self.register_parameter('bias', None)
        self.lamd1 = nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.lamd2 = nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_interact.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.constant_(self.lamd1, val=0.01)
        nn.init.constant_(self.lamd2, val=0.99)

    def _prepare_attentional_mechanism_input(self, x):

        Wh1 = torch.matmul(x, self.a[:self.output_dim, :])
        Wh2 = torch.matmul(x, self.a[self.output_dim:, :])
        e = F.leaky_relu(Wh1 + Wh2.mT, negative_slope=self.alpha)
        return e

    def forward(self, x, adj):

        h = torch.matmul(x, self.weight)
        e = self._prepare_attentional_mechanism_input(h)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.to_dense() > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, training=self.training)
        h_pass = torch.matmul(attention, h)
        output_data = self.lamd2 * h_pass + self.lamd1 * h
        output_data = F.elu(output_data)
        output_data = F.normalize(output_data, p=2, dim=1)
        if self.bias is not None:
            output_data = output_data + self.bias

        return output_data


class Middle_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=(1, 384 + 1 - 256)
        )
        self._initialize_parameters()

    def _initialize_parameters(self):
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        out = self.conv(x)
        nn.BatchNorm2d(256)
        out = F.elu(out)
        return out


class GeneExpressionRes(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.2, bias=True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.weight = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.weight_interact = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.a = nn.Parameter(torch.zeros(size=(2 * self.output_dim, 1)))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_interact.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def _prepare_attentional_mechanism_input(self, x):

        Wh1 = torch.matmul(x, self.a[:self.output_dim, :])
        Wh2 = torch.matmul(x, self.a[self.output_dim:, :])
        e = F.leaky_relu(Wh1 + Wh2.mT, negative_slope=self.alpha)
        return e

    def forward(self, x, adj):

        h = torch.matmul(x, self.weight)
        e = self._prepare_attentional_mechanism_input(h)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.to_dense() > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, training=self.training)
        h_pass = torch.matmul(attention, h)
        output_data = F.elu(h_pass)
        output_data = F.normalize(output_data, p=2, dim=1)
        if self.bias is not None:
            output_data = output_data + self.bias

        return output_data


class CoreModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, hidden_dim, stride=1, padding=0, alpha=0.2, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.bias = bias
        self.num_heads = 4
        self.a = nn.Parameter(torch.zeros(size=(self.num_heads, 2 * self.hidden_dim, 1)))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.lamd1 = nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.lamd2 = nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.hidden_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.constant_(self.lamd1, val=0.01)
        nn.init.constant_(self.lamd2, val=0.99)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def _prepare_attentional_mechanism_input(self, h, head_idx):
        att = self.a[head_idx]
        Wh1 = torch.matmul(h, att[:self.hidden_dim, :])
        Wh2 = torch.matmul(h, att[self.hidden_dim:, :])
        e = F.elu(Wh1 + Wh2.T)
        return e

    def forward(self, x, adj):

        h = self.conv(x)
        h = h.reshape(h.shape[2], h.shape[3])
        attention_heads = []
        for head_idx in range(self.num_heads):
            e = self._prepare_attentional_mechanism_input(h, head_idx)
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj.to_dense() > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, training=self.training)
            attention_heads.append(torch.matmul(attention, h))

        h_pass = torch.stack(attention_heads, dim=0).mean(dim=0)
        output_data = 0.5 * h_pass + 0.5 * h
        output_data = F.elu(output_data)
        output_data = F.normalize(output_data, p=2, dim=1)

        if self.bias is not None:
            output_data = output_data + self.bias

        return output_data
