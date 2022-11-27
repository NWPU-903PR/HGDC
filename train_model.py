import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from utils import load_net_specific_dataset
from sklearn import metrics
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import torch_geometric.transforms as T
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj


class HGDC_C(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        self.linear1 = Linear(in_channels, hidden_channels)

        self.conv_k1_1 = GCNConv(hidden_channels, hidden_channels, add_self_loops = False)
        self.conv_k1_2 = GCNConv(hidden_channels, hidden_channels, add_self_loops = False)

        self.conv_k2_1 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops = False)
        self.conv_k2_2 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops = False)

        self.conv_k3_1 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)
        self.conv_k3_2 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)

        self.linear_r0 = Linear(hidden_channels, 1)
        self.linear_r1 = Linear(2*hidden_channels, 1)
        self.linear_r2 = Linear(2*hidden_channels, 1)
        self.linear_r3 = Linear(2*hidden_channels, 1)
        self.weight_r0 = torch.nn.Parameter(torch.Tensor([0.95]), requires_grad=True)
        self.weight_r1 = torch.nn.Parameter(torch.Tensor([0.90]), requires_grad=True)
        self.weight_r2 = torch.nn.Parameter(torch.Tensor([0.15]), requires_grad=True)
        self.weight_r3 = torch.nn.Parameter(torch.Tensor([0.10]), requires_grad=True)

    def forward(self, x_input, edge_index_1, edge_index_2):
        edge_index_1, _ = dropout_adj(edge_index_1, p=0.5,
                                      force_undirected=True,
                                      num_nodes=x_input.shape[0],
                                      training=self.training)
        edge_index_2, _ = dropout_adj(edge_index_2, p=0.5,
                                      force_undirected=True,
                                      num_nodes=x_input.shape[0],
                                      training=self.training)

        x_input = F.dropout(x_input, p=0.5, training=self.training)

        R0 = torch.relu(self.linear1(x_input))

        R_k1_1 = self.conv_k1_1(R0, edge_index_1)
        R_k1_2 = self.conv_k1_2(R0, edge_index_2)
        R1 = torch.cat((R_k1_1, R_k1_2), 1)

        R_k2_1 = self.conv_k2_1(R1, edge_index_1)
        R_k2_2 =self.conv_k2_2(R1, edge_index_2)
        R2 = torch.cat((R_k2_1, R_k2_2), 1)

        R_k3_1 = self.conv_k3_1(R2, edge_index_1)
        R_k3_2 = self.conv_k3_2(R2, edge_index_2)
        R3 = torch.cat((R_k3_1, R_k3_2), 1)

        R0 = F.dropout(R0, p=0.5, training=self.training)
        res0 = self.linear_r0(R0)
        R1 = F.dropout(R1, p=0.5, training=self.training)
        res1 = self.linear_r1(R1)
        R2 = F.dropout(R2, p=0.5, training=self.training)
        res2 = self.linear_r2(R2)
        R3 = F.dropout(R3, p=0.5, training=self.training)
        res3 = self.linear_r3(R3)

        out = res0 * self.weight_r0 + res1 * self.weight_r1 + res2 * self.weight_r2 + res3 * self.weight_r3
        return out


parser = argparse.ArgumentParser()
parser.add_argument('-epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('-lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('-w_decay', type=float, default=0.00001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('-ppr_alpha', type=float, default=0.9 )
parser.add_argument('-ppr_eps', type=float, default=0.0001 )
args = parser.parse_args(args=[])

net_avg_deg_dic=dict()
net_avg_deg_dic['GGNet']=111
net_avg_deg_dic['PathNet']=24
net_avg_deg_dic['PPNet']=50
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


net_name='PPNet'
features, edge_index, feature_names, node_names, labels, mask = load_net_specific_dataset(net_name)
std = StandardScaler()
features = std.fit_transform(features.detach().numpy())
features = torch.FloatTensor(features)

data = Data(x=features, y=labels, edge_index=edge_index)

data = data.to(device)
labels = labels.to(device)


gdc = T.GDC(self_loop_weight=None, normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=dict(method='ppr', alpha=args.ppr_alpha , eps = args.ppr_eps),
            sparsification_kwargs=dict(method='threshold', avg_degree=net_avg_deg_dic[net_name]), exact=True)
data = gdc(data)

edge_index1 = edge_index
edge_index_ppr = data.edge_index

edge_index1 = edge_index1.to(device)
features = features.to(device)

def train(model, features, edge_index1, edge_index_ppr, mask, labels, weight):
    model.train()
    optimizer.zero_grad()
    pred = model(features, edge_index1, edge_index_ppr)
    if weight is not None:
        loss = F.binary_cross_entropy_with_logits(pred[mask], labels[mask].view(-1,1), weight = weight.to(device)) 
    else:
        loss = F.binary_cross_entropy_with_logits(pred[mask], labels[mask].view(-1, 1))
    loss.backward()
    optimizer.step()

    pred_score = torch.sigmoid(pred[mask]).cpu().detach().numpy()
    Yn = labels[mask].cpu().numpy()
    precision, recall, _thresholds = metrics.precision_recall_curve(Yn, pred_score)
    auprc = metrics.auc(recall, precision)

    return loss, metrics.roc_auc_score(Yn, pred_score), auprc


model = HGDC_C(in_channels = features.shape[1], hidden_channels = 100).to(device)
optimizer = torch.optim.Adam([
    dict(params=model.linear1.parameters(), weight_decay = args.w_decay),
    dict(params=model.linear_r0.parameters(), weight_decay = args.w_decay),
    dict(params=model.linear_r1.parameters(), weight_decay = args.w_decay),
    dict(params=model.linear_r2.parameters(), weight_decay = args.w_decay),
    dict(params=model.linear_r3.parameters(), weight_decay = args.w_decay),
    dict(params=model.weight_r0, lr=args.lr * 0.1),
    dict(params=model.weight_r1, lr=args.lr * 0.1),
    dict(params=model.weight_r2, lr=args.lr * 0.1),
    dict(params=model.weight_r3, lr=args.lr * 0.1)
], lr=args.lr)


for epoch in range(1, args.epochs + 1):
    loss, auc_tr, aupr_tr = train(model, features, edge_index1, edge_index_ppr, mask, labels, weight = None)

pred = model(features, edge_index1, edge_index_ppr)
pred_all = pred.cpu().detach().numpy()
pre_res = pd.DataFrame(pred_all,columns=['socre'],index=node_names)
pre_res.to_csv(path_or_buf='./results/predicted_socre_on_%s.txt' % net_name, sep='\t', index=True, header=True)
