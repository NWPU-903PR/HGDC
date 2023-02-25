import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import ChebConv
from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops, add_self_loops

class MTGCNNet(torch.nn.Module):
    def __init__(self,args):
        super(MTGCNNet, self).__init__()
        self.args = args
        self.conv1 = ChebConv(58, 300, K=2, normalization="sym")
        self.conv2 = ChebConv(300, 100, K=2, normalization="sym")
        self.conv3 = ChebConv(100, 1, K=2, normalization="sym")

        self.lin1 = Linear(58, 100)
        self.lin2 = Linear(58, 100)

        self.c1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.c2 = torch.nn.Parameter(torch.Tensor([0.5]))

    def forward(self,data):
        edge_index, _ = dropout_adj(data.edge_index, p=0.5,
                                    force_undirected=True,
                                    num_nodes=data.x.size()[0],
                                    training=self.training)
        E = data.edge_index
        pb, _ = remove_self_loops(data.edge_index)
        pb, _ = add_self_loops(pb)

        x0 = F.dropout(data.x, training=self.training)
        x = torch.relu(self.conv1(x0, edge_index))
        x = F.dropout(x, training=self.training)
        x1 = torch.relu(self.conv2(x, edge_index))

        x = x1 + torch.relu(self.lin1(x0))
        z = x1 + torch.relu(self.lin2(x0))

        pos_loss = -torch.log(torch.sigmoid((z[E[0]] * z[E[1]]).sum(dim=1)) + 1e-15).mean()

        neg_edge_index = negative_sampling(pb, data.num_nodes, data.num_edges)

        neg_loss = -torch.log(
            1 - torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)) + 1e-15).mean()

        r_loss = pos_loss + neg_loss

        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x, r_loss, self.c1, self.c2