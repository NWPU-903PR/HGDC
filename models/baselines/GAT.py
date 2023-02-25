import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class GATNet(torch.nn.Module):
    def __init__(self,args):
        super(GATNet, self).__init__()
        self.args = args
        self.conv1 = GATConv(58, 300)
        self.conv2 = GATConv(300, 100)
        self.conv3 = GATConv(100, 1)

    def forward(self, data):
        edge_index = data.edge_index

        x = F.dropout(data.x, training=self.training)
        x = torch.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = torch.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x