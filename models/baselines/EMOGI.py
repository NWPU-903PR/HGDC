import torch
from torch_geometric.nn import ChebConv
import torch.nn.functional as F

class EMOGINet(torch.nn.Module):
    def __init__(self,args):
        super(EMOGINet, self).__init__()
        self.args = args
        self.conv1 = ChebConv(58, 300, K=2)
        self.conv2 = ChebConv(300, 100, K=2)
        self.conv3 = ChebConv(100, 1, K=2)

    def forward(self, data):
        edge_index = data.edge_index
        x = F.dropout(data.x, training=self.training)
        x = torch.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = torch.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x