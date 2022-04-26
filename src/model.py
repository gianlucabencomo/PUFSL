import torch
import numpy as np
import torch.nn.functional as F

from torch_geometric.nn import GATConv, global_mean_pool 

# embedding function

class ProtoNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProtoNet, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6, edge_dim=12)
        self.conv2 = GATConv(8 * 8, out_channels, dropout=0.6, edge_dim=12) 

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        x = x.float()
        edge_attr = edge_attr.float()
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        return x.view(x.size(0), -1)    
