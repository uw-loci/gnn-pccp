import torch.nn.functional as F
import torch
from torch import nn
from torch.nn import Dropout, Linear
from torch.nn import Sequential as Layers
from torch_geometric.nn import GCNConv, SGConv, GINConv
from torch_geometric.nn import global_mean_pool, GlobalAttention
from torch_geometric.utils import dropout_adj
import numpy as np

class Preprocess(nn.Module):
    def __init__(self, in_channels=512, hidden_channels=512, out_channels=512, out_class=2, n_layers=1, head=False, backbone=None):
        super(Preprocess, self).__init__()
        torch.manual_seed(12345)

        if not backbone:
            self.lin1 =  Layers(Linear(in_channels, hidden_channels), nn.PReLU())
            if n_layers > 0:
                layers = [Layers(Linear(hidden_channels, hidden_channels), nn.PReLU()) for i in range(n_layers)]
                self.layers = Layers(*layers)
            else:
                self.layers = nn.Identity()
            self.lin2 =  nn.Sequential(Linear(hidden_channels, out_channels), nn.PReLU())
        else: 
            self.backbone=backbone

        if head:
            self.classifier = Linear(out_channels, out_class)

        self.head = head
        self.backbone = backbone

    def forward(self, x, batch=None, **kwargs):
        if not self.backbone:
            x = self.lin1(x)
            x = self.layers(x)
            x = self.lin2(x)
        else: 
            x = self.backbone(x)
            x = x.view(x.shape[0], -1)

        if self.head: 
            x = global_mean_pool(x, batch)
            x = self.classifier(x)

        return x


class AttentionPool(nn.Module):
    def __init__(self, in_chanels, hidden_channels, out_channels, p):
        super(AttentionPool, self).__init__()
        gate = Layers(Linear(in_chanels, hidden_channels), nn.Tanh(), Linear(hidden_channels, 1))
        proj = Layers(Dropout(p), Linear(hidden_channels*2, out_channels))
        self.attention_pool = GlobalAttention(gate)
        self.proj = proj

    def forward(self, x):
        rep = self.attention_pool(x)
        return rep


class GraphNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, gcn_layer='SGC', graph_pool='att', preprocess=nn.Identity(), drop_p0=0.0, drop_p1=0.0):
        super(GraphNet, self).__init__()
        
        self.preproc = preprocess
        self.gcn_layer = gcn_layer
        self.graph_pool = graph_pool

        if gcn_layer=='GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels*2, add_self_loops=True)
            self.act1 = Layers(Dropout(drop_p0), Linear(hidden_channels*2, hidden_channels), nn.PReLU())
            self.conv2 = GCNConv(hidden_channels, hidden_channels*2, add_self_loops=True)
            self.act2 = Layers(Dropout(drop_p0), Linear(hidden_channels*2, hidden_channels), nn.PReLU())
            self.conv3 = GCNConv(hidden_channels, hidden_channels*2, add_self_loops=True)
            self.act3 = Layers(Dropout(drop_p0), Linear(hidden_channels*2, hidden_channels), nn.PReLU())

        if gcn_layer=='SGC':
            self.lin1 = Layers(Linear(hidden_channels, hidden_channels), nn.PReLU())
            self.conv1 = SGConv(in_channels, hidden_channels*2, add_self_loops=True, K=3)
            self.act1 = Layers(Linear(hidden_channels*2, hidden_channels), nn.PReLU())

        if gcn_layer=='GIN':
            self.conv1 = GINConv(Layers(Linear(in_channels, hidden_channels*2), nn.PReLU()))
            self.act1 = Layers(Linear(hidden_channels*2, hidden_channels), nn.PReLU())
            self.conv2 = GINConv(Layers(Linear(hidden_channels, hidden_channels*2), nn.PReLU()))
            self.act2 = Layers(Linear(hidden_channels*2, hidden_channels), nn.PReLU())
            self.conv3 = GINConv(Layers(Linear(hidden_channels, hidden_channels*2), nn.PReLU()))
            self.act3 = Layers(Linear(hidden_channels*2, hidden_channels), nn.PReLU())

        if graph_pool == 'mean':
            self.read = global_mean_pool

        if graph_pool == 'att':
            self.read = GlobalAttention(Layers(
                Dropout(0.25),
                Linear(hidden_channels*2, hidden_channels), nn.PReLU(hidden_channels),
                Linear(hidden_channels, 1)
                ))

        self.proj = Layers(Dropout(drop_p1), Linear(hidden_channels*2, out_channels))


    def forward(self, x, edge_index, edge_weight=None, batch=None):
        t = min(max(0, np.random.normal(0.1, 0.1)), 0.5) # drop edges
        edge_index, edge_weight = dropout_adj(edge_index, p=t, training=self.training)
        t = min(max(0, np.random.normal(0.1, 0.1)), 0.5) # drop nodes
        p = F.dropout(x.new_ones((x.size(0), 1)), p=t, training=self.training)
        x = p * x
        x = self.preproc(x)

        if self.gcn_layer=='GCN' or self.gcn_layer=='GIN':
            x1 = self.act1(self.conv1(x, edge_index, edge_weight))
            x2 = self.act2(self.conv2(x1, edge_index, edge_weight))
            x3 = self.act3(self.conv3(x2, edge_index, edge_weight))
            feats = torch.cat( (x, x3), 1)

        elif self.gcn_layer=='SGC':
            x1 = self.lin1(x)
            x1 = self.act1(self.conv1(x1, edge_index, edge_weight))
            feats = torch.cat( (x, x1), 1)

        reads = self.read(feats, batch)

        out = self.proj(reads)
        return out