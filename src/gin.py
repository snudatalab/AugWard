
import torch
from torch.nn import functional as func
from torch import nn
import torch.nn.init as init
from torch_geometric import nn as gnn


class MLP(nn.Module):
    def __init__(self, num_features, num_classes, hidden_units=32, num_layers=1):
        super(MLP, self).__init__()
        if num_layers == 1:
            self.layers = nn.Linear(num_features, num_classes)
        elif num_layers > 1:
            layers = [nn.Linear(num_features, hidden_units),
                      nn.BatchNorm1d(hidden_units),
                      nn.ReLU()]
            for _ in range(num_layers - 2):
                layers.extend([nn.Linear(hidden_units, hidden_units),
                               nn.BatchNorm1d(hidden_units),
                               nn.ReLU()])
            layers.append(nn.Linear(hidden_units, num_classes))
            self.layers = nn.Sequential(*layers)
        else:
            raise ValueError()

    def forward(self, x):
        return self.layers(x)


class GIN(nn.Module):
    def __init__(self, num_features, num_classes, dropout=0, hidden_units=64, num_layers=3, 
                 mlp_layers=2, train_eps=False):
        super(GIN, self).__init__()
        convs, bns = [], []
        linears = [nn.Linear(num_features, num_classes)]
        for i in range(num_layers - 1):
            input_dim = num_features if i == 0 else hidden_units
            convs.append(gnn.GINConv(MLP(input_dim, hidden_units, hidden_units, mlp_layers),
                                     train_eps=train_eps))
            bns.append(nn.BatchNorm1d(hidden_units))
            linears.append(nn.Linear(hidden_units, num_classes))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.linears = nn.ModuleList(linears)
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        h_list = [x]
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h_list[-1], edge_index)
            h_list.append(torch.relu(bn(h)))
        out = 0
        for i in range(self.num_layers):
            h_pooled = gnn.global_add_pool(h_list[i], batch)
            h_pooled = self.linears[i](h_pooled)
            out += func.dropout(h_pooled, self.dropout, self.training)
        return h_list, out

    def embed(self, x, edge_index, batch):
        node_list = []
        for conv, bn in zip(self.convs, self.bns):
            if len(node_list) == 0:
                h = conv(x, edge_index)
            else:
                h = conv(node_list[-1], edge_index)
            node_list.append(torch.relu(bn(h)))
        graph_list = [gnn.global_add_pool(x, batch) for x in node_list]
        return node_list, graph_list
    
    def rep_pool(self, hs):
        out = 0
        for i in range(1, self.num_layers):
            h_pooled = self.linears[i](hs[i-1])
            out += func.dropout(h_pooled, self.dropout, self.training)
        return out
    
    def freeze_encoder(self):
        for conv, bn in zip(self.convs, self.bns):
            for param in conv.parameters():
                param.requires_grad = False
            for param in bn.parameters():
                param.requires_grad = False

    def freeze_classifier(self):
        for linear in self.linears:
            for param in linear.parameters():
                param.requires_grad = False

    def reinitialize_encoder(self):
        for conv, bn in zip(self.convs, self.bns):
            conv.reset_parameters()
            bn.reset_parameters()

    def reinitialize_classifier(self):
        for linear in self.linears:
            linear.reset_parameters()

    
class DistanceGIN(nn.Module):
    def __init__(self, num_features, num_classes, dropout=0, hidden_units=64, num_layers=3, 
                 mlp_layers=2, train_eps=False):
        super(DistanceGIN, self).__init__()
        convs, bns = [], []
        linears = [nn.Linear(num_features, num_classes)]
        for i in range(num_layers - 1):
            input_dim = num_features if i == 0 else hidden_units
            convs.append(gnn.GINConv(MLP(input_dim, hidden_units, hidden_units, mlp_layers),
                                     train_eps=train_eps))
            bns.append(nn.BatchNorm1d(hidden_units))
            linears.append(nn.Linear(hidden_units, num_classes))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.linears = nn.ModuleList(linears)
        self.num_layers = num_layers
        self.dropout = dropout
        self.disnet = nn.Linear(hidden_units * 2, 1)
        self.hidden_units = hidden_units

    def forward(self, x_1, edge_index_1, x_2, edge_index_2, batch_1, batch_2):
        h_list_1 = [x_1]
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h_list_1[-1], edge_index_1)
            h_list_1.append(torch.relu(bn(h)))
        out_1 = 0
        for i in range(self.num_layers):
            h_pooled_1 = gnn.global_add_pool(h_list_1[i], batch_1)
            h_pooled = self.linears[i](h_pooled_1)
            out_1 += func.dropout(h_pooled, self.dropout, self.training)
        
        h_list_2 = [x_2]
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h_list_2[-1], edge_index_2)
            h_list_2.append(torch.relu(bn(h)))
        out_2 = 0
        for i in range(self.num_layers):
            h_pooled_2 = gnn.global_add_pool(h_list_2[i], batch_2)
            h_pooled = self.linears[i](h_pooled_2)
            out_2 += func.dropout(h_pooled, self.dropout, self.training)

        distance = self.disnet(torch.cat((h_pooled_1, h_pooled_2), dim=1))

        return distance, out_1, out_2

    def embed(self, x, edge_index, batch):
        node_list = []
        for conv, bn in zip(self.convs, self.bns):
            if len(node_list) == 0:
                h = conv(x, edge_index)
            else:
                h = conv(node_list[-1], edge_index)
            node_list.append(torch.relu(bn(h)))
        graph_list = [gnn.global_add_pool(x, batch) for x in node_list]
        return node_list, graph_list