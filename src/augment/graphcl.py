import torch.nn as nn
import torch
import numpy as np
import random

from torch_geometric.data import Data

class Identity(nn.Module):
    def __init__(self, graphs):
        super(Identity, self).__init__()
        self.graphs = graphs

    def forward(self, index):
        graph = self.graphs[index]
        return Data(x=graph.x, edge_index=graph.edge_index, y=graph.y)

class NodeDrop(nn.Module):
    def __init__(self, graphs, aug_size):
        super(NodeDrop, self).__init__()
        self.graphs = graphs
        self.aug_size = aug_size

    def drop_nodes(self, data, aug_size):
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        drop_num = int(node_num * aug_size)
        idx_perm = np.random.permutation(node_num)

        idx_drop = idx_perm[:drop_num]
        idx_nondrop = idx_perm[drop_num:]
        idx_nondrop.sort()
        idx_dict = {idx_nondrop[n]:n for n in list(range(idx_nondrop.shape[0]))}

        edge_index = data.edge_index.numpy()
        adj = torch.zeros((node_num, node_num))
        adj[edge_index[0], edge_index[1]] = 1
        adj = adj[idx_nondrop, :][:, idx_nondrop]
        edge_index = adj.nonzero().t()

        try:
            data.edge_index = edge_index
            data.x = data.x[idx_nondrop]
            data.num_nodes = data.x.size(0)
        except:
            data = data
        return data
    
    def forward(self, index):
        graph = self.graphs[index]
        graph = self.drop_nodes(graph, self.aug_size)
        return Data(x=graph.x, edge_index=graph.edge_index, y=graph.y)

class Subgraph(nn.Module):
    def __init__(self, graphs, aug_size):
        super(Subgraph, self).__init__()
        self.aug_size = aug_size
    
    def subgraph(self, data, aug_size):
        if(aug_size == 0):
            return data
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        sub_num = max(int(node_num * aug_size), 2)

        edge_index = data.edge_index.numpy()

        idx_sub = [np.random.randint(node_num, size=1)[0]]
        idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])
        count = 0
        while len(idx_sub) <= sub_num:
            count = count + 1
            if count > node_num:
                break
            if len(idx_neigh) == 0:
                break
            sample_node = np.random.choice(list(idx_neigh))
            if sample_node in idx_sub:
                continue
            idx_sub.append(sample_node)
            idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

        idx_drop = [n for n in range(node_num) if not n in idx_sub]
        idx_nondrop = idx_sub
        data.x = data.x[idx_nondrop]
        idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}
        edge_index = data.edge_index.numpy()
        adj = torch.zeros((node_num, node_num))
        adj[edge_index[0], edge_index[1]] = 1
        adj[list(range(node_num)), list(range(node_num))] = 1
        adj = adj[idx_nondrop, :][:, idx_nondrop]
        edge_index = adj.nonzero().t()

        # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
        data.edge_index = edge_index
        # data.num_nodes = data.x.size(0)
        # print(data.num_nodes)
        return data
    
    def forward(self, index):
        graph = self.graphs[index]
        graph = self.subgraph(graph, self.aug_size)
        return Data(x=graph.x, edge_index=graph.edge_index, y=graph.y)
    
class EdgePert(nn.Module):
    def __init__(self, graphs, aug_size):
        super(EdgePert, self).__init__()
        self.graphs = graphs
        self.aug_size = aug_size

    def permute_edges(self, data, aug_size):
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        permute_num = int(edge_num * aug_size)

        edge_index = data.edge_index.numpy()

        idx_add = np.random.choice(node_num, (2, permute_num))

        # idx_add = [[idx_add[0, n], idx_add[1, n]] for n in range(permute_num) if not (idx_add[0, n], idx_add[1, n]) in edge_index]
        # edge_index = [edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)] + idx_add

        edge_index = np.concatenate((edge_index[:, np.random.choice(edge_num, (edge_num - permute_num), replace=False)], idx_add), axis=1)
        data.edge_index = torch.tensor(edge_index)
        data.num_nodes = data.x.size(0)
        return data

    def forward(self, index):
        graph = self.graphs[index]
        graph = self.permute_edges(graph, self.aug_size)
        return Data(x=graph.x, edge_index=graph.edge_index, y=graph.y)


class AttrMask(nn.Module):
    def __init__(self, graphs, aug_size):
        super(AttrMask, self).__init__()
        self.graphs = graphs
        self.aug_size = aug_size

    def mask_nodes(self, data, aug_size):
        node_num, feat_dim = data.x.size()
        mask_num = int(node_num * aug_size)

        token = data.x.mean(dim=0)
        idx_mask = np.random.choice(node_num, mask_num, replace=False)
        data.x[idx_mask] = torch.tensor(token, dtype=torch.float32)
        data.num_nodes = data.x.size(0)
        return data

    def forward(self, index):
        graph = self.graphs[index]
        graph = self.mask_nodes(graph, self.aug_size)
        return Data(x=graph.x, edge_index=graph.edge_index, y=graph.y)
