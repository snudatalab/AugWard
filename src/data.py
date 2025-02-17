import json
import os
import pickle
import torch
import torch_geometric

from sklearn.model_selection import StratifiedKFold
from torch_geometric.datasets import TUDataset
import numpy as np
import networkx as nx
from torch_geometric.utils import contains_self_loops, contains_isolated_nodes, \
    is_undirected, to_networkx, degree


DATASETS = ['DD', 'ENZYMES', 'IMDB-BINARY', 'IMDB-MULTI', 'NCI1', 'NCI109', 'PROTEINS', 'PTC_MR', 'REDDIT-BINARY']

ROOT = './data'

def to_degree_features(data):
    d_list = []
    for graph in data:
        d_list.append(degree(graph.edge_index[0], num_nodes=graph.num_nodes))
    x = torch.cat(d_list).long()
    unique_degrees = torch.unique(x)
    mapper = torch.full_like(x, fill_value=1000000000)
    mapper[unique_degrees] = torch.arange(len(unique_degrees))
    x_onehot = torch.zeros(x.size(0), len(unique_degrees))
    x_onehot[torch.arange(x.size(0)), mapper[x]] = 1
    return x_onehot


def load_data(dataset, degree_x=True):
    data = TUDataset(root=os.path.join(ROOT, 'graphs'), name=dataset,
                     use_node_attr=False)
    data.data.edge_attr = None
    if data.num_node_features == 0:
        num_nodes_list = [g.num_nodes for g in data]
        data.slices['x'] = torch.tensor([0] +  num_nodes_list).cumsum(0)
        if degree_x:
            data.data.x = to_degree_features(data)
        else:
            num_all_nodes = sum(g.num_nodes for g in data)
            data.data.x = torch.ones((num_all_nodes, 1))
    return data


def load_data_fold(dataset, fold, degree_x=True, num_folds=10, seed=0):
    assert 0 <= fold < 10

    data = load_data(dataset, degree_x)
    path = os.path.join(ROOT, 'splits', dataset, f'{fold}_{seed}.json')
    if not os.path.exists(path):
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        trn_idx, test_idx = list(skf.split(np.zeros(data.len()), data.data.y))[fold]
        trn_idx = [int(e) for e in trn_idx]
        test_idx = [int(e) for e in test_idx]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(dict(training=trn_idx, test=test_idx), f, indent=4)

    with open(path) as f:
        indices = json.load(f)
    trn_graphs = [data[i] for i in indices['training']]
    test_graphs = [data[i] for i in indices['test']]
    return trn_graphs, test_graphs

def print_stats():
    for data in DATASETS:
        out = load_data(data)
        num_graphs = len(out)
        num_nodes = out.data.x.size(0)
        num_edges = out.data.edge_index.size(1) // 2
        num_features = out.num_features
        num_classes = out.num_classes
        print(f'{data}\t{num_graphs}\t{num_nodes}\t{num_edges}\t{num_features}\t'
              f'{num_classes}', end='\t')

        undirected, self_loops, onehot, connected, isolated_nodes = \
            True, False, True, True, False
        for graph in out:
            if not is_undirected(graph.edge_index, num_nodes=graph.num_nodes):
                undirected = False
            if contains_self_loops(graph.edge_index):
                self_loops = True
            if ((graph.x > 0).sum(dim=1) != 1).sum() > 0:
                onehot = False
            if not is_connected(graph):
                connected = False
            if contains_isolated_nodes(graph.edge_index, num_nodes=graph.num_nodes):
                isolated_nodes = True
        print(f'{undirected}\t{self_loops}\t{onehot}\t{connected}\t{isolated_nodes}')


def download():
    for data in DATASETS:
        load_data(data)
        for fold in range(10):
            load_data_fold(data, fold)

def is_connected(graph):
    return nx.is_connected(to_networkx(graph, to_undirected=True))


def save_pickle(file_name, data):
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)

def load_pickle(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)

def data_file_exists(file_name):
    return os.path.exists(file_name)

def batch_to_dicts(batch):
    return [data.to_dict() for data in batch.to_data_list()]

def dicts_to_batch(dicts):
    data_list = [torch_geometric.data.Data(**d) for d in dicts]
    return torch_geometric.data.Batch().from_data_list(data_list)

if __name__ == '__main__':
    download()
    print_stats()

