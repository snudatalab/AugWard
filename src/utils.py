import argparse
import copy
import csv
import networkx as nx
import numpy as np
import os
import pandas as pd
import random
import torch
import torch_geometric
import time

from scipy.sparse.csgraph import shortest_path
import torch
import networkx as nx
import numpy as np
import torch.nn.functional as F

from typing import Optional
from torch_geometric.transforms import GDC
from torch.distributions import Uniform, Beta
from torch_geometric.utils import dropout_adj, to_networkx, to_undirected, degree, to_scipy_sparse_matrix, \
    from_scipy_sparse_matrix, add_self_loops, subgraph
from torch.distributions.bernoulli import Bernoulli

from typing import *
import os
import torch
import random
import numpy as np
from ot.backend import get_backend
from ot.bregman import sinkhorn
from ot.optim import cg
from ot.lp import emd_1d, emd
from ot.gromov import update_feature_matrix, fused_gromov_wasserstein
from ot.utils import check_random_state, unif, dist, UndefinedParameter, list_to_array
from grakel import GraphKernel, Graph
from gin import GIN
from sklearn.metrics import f1_score, accuracy_score
from scipy.spatial.distance import cdist

def update_structure_matrix(p, lambdas, T, Cs):
    r"""Updates :math:`\mathbf{C}` according to the L2 Loss kernel with the `S` :math:`\mathbf{T}_s` couplings.

    It is calculated at each iteration

    Parameters
    ----------
    p : array-like, shape (N,)
        Masses in the targeted barycenter.
    lambdas : list of float
        List of the `S` spaces' weights.
    T : list of S array-like of shape (ns, N)
        The `S` :math:`\mathbf{T}_s` couplings calculated at each iteration.
    Cs : list of S array-like, shape (ns, ns)
        Metric cost matrices.

    Returns
    -------
    C : array-like, shape (`nt`, `nt`)
        Updated :math:`\mathbf{C}` matrix.
    """
    p = list_to_array(p)
    T = list_to_array(*T)
    Cs = list_to_array(*Cs)
    nx = get_backend(*Cs, *T, p)

    tmpsum = sum([
        lambdas[s] * nx.dot(
            nx.dot(T[s].T, Cs[s]),
            T[s]
        ) for s in range(len(T))
    ])
    ppt = nx.outer(p, p)
    return tmpsum / ppt

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ['true']:
        return True
    elif v.lower() in ['false']:
        return False
    else:
        raise argparse.ArgumentTypeError()

def find_best_epoch(test_loss, test_acc):
    if isinstance(test_loss, np.ndarray) and test_loss.ndim > 1:
        test_loss = test_loss.mean(axis=1)
    if isinstance(test_acc, np.ndarray) and test_acc.ndim > 1:
        test_acc = test_acc.mean(axis=1)

    best_epoch, best_loss, best_acc = -1, np.inf, 0
    for i in range(len(test_acc)):
        if (test_acc[i] > best_acc) or (test_acc[i] == best_acc and test_loss[i] < best_loss):
            best_epoch = i
            best_loss = test_loss[i]
            best_acc = test_acc[i]
    return best_epoch

def check_path(model_path):
    directory = os.path.dirname(model_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def to_device(gpu):
    if gpu is not None and torch.cuda.is_available():
        return torch.device('cuda:{}'.format(gpu))
    else:
        return torch.device('cpu')


@torch.no_grad()
def eval_acc(model, loader, device, metric='acc'):
    model.eval()
    y_true, y_pred = [], []
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        y_pred.append(output.argmax(dim=1).cpu())
        y_true.append(data.y.cpu())
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    return accuracy_score(y_true, y_pred)
 

@torch.no_grad()
def eval_loss(model, loss_func, loader, device):
    model.eval()
    count_sum, loss_sum = 0, 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        loss = loss_func(output, data.y).item()
        loss_sum += loss * len(data.y)
        count_sum += len(data.y)
    return loss_sum / count_sum

def divide_by_label(len):
    label_num = int(len / 10)
    rand = torch.randperm(len)
    label_idx = rand[:label_num]
    unlabel_idx = rand[label_num:]

    return label_idx, unlabel_idx

def into_batch_semi(label_idx, unlabel_idx, batch_size):
    label_batch_num = int(batch_size / 10)
    if label_batch_num == 0:
        label_batch_num = 1
    label_batch_idx = label_idx[torch.randperm(len(label_idx))][:label_batch_num]
    unlabel_batch_idx = unlabel_idx[torch.randperm(len(unlabel_idx))][:(batch_size-label_batch_num)]
    return label_batch_idx, unlabel_batch_idx

def to_data(index, graph):
    data_list = []
    for i in index:
        data = copy.deepcopy(graph[i])
        data_list.append(data)
        del data
    return torch_geometric.data.Batch.from_data_list(data_list)

def save_csv_trial(out_list, path):
    with open(path, 'w', newline='') as csvfile:
        fieldnames = ['epochs', 'trn_loss', 'trn_acc', 'test_loss', 'test_acc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(out_list['trn_loss'])):
            writer.writerow({
                'epochs': i,
                'trn_loss': out_list['trn_loss'][i],
                'trn_acc': out_list['trn_acc'][i],
                'test_loss': out_list['test_loss'][i],
                'test_acc': out_list['test_acc'][i]
            })

def save_csv_results(results, csv_path):
    # Create a DataFrame from the results
    df_new = pd.DataFrame([results])

    # Format the accuracy values
    df_new['trn_acc'] = df_new['trn_acc'].apply(lambda x: '{:.4f}'.format(x))
    df_new['val_acc'] = df_new['val_acc'].apply(lambda x: '{:.4f}'.format(x))
    df_new['test_acc'] = df_new['test_acc'].apply(lambda x: '{:.4f}'.format(x))
    df_new['train_time'] = df_new['train_time'].apply(lambda x: '{:.4f}'.format(x))

    # Check if the file exists
    if not os.path.exists(csv_path):
        df_new.to_csv(csv_path, index=False)
    else:
        # Read the existing CSV file
        df = pd.read_csv(csv_path)
        
        # Append the new results
        df = pd.concat([df, df_new], ignore_index=True)
        
        # Sort the DataFrame by dataset and trial
        df = df.sort_values(by=["dataset", "trial"])
        
        # Save the updated DataFrame to the CSV file
        df.to_csv(csv_path, index=False)

def load_model(args, data):
    assert args.model == "GIN"
    print(f"GIN model with {args.layers} layers")
    num_features = data.num_features
    num_classes = data.num_classes
    model = GIN(num_features, num_classes, args.units, args.layers + 1, args.dropout)
    model_path = f"./models/{args.folder}_{args.layers}/{args.dataset}_{args.trial}.pt"
    
    ## temp
    if not os.path.exists(os.path.dirname(model_path)):
        model_path = f"./models/{args.folder}/{args.dataset}_{args.trial}.pt"

    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def print_list_tensor_shape(l):
    print(f">> {len(l)} tensors")
    for item in l:
        print(f"  {item.shape}")
    
def edge_set(edge_index, batch):
    """
    Converts edge_index and batch to a list of sets, each set containing the edges of a graph.
    """
    num_graphs = batch.max().item() + 1
    edge_sets = [set() for _ in range(num_graphs)]
    for i, (u, v) in enumerate(edge_index.t().tolist()):
        graph_id = batch[u].item()
        edge_sets[graph_id].add((u, v))
    return edge_sets

def jaccard_similarity(batch1, batch2):
    """
    Computes the Jaccard similarity between each pair of graphs in batch1 and batch2.
    """
    edge_sets1 = edge_set(torch_geometric.utils.to_undirected(batch1.edge_index), batch1.batch)
    edge_sets2 = edge_set(torch_geometric.utils.to_undirected(batch2.edge_index), batch2.batch)

    similarities = []
    for set1, set2 in zip(edge_sets1, edge_sets2):
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        similarity = len(intersection) / len(union) if union else 1
        similarities.append(similarity)
    return torch.tensor(similarities)

def feature_l2(batch1, batch2):
    """
    Computes the L2 norm of feature matrices between each pair of graphs in batch1 and batch2,
    padding the matrices if they have different sizes.
    """
    def pad_features(features, max_size, feature_size):
        pad_size = max_size - features.size(0)
        if pad_size > 0:
            padding = torch.zeros((pad_size, feature_size), dtype=features.dtype, device=features.device)
            features = torch.cat([features, padding], dim=0)
        return features

    max_nodes = max(batch1.num_nodes, batch2.num_nodes)

    graphs1 = [pad_features(data.x, max_nodes, data.num_features) for data in batch1.to_data_list()]
    graphs2 = [pad_features(data.x, max_nodes, data.num_features) for data in batch2.to_data_list()]

    l2_norms = []
    for graph1, graph2 in zip(graphs1, graphs2):
        diff = graph1 - graph2
        l2_norm = torch.sqrt(torch.sum(diff ** 2))
        l2_norms.append(l2_norm)

    return torch.tensor(l2_norms)

def adjacency_l2(batch1, batch2):
    """
    Computes the L2 norm of adjacency matrices between each pair of graphs in batch1 and batch2,
    padding the matrices if they have different sizes.
    """
    def pad_adj(adj, max_size):
        pad_size = max_size - adj.size(0)
        if pad_size > 0:
            padding = torch.zeros((adj.size(0), pad_size), dtype=adj.dtype, device=adj.device)
            adj = torch.cat([adj, padding], dim=1)
            padding = torch.zeros((pad_size, max_size), dtype=adj.dtype, device=adj.device)
            adj = torch.cat([adj, padding], dim=0)
        return adj
    
    max_nodes = max(batch1.num_nodes, batch2.num_nodes)

    adj_matrices1 = [pad_adj(torch_geometric.utils.to_dense_adj(data.edge_index, max_num_nodes=batch1.num_nodes).squeeze(0), max_nodes) 
                     for data in batch1.to_data_list()]
    adj_matrices2 = [pad_adj(torch_geometric.utils.to_dense_adj(data.edge_index, max_num_nodes=batch2.num_nodes).squeeze(0), max_nodes) 
                     for data in batch2.to_data_list()]

    l2_norms = []
    for adj1, adj2 in zip(adj_matrices1, adj_matrices2):
        diff = adj1 - adj2
        l2_norm = torch.sqrt(torch.sum(diff ** 2))
        l2_norms.append(l2_norm)

    return torch.tensor(l2_norms)

def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

def to_networkx(data):
    G = nx.Graph()

    for i in range(data.num_nodes):
        G.add_node(i)

    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        source = edge_index[0, i]
        target = edge_index[1, i]
        G.add_edge(source, target)

    return G

def compute_geo(G):
    return nx.floyd_warshall_numpy(G)

def compute_fea(features1, features2, metric='euclidean'):
    return cdist(features1, features2, metric)


def fused_ACC_numpy(M, A, B, a=None, b=None, X=None, alpha=0, epoch=2000, eps=1e-5, rho=1e-1):
    # Check inputs for NaN or extreme values
    if A.shape[0] != A.shape[1] or B.shape[0] != B.shape[1]:
        raise ValueError("A and B must be square matrices.")
    if M.shape[0] != A.shape[0] or M.shape[1] != B.shape[0]:
        raise ValueError("The shape of M must be compatible with A and B.")
    if a is None:
        a = np.ones([A.shape[0], 1], dtype=np.float32) / A.shape[0]
    else:
        a = a[:, np.newaxis]
        
    if b is None:
        b = np.ones([B.shape[0], 1], dtype=np.float32) / B.shape[0]
    else:
        b = b[:, np.newaxis]
    
    if X is None:
        X = a @ b.T
    
    obj_list = []
    
    for ii in range(epoch):
        # if ii % 30 == 0:
        # print(f"Iteration: {ii}")
        X = X + 1e-10  # Ensure X is not zero
        grad = 4 * alpha * A @ X @ B - (1 - alpha) * M

        # Adjusting the clipping range and apply exponential operation carefully
        clip_range = np.log(10)
        exp_grad = np.exp(np.clip(grad / rho, -clip_range, clip_range))
        X_update = exp_grad * X

        if np.isnan(X_update).any():
            print('NaN values found in X_update after exp_grad')
            break

        X = X_update
        X = X * (a / np.clip(X @ np.ones_like(b), 1e-10, np.inf))

        # Repeat for the second update
        grad = 4 * alpha * A @ X @ B - (1 - alpha) * M
        exp_grad = np.exp(np.clip(grad / rho, -clip_range, clip_range))
        X_update = exp_grad * X

        if np.isnan(X_update).any():
            print('NaN values found in X_update after second exp_grad')
            break

        X = X_update
        X = X * (b.T / np.clip(X.T @ np.ones_like(a), 1e-10, np.inf).T)

        # Check convergence condition
        if ii > 0 and ii % 10 == 0:
            objective = np.trace(((1-alpha) * M - 2 * alpha * A @ X @ B) @ X.T)
            if len(obj_list) > 0 and np.abs((objective-obj_list[-1])/obj_list[-1]) < eps:
                break
            obj_list.append(objective)

    return X, obj_list

def fgw_distance(batch1, batch2, trns, augs, alpha, rho):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch1 = batch1.to(device)
    batch2 = batch2.to(device)
    indices = batch1.graph_idx_batch
    distances = []
    for i in range(batch1.num_graphs):
        run_time = time.time()
        graph1 = trns[indices[i]]
        graph2 = augs[indices[i]]  
        num_nodes = graph1.x.size(0)  # Number of nodes
        num_edges = graph1.edge_index.size(1) // 2  # Number of edges

        # print(f"Number of nodes: {num_nodes}")
        # print(f"Number of edges: {num_edges}")
        Ys, Cs, ps = extract_graph_matrices(graph1, graph2)
        # print(Ys[0].shape, Ys[1].shape)
        Ms = [cdist(Ys[0], Ys[1], 'euclidean')]

        dists = estimate_fgw_distance(Ms, Cs[0], Cs[1:], ps[0], ps[1:], alpha, rho)
        # print(Ms, Cs[0], Cs[1:], ps[0], ps[1:], alpha, rho)
        distances.append(dists[0])
        print(f"Graph {i} done in {time.time() - run_time:.2f} seconds")

    return torch.tensor(distances, device=device)

def extract_graph_matrices(graph1, graph2):
    Ys = [graph1.x.detach().cpu().numpy(), graph2.x.detach().cpu().numpy()]

    def get_structure_matrix(graph):
        num_nodes = graph.x.size(0)
        if graph.edge_index is not None:
            C = torch_geometric.utils.to_dense_adj(graph.edge_index, max_num_nodes=num_nodes).cpu().numpy()[0]
        else:
            # Assuming a zero matrix if there are no edges
            C = np.zeros((num_nodes, num_nodes))
        return C

    C1 = get_structure_matrix(graph1)
    C2 = get_structure_matrix(graph2)

    # Normalization with handling of zero rows
    C1_row_sums = C1.sum(axis=1, keepdims=True)
    C1 = np.divide(C1, C1_row_sums, out=np.zeros_like(C1), where=C1_row_sums!=0)

    C2_row_sums = C2.sum(axis=1, keepdims=True)
    C2 = np.divide(C2, C2_row_sums, out=np.zeros_like(C2), where=C2_row_sums!=0)

    Cs = [C1, C2]
    ps = [np.ones(C1.shape[0]) / C1.shape[0], np.ones(C2.shape[0]) / C2.shape[0]]

    return Ys, Cs, ps


def estimate_fgw_distance(Ms, C, Cs, p, ps, alpha, rho):
    dists = []
    for s in range(len(Cs)):
        if Ms[0].shape != (C.shape[0], Cs[s].shape[0]):
            raise ValueError(f"Shape mismatch: M shape {Ms[0].shape} does not match with A and B shapes {C.shape[0]}, {Cs[s].shape[0]}")
        a, b = p, ps[s]
        X_init = np.outer(a, b)
        cur_T, cur_dist = fused_ACC_numpy(Ms[0], C, Cs[s], a, b, X_init, alpha=alpha, epoch=100, eps=1e-5, rho=rho)
        # print(cur_T, cur_dist)
        c1 = np.dot(C * C, np.outer(p, np.ones_like(ps[s]))) + np.dot(np.outer(np.ones_like(p), ps[s]), Cs[s] * Cs[s])
        res = np.trace(np.dot(c1.T, cur_T))
        dists.append(cur_dist[-1] + alpha * res)

        # cur_T, cur_log = fused_gromov_wasserstein(Ms[0], C, Cs[s], p, ps[s], "square_loss", alpha,
        #                                           max_iter=300, tol_rel=1e-5, verbose=False, log=True)
        # dists.append(cur_log['fgw_dist'])

    return dists

def semi_sup_graphs(trn_graphs, labeled_rate=0.1):
    num_labeled = int(len(trn_graphs) * labeled_rate)

    random.shuffle(trn_graphs)

    labeled_graphs = trn_graphs[:num_labeled]
    unlabeled_graphs = trn_graphs[num_labeled:]

    return labeled_graphs, unlabeled_graphs

def calculate_fgwdistances(args, trn_loader, aug_loader, trn_graphs, augmented_batches, device):
    fgw_distances = []
    for (trn_batch, aug_batch) in zip(trn_loader, aug_loader):
        trn_batch = trn_batch.to(device)
        aug_batch = aug_batch.to(device)
        fgwdistance = fgw_distance(trn_batch, aug_batch, trn_graphs, augmented_batches, args.alpha, 0.1).to(device)
        fgw_distances.append(fgwdistance)
    return fgw_distances

def split_dataset(dataset, split_mode, *args, **kwargs):
    assert split_mode in ['rand', 'ogb', 'wikics', 'preload']
    if split_mode == 'rand':
        assert 'train_ratio' in kwargs and 'test_ratio' in kwargs
        train_ratio = kwargs['train_ratio']
        test_ratio = kwargs['test_ratio']
        num_samples = dataset.x.size(0)
        train_size = int(num_samples * train_ratio)
        test_size = int(num_samples * test_ratio)
        indices = torch.randperm(num_samples)
        return {
            'train': indices[:train_size],
            'val': indices[train_size: test_size + train_size],
            'test': indices[test_size + train_size:]
        }
    elif split_mode == 'ogb':
        return dataset.get_idx_split()
    elif split_mode == 'wikics':
        assert 'split_idx' in kwargs
        split_idx = kwargs['split_idx']
        return {
            'train': dataset.train_mask[:, split_idx],
            'test': dataset.test_mask,
            'val': dataset.val_mask[:, split_idx]
        }
    elif split_mode == 'preload':
        assert 'preload_split' in kwargs
        assert kwargs['preload_split'] is not None
        train_mask, test_mask, val_mask = kwargs['preload_split']
        return {
            'train': train_mask,
            'test': test_mask,
            'val': val_mask
        }


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize(s):
    return (s.max() - s) / (s.max() - s.mean())



def batchify_dict(dicts: List[dict], aggr_func=lambda x: x):
    res = dict()
    for d in dicts:
        for k, v in d.items():
            if k not in res:
                res[k] = [v]
            else:
                res[k].append(v)
    res = {k: aggr_func(v) for k, v in res.items()}
    return res


###################
def get_adj_tensor(edge_index):
    # from edge index to scipy coo sparse matrix
    coo = to_scipy_sparse_matrix(edge_index)
    # from scipy coo sparse matrix to csr fromat
    adj = coo.tocsr()
    adj = adj + adj.T
    adj = adj.tolil()
    adj[adj > 1] = 1
    
    adj.setdiag(0)
    adj = adj.astype("float32").tocsr()
    adj.eliminate_zeros()

    assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
    assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph should be unweighted"
    
    adj = torch.LongTensor(adj.todense())
    
    return adj
    
    
def get_normalize_adj_tensor(adj, device='cuda:0'):
    device = torch.device(device if adj.is_cuda else "cpu")
    
    mx = adj + torch.eye(adj.shape[0]).to(device)
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    mx = mx @ r_mat_inv
    
    return mx


def get_spectral_weights(data):
    edge_index_ = to_undirected(data.edge_index)
    edge_index, edge_weight = torch_geometric.utils.get_laplacian(edge_index_, None, 'sym')
    return edge_index, edge_weight

##################

def permute(x: torch.Tensor) -> torch.Tensor:
    """
    Randomly permute node embeddings or features.

    Args:
        x: The latent embedding or node feature.

    Returns:
        torch.Tensor: Embeddings or features resulting from permutation.
    """
    return x[torch.randperm(x.size(0))]


def get_mixup_idx(x: torch.Tensor) -> torch.Tensor:
    """
    Generate node IDs randomly for mixup; avoid mixup the same node.

    Args:
        x: The latent embedding or node feature.

    Returns:
        torch.Tensor: Random node IDs.
    """
    mixup_idx = torch.randint(x.size(0) - 1, [x.size(0)])
    mixup_self_mask = mixup_idx - torch.arange(x.size(0))
    mixup_self_mask = (mixup_self_mask == 0)
    mixup_idx += torch.ones(x.size(0), dtype=torch.int) * mixup_self_mask
    return mixup_idx


def mixup(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Randomly mixup node embeddings or features with other nodes'.

    Args:
        x: The latent embedding or node feature.
        alpha: The hyperparameter controlling the mixup coefficient.

    Returns:
        torch.Tensor: Embeddings or features resulting from mixup.
    """
    device = x.device
    mixup_idx = get_mixup_idx(x).to(device)
    lambda_ = Uniform(alpha, 1.).sample([1]).to(device)
    x = (1 - lambda_) * x + lambda_ * x[mixup_idx]
    return x


def multiinstance_mixup(x1: torch.Tensor, x2: torch.Tensor,
                        alpha: float, shuffle=False) -> (torch.Tensor, torch.Tensor):
    """
    Randomly mixup node embeddings or features with nodes from other views.

    Args:
        x1: The latent embedding or node feature from one view.
        x2: The latent embedding or node feature from the other view.
        alpha: The mixup coefficient `\lambda` follows `Beta(\alpha, \alpha)`.
        shuffle: Whether to use fixed negative samples.

    Returns:
        (torch.Tensor, torch.Tensor): Spurious positive samples and the mixup coefficient.
    """
    device = x1.device
    lambda_ = Beta(alpha, alpha).sample([1]).to(device)
    if shuffle:
        mixup_idx = get_mixup_idx(x1).to(device)
    else:
        mixup_idx = x1.size(0) - torch.arange(x1.size(0)) - 1
    x_spurious = (1 - lambda_) * x1 + lambda_ * x2[mixup_idx]

    return x_spurious, lambda_


def drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    device = x.device
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < drop_prob
    drop_mask = drop_mask.to(device)
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def dropout_feature(x: torch.FloatTensor, drop_prob: float) -> torch.FloatTensor:
    return F.dropout(x, p=1. - drop_prob)


class AugmentTopologyAttributes(object):
    def __init__(self, pe=0.5, pf=0.5):
        self.pe = pe
        self.pf = pf

    def __call__(self, x, edge_index):
        edge_index = dropout_adj(edge_index, p=self.pe)[0]
        x = drop_feature(x, self.pf)
        return x, edge_index


def get_feature_weights(x, centrality, sparse=True):
    if sparse:
        x = x.to(torch.bool).to(torch.float32)
    else:
        x = x.abs()
    w = x.t() @ centrality
    w = w.log()

    return normalize(w)


def drop_feature_by_weight(x, weights, drop_prob: float, threshold: float = 0.7):
    weights = weights / weights.mean() * drop_prob
    weights = weights.where(weights < threshold, torch.ones_like(weights) * threshold)  # clip
    drop_mask = torch.bernoulli(weights).to(torch.bool)
    x = x.clone()
    x[:, drop_mask] = 0.
    return x


def get_eigenvector_weights(data):
    def _eigenvector_centrality(data):
        graph = to_networkx(data)
        x = nx.eigenvector_centrality_numpy(graph)
        x = [x[i] for i in range(data.num_nodes)]
        return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)

    evc = _eigenvector_centrality(data)
    scaled_evc = evc.where(evc > 0, torch.zeros_like(evc))
    scaled_evc = scaled_evc + 1e-8
    s = scaled_evc.log()

    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]

    return normalize(s_col), evc


def get_degree_weights(data):
    edge_index_ = to_undirected(data.edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[data.edge_index[1]].to(torch.float32)
    scaled_deg_col = torch.log(deg_col)

    return normalize(scaled_deg_col), deg


