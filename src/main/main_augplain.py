import argparse
import json
import math
import sys
import os
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch_geometric.data import DataLoader, Batch

from augment import Augment
from gin import GIN
from data import load_data_fold, load_data
from utils import fix_seed, check_path
from loss import SoftCELoss


def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--fold', type=int)

    # Experimental setting
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--verbose', type=int)
    parser.add_argument('--path', type=str)

    # Augmentation
    parser.add_argument('--aug', type=str)
    parser.add_argument('--aug-size', type=float)

    # Training setup
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--lr', type=float)

    # Classifier
    parser.add_argument('--model', type=str)
    parser.add_argument('--dropout', type=float)

    return parser.parse_args()


def to_device(gpu):
    if gpu is not None and torch.cuda.is_available():
        return torch.device('cuda:{}'.format(gpu))
    else:
        return torch.device('cpu')


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for data in loader:
        data = data.to(device)
        _, output = model(data.x, data.edge_index, data.batch)
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
        _, output = model(data.x, data.edge_index, data.batch)
        loss = loss_func(output, data.y).item()
        loss_sum += loss * len(data.y)
        count_sum += len(data.y)
    return loss_sum / count_sum


def main():
    args = parse_args()

    device = to_device(args.gpu)

    fix_seed(args.seed)

    data = load_data(args.data)
    num_features = data.num_features
    num_classes = data.num_classes

    augmented_batches_file = f"./data/pickle/{args.data}_{args.aug}_dataset.pkl"
    
    check_path(augmented_batches_file)

    trn_graphs, test_graphs = load_data_fold(args.data, args.seed)

    for i, data in enumerate(trn_graphs):
        data.graph_idx = i
        data.num_nodes = data.x.shape[0]
    trn_loader = DataLoader(trn_graphs, batch_size=args.batch_size, follow_batch=['graph_idx'])
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

    model = GIN(num_features, num_classes, args.dropout)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ce_loss = SoftCELoss() 

    if args.aug_size == 0.4:
        augward = [0.75, 1, 1.25, 1.5]
    elif args.aug_size == 0.5:
        augward = [1, 1, 1, 1]
    else:
        augward = [0.5, 1, 1.5, 2]
    
    augment_1 = Augment(trn_graphs, args.aug, aug_size=args.aug_size * augward[0])
    augment_2 = Augment(trn_graphs, args.aug, aug_size=args.aug_size * augward[1])
    augment_3 = Augment(trn_graphs, args.aug, aug_size=args.aug_size * augward[2]) 
    augment_4 = Augment(trn_graphs, args.aug, aug_size=args.aug_size * augward[3])

    augments = [augment_1, augment_2, augment_3, augment_4]
    augmentation_indices = []
    def random_augment(indices):
        randint = random.randint(0, len(augments)-1)
        augment_choice = augments[randint]
        augmentation_indices.append(randint)
        return augment_choice(indices)

    augmented_batches = [random_augment([i]) for i in range(len(trn_graphs))]

    combined_batch = Batch().from_data_list([b for batch in augmented_batches for b in batch.to_data_list()])
    aug_loader = DataLoader(combined_batch, batch_size=args.batch_size)
    
    if args.verbose > 0:
        print(' epochs\t   loss\ttrn_acc\tval_acc')

    out_list = dict(trn_loss=[], trn_acc=[], test_loss=[], test_acc=[])
    highest_test_acc = 0
    epoch_highest_test_acc = 0
    highest_train_acc = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        loss_sum = 0

        for batch_idx, batch in enumerate(trn_loader):
            batch = batch.to(device)
            _, output = model(batch.x, batch.edge_index, batch.batch)
            loss = ce_loss(output, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        for batch_idx, batch in enumerate(aug_loader):
            if args.data == "PROTEINS" or args.data == "PTC_MR":
                if batch_idx == len(trn_loader) - 1:
                    continue
            batch = batch.to(device)
            _, output = model(batch.x, batch.edge_index, batch.batch)
            loss = ce_loss(output, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()


        trn_loss = loss_sum / (len(trn_loader) * (2))
        trn_acc = eval_acc(model, trn_loader, device)
        test_loss = eval_loss(model, ce_loss, test_loader, device)
        test_acc = eval_acc(model, test_loader, device)

        if test_acc > highest_test_acc:
            highest_test_acc = test_acc
            highest_train_acc = trn_acc
            epoch_highest_test_acc = epoch
            print(f'BEST{epoch:3d}\t{trn_loss:7.4f}\t{trn_acc:7.4f}\t{test_acc:7.4f}')
        
        out_list['trn_loss'].append(trn_loss)
        out_list['trn_acc'].append(trn_acc)
        out_list['test_loss'].append(test_loss)
        out_list['test_acc'].append(test_acc)

        if args.verbose > 0 and (epoch) % args.verbose == 0:
            print(f'{epoch:7d}\t{trn_loss:7.4f}\t{trn_acc:7.4f}\t{test_acc:7.4f}')

    if args.path:
        out = {arg: getattr(args, arg) for arg in vars(args)}
        out['all'] = out_list
        out['best_epoch'] = epoch_highest_test_acc
        out['best_trn_acc'] = highest_train_acc
        out['best_test_acc'] = highest_test_acc
        with open(args.path, 'w') as file:
            json.dump(out, file)
    
    print(f'Training accuracy: {highest_train_acc}')
    print(f'Test accuracy: {highest_test_acc}')
    print(f"Epoch: {epoch_highest_test_acc}")

    print(json.dumps(out))


if __name__ == '__main__':
    main()
