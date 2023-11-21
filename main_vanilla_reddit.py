import os
import random
import argparse

from torch_geometric.datasets import Reddit2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pdb
import copy
import wandb
import warnings, pdb, pickle, time

import torch_sparse
from torch_sparse import SparseTensor
import scipy.sparse as ssp
from net_vanilla import GCN, SAGE
from pruning import setup_seed

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def accuracy(output, labels):
    output = F.log_softmax(output, dim=-1)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def initialize_model(data, args):
    out_channels = args.n_classes
    if args.net == 'GCN':
        model = GCN(data.x.shape[1], args.embedding_dim, out_channels, args.n_layers, \
                data.adj_t, args.dropout, args).to(device)
    elif args.net == 'SAGE':
        model = SAGE(data.x.shape[1], args.embedding_dim, out_channels, args.n_layers, \
                data.adj_t, args.dropout, args).to(device)
    return model

def run_fix_mask(args, seed):
    ## Load dataset.
    setup_seed(seed)
    dataset = Reddit2('dataset')
    data = dataset[0]
    values = torch.ones(data.edge_index.shape[1])
    data.adj_t = SparseTensor.from_edge_index(data.edge_index, values)
    data = data.to(device)
    
    args.n_classes = data.y.max().item() + 1
    args.n_nodes = data.num_nodes
    args.n_edges = data.adj_t.nnz()
    
    ## Initialize GNN.
    model = initialize_model(data, args)
    
    ### make sure to disable the gradient flow
    for name, param in model.named_parameters():
        if 'mask' in name:
            param.requires_grad = False

    ## Start training.
    best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc': 0}
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(args.total_epoch):
        model.train()
        optimizer.zero_grad()
        
        output= model(data.x, data.adj_t)
        loss = loss_func(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output = model(data.x, data.adj_t, val_test=True, compute_macs=True)
            loss_val = loss_func(output[data.val_mask], data.y[data.val_mask])
            acc_val = accuracy(output[data.val_mask].to(device), data.y[data.val_mask].to(device))
            acc_test = accuracy(output[data.test_mask].to(device), data.y[data.test_mask].to(device))
        
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['val_acc'] = acc_val
                best_val_acc['test_acc'] = acc_test
                best_val_acc['epoch'] = epoch
        
        print("(Fix Mask) Epoch:[{}] Val:[{:.2f}] Test:[{:.2f}] | Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
                 .format(epoch, acc_val * 100, 
                                acc_test * 100, 
                                best_val_acc['val_acc'] * 100, 
                                best_val_acc['test_acc'] * 100, 
                                best_val_acc['epoch']))
    return best_val_acc['test_acc'], best_val_acc['epoch']


def parser_loader():
    parser = argparse.ArgumentParser(description='GLT')
    parser.add_argument('--type', type=str, default='proposed', 
                        help='name of the pruning framework')
    
    parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--net', type=str, default='GCN', 
                        help='backbone architecture')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--total_epoch', type=int, default=200)
    parser.add_argument('--mask_epoch', type=int, default=200)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parser_loader()
    print(args)

    seed_dict = {'reddit': 0}
    seed = seed_dict[args.dataset]

    final_acc_test, final_epoch_list = run_fix_mask(args, seed)
    adj_spar, wei_spar = 100, 100
    
    print("=" * 120)
    print("syd : Final Test Acc:[{:.2f}] Adj:[{:.2f}%] Wei:[{:.2f}%]"
        .format(final_acc_test * 100, adj_spar, wei_spar))
    print("=" * 120)
            
    print('Finished,')