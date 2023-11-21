import os
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import load_data
import pdb
import copy
import wandb
import warnings, pdb, pickle, time

import torch_sparse
from torch_sparse import SparseTensor, set_diag
import scipy.sparse as ssp
from net_vanilla import GCN, GAT, GIN, GraphTransformer
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
    if args.net == 'GCN':
        model = GCN(data.x.shape[1], args.embedding_dim, args.n_classes, args.n_layers, \
                data.adj_t, args.dropout, args).to(device)
    elif args.net == 'GAT':
        model = GAT(data.x.shape[1], args.embedding_dim, args.n_classes, args.n_layers, \
                args.n_heads, data.adj_t, args=args).to(device)
    elif args.net == 'GIN':
        model = GIN(data.x.shape[1], args.embedding_dim, args.n_classes, args.n_layers, \
                data.adj_t, args).to(device)
    elif args.net == 'GT':
        model = GraphTransformer(data.x.shape[1], args.embedding_dim, args.n_classes, args.n_layers, \
                data.adj_t, args.dropout, args).to(device)
    return model

def run_fix_mask(args, seed):
    ## Load dataset.
    setup_seed(seed)
    data = load_data(args.dataset)
    if args.net == 'GAT':
        data.adj_t = set_diag(data.adj_t)
    adj_pruned = data.adj_t.clone()
    row, col, _ = data.adj_t.coo()
    data.edge_index = torch.stack((row, col))
        
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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
                                weight_decay=args.weight_decay)
        
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(args.total_epoch):
        model.train()
        optimizer.zero_grad()

        output = model(data.x, data.adj_t)
        
        loss = loss_func(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output = model(data.x, adj_pruned, val_test=True)
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
    parser.add_argument('--type', type=str, default='vanilla', 
                        help='name of the pruning framework')
    
    parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--net', type=str, default='GCN', 
                        help='backbone architecture')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--total_epoch', type=int, default=200)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parser_loader()
    print(args)

    seed_dict = {'cora': 3846, 'citeseer': 2839, 'pubmed': 3333}
    seed = seed_dict[args.dataset]

    adj_percent, wei_percent = 0, 0
    adj_spar, wei_spar = 100, 100
    
    final_acc_test, final_epoch_list = run_fix_mask(args, seed)
    
    print("=" * 120)
    print("syd : Final Test Acc:[{:.2f}] Adj:[{:.2f}%] Wei:[{:.2f}%]"
        .format(final_acc_test * 100, adj_spar, wei_spar))
    print("=" * 120)
            
    print('Finished,')
