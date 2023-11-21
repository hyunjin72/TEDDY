import os
import random
import argparse

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T

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
from net_vanilla import GCN, SAGE
from pruning import setup_seed

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def test_arxiv(model, data, adj_t, split_idx, evaluator):
    model.eval()

    output, macs, duration = model(data.x, adj_t, val_test=True, compute_macs=True)
    y_pred = output.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    return (train_acc, valid_acc, test_acc), macs, duration

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
    dataset = PygNodePropPredDataset(name=f'ogbn-arxiv',
                                        transform=T.ToSparseTensor())
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)
    
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    test_func = test_arxiv
        
    row, col, _ = data.adj_t.coo()
    edge_index = torch.stack((row, col))
    values = torch.ones(data.adj_t.nnz()).to(device)
    data.adj_t = SparseTensor.from_edge_index(edge_index, values, 
                        sparse_sizes=(data.num_nodes, data.num_nodes)).to(device)
        
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
    evaluator = Evaluator(name=f'ogbn-arxiv')
    for epoch in range(args.total_epoch):
        model.train()
        optimizer.zero_grad()
        
        output, _, _ = model(data.x, data.adj_t)
        
        loss = F.nll_loss(F.log_softmax(output[train_idx], dim=1), data.y.squeeze(1)[train_idx])
        loss.backward()
        
        optimizer.step()

        results = test_func(model, data, data.adj_t, split_idx, evaluator)
        
        if results[1] > best_val_acc['val_acc']:
            best_val_acc['val_acc'] = results[1]
            best_val_acc['test_acc'] = results[2]
            best_val_acc['epoch'] = epoch
    
        print(f'(FIX MASK) Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * results[0]:.2f}%, '
                      f'Valid: {100 * results[1]:.2f}% '
                      f'Test: {100 * results[2]:.2f}%')
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

    seed_dict = {'arxiv': 10}
    seed = seed_dict[args.dataset]
    
    final_acc_test, final_epoch_list = run_fix_mask(args, seed)
    adj_spar, wei_spar = 100, 100
    
    print("=" * 120)
    print("syd : Final Test Acc:[{:.2f}] Adj:[{:.2f}%] Wei:[{:.2f}%]"
        .format(final_acc_test * 100, adj_spar, wei_spar))
    print("=" * 120)
            
    print('Finished,')
