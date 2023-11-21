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
from net_baseline import GCN, SAGE
from pruning import setup_seed, add_weight_mask, subgradient_update_mask, update_rewind_weight, \
        get_best_epoch_mask, add_trainable_mask_noise, print_sparsity
from sinkhorn import sinkhorn

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_simulations = 20

@torch.no_grad()
def test_arxiv(model, data, adj_t, split_idx, evaluator):
    model.eval()

    output = model(data.x, adj_t, val_test=True)
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

    return (train_acc, valid_acc, test_acc)

def initialize_model(data, args):
    out_channels = args.n_classes
    if args.net == 'GCN':
        model = GCN(data.x.shape[1], args.embedding_dim, out_channels, args.n_layers, \
                data.adj_t, args.dropout, args).to(device)
    elif args.net == 'SAGE':
        model = SAGE(data.x.shape[1], args.embedding_dim, out_channels, args.n_layers, \
                data.adj_t, args.dropout, args).to(device)
    return model

def run_fix_mask(args, seed, adj_percent, wei_percent, rewind_weight_mask=None):
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
    data.adj_t = SparseTensor.from_edge_index(edge_index, values, sparse_sizes=(data.num_nodes, data.num_nodes)).to(device)
        
    args.n_classes = data.y.max().item() + 1
    args.n_nodes = data.num_nodes
    args.n_edges = data.adj_t.nnz()
    
    ## Initialize GNN.
    model = initialize_model(data, args)

    ## Start pruning.
    add_weight_mask(model, args)
    assert rewind_weight_mask is not None
    # load differentiable mask at previous pruning regime
    model.load_state_dict(rewind_weight_mask, strict=False)
    print('FIX MASK,')

    adj_spar, wei_spar = print_sparsity(model, args)
    
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
        
        output = model(data.x, data.adj_t)
        
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

    return best_val_acc['test_acc'], best_val_acc['epoch'], adj_spar, wei_spar

# For UGS & Rethinking GLT #
def run_get_mask(args, seed, adj_percent, wei_percent, rewind_weight_mask=None):
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
    
    ## Differentiable mask addition & Noise injection.
    add_weight_mask(model, args)
    if rewind_weight_mask:
        model.load_state_dict(rewind_weight_mask, strict=False)
    add_trainable_mask_noise(model, c=1e-5)
    print('GET MASK,')

    ## Start training.
    best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc':0}
    rewind_weight = copy.deepcopy(model.state_dict())
    evaluator = Evaluator(name=f'ogbn-arxiv')
    
    if args.type == 'wasserstein':
        graph_params = []
        weight_params = []
        for name, param in model.named_parameters():
            if 'adj_mask_train' in name:
                if param.requires_grad:
                    graph_params.append(param)
            else:
                if param.requires_grad:
                    weight_params.append(param)

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.wasserstein_eta1,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)
        optimizer.zero_grad()
    
    for epoch in range(args.mask_epoch):
        model.train()
        optimizer.zero_grad()

        output = model(data.x, data.adj_t, fix_mask=False)
        loss = F.nll_loss(F.log_softmax(output[train_idx], dim=1), data.y.squeeze(1)[train_idx])
        
        if args.type == 'wasserstein':
            wd_loss = 0.0
            for c in range(args.n_classes):
                index_c = (torch.argmax(output, dim=1, keepdim=False) == c)
                if torch.sum(index_c).item() == 0 or torch.sum(index_c).item() == args.n_nodes:
                    pass
                else:
                    output_c = output[index_c]
                    output_not_c = output[~index_c]
                    wd_dist_c, _, _ = sinkhorn(x=torch.softmax(output_c, dim=1), 
                                               y=torch.softmax(output_not_c, dim=1),
                                               p=2,
                                               w_x=None, 
                                               w_y=None,
                                               eps=args.eps,
                                               max_iters=args.max_iters)
                    wd_loss -= wd_dist_c
            loss = loss + args.wasserstein_lamb * wd_loss
            loss.backward()
            optimizer.step()

            model.adj_mask_train.data.clamp_(min=0.0, max=1.0)
        else:
            loss.backward()
            subgradient_update_mask(model, args) # l1 norm
            optimizer.step()
        
        results = test_func(model, data, data.adj_t, split_idx, evaluator)
  
        if results[1] > best_val_acc['val_acc']:
            best_val_acc['val_acc'] = results[1]
            best_val_acc['test_acc'] = results[2]
            best_val_acc['epoch'] = epoch
            # store differentiable masks
            best_epoch_mask, _ = get_best_epoch_mask(model, args, adj_percent, wei_percent)

    return best_epoch_mask, rewind_weight


def parser_loader():
    parser = argparse.ArgumentParser(description='GLT')
    parser.add_argument('--wandb', action='store_true', 
                        help='log model performance using wandb')
    parser.add_argument('--exp_name', type=str, default='exp', 
                        help='name of the wandb experiment')
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
    
    parser.add_argument('--pruning_percent_wei', type=float, default=0.1,
                        help='weight pruning percentage per simulation')
    parser.add_argument('--pruning_percent_adj', type=float, default=0.1,
                        help='edge pruning percentage per simulation')
   
    # UGS #
    parser.add_argument('--lamb_a', type=float, default=0.0001, 
                        help='scale edge sparse rate (default: 0.0001)')
    parser.add_argument('--lamb_w', type=float, default=0.0001, 
                        help='scale weight sparse rate (default: 0.0001)')
    # Rethinking GLT #
    parser.add_argument('--wasserstein_lamb', type=float, default=0.1,
                        help='the regularization coefficient for Wasserstein Distance')
    parser.add_argument('--eps', type=float, default=1e-3, 
                        help='sinkhorn entopry regularization parameter')
    parser.add_argument('--max_iters', type=int, default=10,
                        help='the maximum iterations for sinkhorn')
    parser.add_argument('--wasserstein_eta1', type=float, default=1e-2,
                        help='the learning rate for adj_mask_train (eta1 in paper)')
    parser.add_argument('--wasserstein_eta2', type=float, default=1e-2,
                        help='the learning rate for weight')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parser_loader()
    print(args)

    seed_dict = {'arxiv': 10}
    seed = seed_dict[args.dataset]

    rewind_weight = None
    percent_list = [(1 - (1 - args.pruning_percent_adj) ** (i + 1), 
                     1 - (1 - args.pruning_percent_wei) ** (i + 1)) for i in range(n_simulations)]

    for p in range(n_simulations):
        adj_percent, wei_percent = percent_list[p]
        ######################################## UGS & WD-GLT ########################################
        epoch_mask, rewind_weight = run_get_mask(args, seed, 
                                    args.pruning_percent_adj, 
                                    args.pruning_percent_wei, 
                                    rewind_weight_mask=rewind_weight)
        
        rewind_weight = update_rewind_weight(rewind_weight, epoch_mask, args)
        #############################################################################################
        
        final_acc_test, final_epoch_list, adj_spar, wei_spar = run_fix_mask(args, seed, 
                                                                adj_percent, wei_percent,
                                                                rewind_weight_mask=rewind_weight)

        print("=" * 120)
        print("syd : Sparsity:[{}] - Final Test Acc:[{:.2f}] Adj:[{:.2f}%] Wei:[{:.2f}%]"
            .format(p + 1,  final_acc_test * 100, adj_spar, wei_spar))
        print("=" * 120)
    
    print('Finished,')
    
