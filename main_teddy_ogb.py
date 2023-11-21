import os
import random
import argparse

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import copy
import wandb
import warnings, pdb, pickle, time

import torch_sparse
from torch_sparse import SparseTensor
from net_teddy import GCN, SAGE
from pruning import compute_edge_score, setup_seed, prune_adj

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_simulations = 20

@torch.no_grad()
def test_arxiv(model, data, edge_index, pruned_values, split_idx, evaluator):
    model.eval()

    output = model(data.x, edge_index, pruned_values, val_test=True)
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

    return (train_acc, valid_acc, test_acc), output
   
def initialize_model(data, args):
    out_channels = args.n_classes
    if args.net == 'GCN':
        model = GCN(data.x.shape[1], args.embedding_dim, out_channels, args.n_layers, \
                args.dropout, args).to(device)
    elif args.net == 'SAGE':
        model = SAGE(data.x.shape[1], args.embedding_dim, out_channels, args.n_layers, \
                args.dropout, args).to(device)
    return model


def run_fix_mask(args, seed, adj_percent, wei_percent, prev_pruned_indices=None, 
                 edge_score=None, output_ori=None, cnt=0, init=False):
    # 1. Load dataset
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
    values_ori = torch.ones_like(row).float()
    edge_index = torch.stack((row, col))
    data.adj_t = SparseTensor.from_edge_index(edge_index, values_ori, 
                        sparse_sizes=(data.num_nodes, data.num_nodes)).to(device)

    args.n_classes = data.y.max().item() + 1
    args.n_nodes = data.num_nodes
    args.n_edges = data.adj_t.nnz()
    
    # 2. Initialize GNN
    model = initialize_model(data, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
                                 weight_decay=args.weight_decay)  
    evaluator = Evaluator(name=f'ogbn-arxiv')
        
    if init:
        pruned_values = values_ori
        # 3-1. Compute degree-based edge scores
        edge_score = compute_edge_score(data.adj_t, args)
    
    # for unstructured weight pruning, 
    # we initialize necessary quantities
    param_vec = []
    param_shapes = {}
    param_masks = {}
    n_params = {}
    total_params = 0
    for name, param in model.named_parameters():
        param_vec.append(param.data.view(-1))
        param_shapes[name] = param.data.shape
        param_masks[name] = torch.ones_like(param.data)
        n_params[name] = param.data.numel()
        total_params += param.data.numel()
    param_vec = torch.cat(param_vec, dim=0)
    print ("Total number of trainable parameters: {}".format(total_params))

    # 3-2. Prune edges
    if not init:
        if cnt > 0: ## make sure that pruned edges in previous simulations not be revived
            edge_score[prev_pruned_indices] = 0
        pruned_values, pruned_indices, adj_spar = prune_adj(edge_index, edge_score, adj_percent)
    
    # 4. Start training
    best_output = None
    best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc': 0}
    for epoch in range(args.total_epoch):
        model.train()
        optimizer.zero_grad()
                 
        output = model(data.x, edge_index, pruned_values)
        
        loss = F.nll_loss(F.log_softmax(output[train_idx], dim=-1), data.y.squeeze(1)[train_idx])
        if init or not args.distill_reg:
            loss.backward()
        else:
            p_s = F.log_softmax(output, dim=1)
            p_t = F.softmax(output_ori, dim=1)
            loss_dt = F.kl_div(p_s, p_t, size_average=False) / output.size(0)
            loss_tot = loss + args.lamb * loss_dt
            loss_tot.backward()
        
        optimizer.step()
        if not init:
            param_vec = []
            for name, param in model.named_parameters():
                param_vec.append(param.data.view(-1))
            param_vec = torch.cat(param_vec, dim=0)
            
            # projection onto the L0 ball
            # is nothing but zeroing the smallest coordinates
            n_pruning = math.ceil(total_params * wei_percent)
            smallest_edge_indices = torch.topk(param_vec.data.abs(), 
                                                n_pruning,
                                                largest=False)[1]
            total_mask = torch.ones_like(param_vec)
            total_mask[smallest_edge_indices] = 0.0

            start_index = 0
            for name, param in model.named_parameters():
                end_index = start_index + n_params[name]
                mask = total_mask[start_index:end_index].reshape(param_shapes[name])
                param_masks[name] = mask
                start_index = end_index

                # L0 projection (removing smallest h entries)
                param.data = param.data * mask.data
            wei_spar = 100.0 * (total_params - n_pruning) / total_params
          
        # 5. Per-step validation
        model.eval()
        with torch.no_grad():
            results, output = test_func(model, data, edge_index, pruned_values,
                                        split_idx, evaluator)
            
        if results[1] > best_val_acc['val_acc']:
            best_val_acc['val_acc'] = results[1]
            best_val_acc['test_acc'] = results[2]
            best_val_acc['epoch'] = epoch
            best_output = output.detach()
                
        print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * results[0]:.2f}%, '
                      f'Valid: {100 * results[1]:.2f}% '
                      f'Test: {100 * results[2]:.2f}%')
        
    if init: 
        pruned_indices = None
        adj_spar = 100
        wei_spar = 100
        output_ori = best_output.clone()
        
    return best_val_acc["test_acc"], adj_spar, wei_spar, pruned_indices, \
            edge_score, output_ori


def parser_loader():
    parser = argparse.ArgumentParser(description='Graph-Pruning')
    parser.add_argument('--wandb', action='store_true', 
                        help='log model performance using wandb')
    parser.add_argument('--exp_name', type=str, default='exp', 
                        help='name of the wandb experiment')
    parser.add_argument('--type', type=str, default='proposed', 
                        help='name of the pruning framework')
    
    parser.add_argument('--dataset', type=str, default='citeseer', 
                        help='name of the dataset')
    parser.add_argument('--net', type=str, default='GCN', 
                        help='backbone architecture')
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--total_epoch', type=int, default=200)
    
    parser.add_argument('--pruning_percent_adj', type=float, default=0.1, 
                        help='pruning percentage per simulation')
    parser.add_argument('--pruning_percent_wei', type=float, default=0.1, 
                        help='weight pruning percentage per simulation')
    
    
    parser.add_argument('--distill_reg', action='store_true', 
                        help='enable distillation regularization')
    parser.add_argument('--lamb', type=float, default=0.1, 
                        help='regularization coef. for distill KL loss')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parser_loader()
    print(args)

    seed_dict = {'arxiv': 10}
    seed = seed_dict[args.dataset]

    accuracies, adj_spars, wei_spars = [], [], []
    percent_list = [(1 - (1 - args.pruning_percent_adj) ** (i + 1), 
                     1 - (1 - args.pruning_percent_wei) ** (i + 1)) for i in range(n_simulations)]

    ########################################## Pre-training with full edges ##########################################
    print('Start Pretraining,')
    final_acc_test, adj_spar, wei_spar, _, edge_score, output_ori = run_fix_mask(args, seed, adj_percent=0, wei_percent=0, 
                                                                    cnt=0, init=True)
    print("=" * 120)
    print("syd : Sparsity:[{}] - Final Test Acc:[{:.2f}] Adj:[{:.2f}%] Wei:[{:.2f}%]"
        .format(0,  final_acc_test * 100, adj_spar, wei_spar))
    print("=" * 120)
    ##################################################################################################################
    
    ################################################# Pruning regime #################################################
    pruned_indices, epoch_mask = None, None
    print('Start training,')
    for p in range(n_simulations):
        adj_percent, wei_percent = percent_list[p]
        ts = time.time()
        final_acc_test, adj_spar, wei_spar, pruned_indices, edge_score, output_ori = run_fix_mask(args, seed, 
                                                                                    adj_percent, wei_percent, cnt=p, 
                                                                                    init=False, prev_pruned_indices=pruned_indices, 
                                                                                    edge_score=edge_score, output_ori=output_ori)
        
        print("=" * 120)
        print("syd : Sparsity:[{}] - Final Test Acc:[{:.2f}] Adj:[{:.2f}%] Wei:[{:.2f}%]"
            .format(p + 1,  final_acc_test * 100, adj_spar, wei_spar))
        print("=" * 120)
    ##################################################################################################################
    
    print('Finished,')