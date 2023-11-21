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
from net_baseline import GCN, GAT, GIN, GraphTransformer
from pruning import setup_seed, add_weight_mask, subgradient_update_mask, update_rewind_weight, \
        get_best_epoch_mask, add_trainable_mask_noise, print_sparsity
from sinkhorn import sinkhorn

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_simulations = 20

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

def run_fix_mask(args, seed, adj_percent, wei_percent, rewind_weight_mask=None):
    ## Load dataset.
    setup_seed(seed)
    data = load_data(args.dataset)
    if args.net == 'GAT':
        data.adj_t = set_diag(data.adj_t)

    row, col, _ = data.adj_t.coo()
    data.edge_index = torch.stack((row, col))
    
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
            output= model(data.x, data.adj_t, val_test=True)
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

    return best_val_acc['test_acc'], best_val_acc['epoch'], adj_spar, wei_spar

# For UGS & Rethinking GLT #
def run_get_mask(args, seed, adj_percent, wei_percent, rewind_weight_mask=None):
    ## Load dataset.
    setup_seed(seed)
    data = load_data(args.dataset)
    if args.net == 'GAT':
        data.adj_t = set_diag(data.adj_t)
    row, col, _ = data.adj_t.coo()
    data.edge_index = torch.stack((row, col))
        
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
    acc_test = 0.0
    best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc':0}
    rewind_weight = copy.deepcopy(model.state_dict())

    if args.type == 'wasserstein': ### Rethinking GLT
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
    
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(args.mask_epoch):
        model.train()
        optimizer.zero_grad()

        output = model(data.x, data.adj_t, fix_mask=False)
        loss = loss_func(output[data.train_mask], data.y[data.train_mask])
        
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

        model.eval()
        with torch.no_grad():
            output = model(data.x, data.adj_t, val_test=True)
            loss_val = loss_func(output[data.val_mask], data.y[data.val_mask])
            acc_val = accuracy(output[data.val_mask].to(device), data.y[data.val_mask].to(device))
            acc_test = accuracy(output[data.test_mask].to(device), data.y[data.test_mask].to(device))
            
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['val_acc'] = acc_val
                best_val_acc['test_acc'] = acc_test
                best_val_acc['epoch'] = epoch
                # store differentiable masks
                best_epoch_mask, _ = get_best_epoch_mask(model, args, adj_percent, wei_percent)
                
        print("(Get Mask) Epoch:[{}] Val:[{:.2f}] Test:[{:.2f}] | Best Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
                .format(epoch, acc_val * 100, acc_test * 100, 
                            best_val_acc['val_acc'] * 100,  
                            best_val_acc['test_acc'] * 100,
                            best_val_acc['epoch']))

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
    parser.add_argument('--distill_reg', action='store_true', 
                        help='apply distill KL loss')
    # Rethinking GLT #
    parser.add_argument('--wasserstein_eta1', type=float, default=0.01)
    parser.add_argument('--wasserstein_eta2', type=float, default=0.01)
    parser.add_argument('--wasserstein_lamb', type=float, default=0.1,
                        help='the regularization coefficient for Wasserstein Distance')
    parser.add_argument('--eps', type=float, default=1e-3, 
                        help='sinkhorn entopry regularization parameter')
    parser.add_argument('--max_iters', type=int, default=10,
                        help='the maximum iterations for sinkhorn')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parser_loader()
    print(args)

    seed_dict = {'cora': 3846, 'citeseer': 2839, 'pubmed': 3333}
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
    
