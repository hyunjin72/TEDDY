import torch
import torch.nn as nn
from abc import ABC
import random
import os
import matplotlib.pyplot as plt
import torch.nn.init as init
import math

import numpy as np
import networkx as nx
import torch_sparse
from torch_sparse import SparseTensor, to_scipy, from_scipy, to_torch_sparse, from_torch_sparse
from torch_sparse import matmul, spspmm
from torch_geometric.utils import add_self_loops, degree
from scipy.linalg import fractional_matrix_power, inv
import scipy.sparse as sp
from scipy.sparse import identity
# from torch_geometric.transforms import GCNNorm
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import degree, add_self_loops
import random
import torch.nn.functional as F
import math, pdb, pickle
from scipy.linalg import inv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prune_adj(score, adj_percent):
    n_total = len(score)
    adj_thre_index = int(n_total * adj_percent)
    low_score_edges = torch.topk(score, adj_thre_index, largest=False)[1]

    pruning_mask = torch.zeros(n_total, dtype=torch.bool).to(device)
    pruning_mask[low_score_edges] = True
    score[pruning_mask] = 0
    score[~pruning_mask] = 1
    
    n_remained = (pruning_mask==False).int().sum().item()
    adj_spar = n_remained * 100 / n_total
    return score, low_score_edges.detach().cpu().tolist(), adj_spar

def compute_edge_mp(adj_t, args):
    row, col, _ = adj_t.coo()
    deg = torch_sparse.sum(adj_t, dim=1)
    deg_inv = deg.pow(-1)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    norm_A = torch_sparse.matmul(adj_t, deg_inv_sqrt.view(-1, 1))
    norm_A = norm_A.squeeze() * deg_inv
    
    transition = norm_A.view(-1, 1) * norm_A.view(1, -1) # TODO - should we make norm_A to a unit vector?
    transition = transition * deg_inv.view(-1, 1) * deg_inv.view(1, -1)
    # importance = args.teleport * (torch.eye(adj_t.size(0)).cuda() + ((1 - args.teleport) * transition) / (args.teleport))
    # edge_importance = importance[row, col]
    edge_importance = transition[row, col]
    return edge_importance


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


class AddTrainableMask(ABC):
    
    _tensor_name: str
    
    def __init__(self):
        pass
    
    def __call__(self, module, inputs):
        setattr(module, self._tensor_name, self.apply_mask(module))

    def apply_mask(self, module):

        mask_train = getattr(module, self._tensor_name + "_mask_train")
        mask_fixed = getattr(module, self._tensor_name + "_mask_fixed")
        orig_weight = getattr(module, self._tensor_name + "_orig_weight")
        pruned_weight = mask_train * mask_fixed * orig_weight 
        return pruned_weight

    @classmethod
    def apply(cls, module, name, mask_train, mask_fixed, *args, **kwargs):
        method = cls(*args, **kwargs)  
        method._tensor_name = name
        orig = getattr(module, name)

        # Add parameters inside the module
        module.register_parameter(name + "_mask_train", mask_train.to(dtype=orig.dtype)) # weight_mask_train
        module.register_parameter(name + "_mask_fixed", mask_fixed.to(dtype=orig.dtype)) # weight_mask_fixed
        module.register_parameter(name + "_orig_weight", orig) # weight_orig_weight
        del module._parameters[name]

        setattr(module, name, method.apply_mask(module)) 
        module.register_forward_pre_hook(method)
        return method

# Weight mask #
def add_weight_mask(model, args):
    if args.net in ['GCN', 'GAT', 'GIN']:
        for layer in range(args.n_layers):
            mask_train = nn.Parameter(torch.ones_like(model.convs[layer].lin.weight))
            mask_fixed = nn.Parameter(torch.ones_like(model.convs[layer].lin.weight), requires_grad=False)
            
            AddTrainableMask.apply(model.convs[layer].lin, 'weight', mask_train, mask_fixed)
            
        if args.net == 'GIN':
            mask_train = nn.Parameter(torch.ones_like(model.mlp.weight))
            mask_fixed = nn.Parameter(torch.ones_like(model.mlp.weight), requires_grad=False)
            
            AddTrainableMask.apply(model.mlp, 'weight', mask_train, mask_fixed)
            
    elif args.net == 'SAGE':
        for layer in range(args.n_layers):
            mask_train = nn.Parameter(torch.ones_like(model.convs[layer].lin_l.weight))
            mask_fixed = nn.Parameter(torch.ones_like(model.convs[layer].lin_l.weight), requires_grad=False)
            
            AddTrainableMask.apply(model.convs[layer].lin_l, 'weight', mask_train, mask_fixed)

            mask_train = nn.Parameter(torch.ones_like(model.convs[layer].lin_r.weight))
            mask_fixed = nn.Parameter(torch.ones_like(model.convs[layer].lin_r.weight), requires_grad=False)
            
            AddTrainableMask.apply(model.convs[layer].lin_r, 'weight', mask_train, mask_fixed)
   