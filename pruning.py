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
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import degree, add_self_loops
import random
import torch.nn.functional as F
import math, pdb, pickle
from scipy.linalg import inv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_edge_score(adj_t, args):
    row, col, _ = adj_t.coo()
    deg = torch_sparse.sum(adj_t, dim=1).float()
    deg_inv = deg.pow(-1)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    norm_A = torch_sparse.matmul(adj_t, deg_inv_sqrt.view(-1, 1)) # O(N)
    norm_A = norm_A.squeeze() * deg_inv
    
    ### Efficiently calculated using this part ###
    edge_importance = norm_A[row] * deg_inv[row] * norm_A[col] * deg_inv[col] # O(N)
    
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
    
    elif args.net == 'GT':
        for layer in range(args.n_layers):
            mask_train = nn.Parameter(torch.ones_like(model.convs[layer].lin_query.weight))
            mask_fixed = nn.Parameter(torch.ones_like(model.convs[layer].lin_query.weight), requires_grad=False)
            
            AddTrainableMask.apply(model.convs[layer].lin_query, 'weight', mask_train, mask_fixed)

            mask_train = nn.Parameter(torch.ones_like(model.convs[layer].lin_key.weight))
            mask_fixed = nn.Parameter(torch.ones_like(model.convs[layer].lin_key.weight), requires_grad=False)
            
            AddTrainableMask.apply(model.convs[layer].lin_key, 'weight', mask_train, mask_fixed)
            
            mask_train = nn.Parameter(torch.ones_like(model.convs[layer].lin_value.weight))
            mask_fixed = nn.Parameter(torch.ones_like(model.convs[layer].lin_value.weight), requires_grad=False)
            
            AddTrainableMask.apply(model.convs[layer].lin_value, 'weight', mask_train, mask_fixed)

            mask_train = nn.Parameter(torch.ones_like(model.convs[layer].lin_skip.weight))
            mask_fixed = nn.Parameter(torch.ones_like(model.convs[layer].lin_skip.weight), requires_grad=False)
            
            AddTrainableMask.apply(model.convs[layer].lin_skip, 'weight', mask_train, mask_fixed)


def subgradient_update_mask(model, args):
    model.adj_mask_train.grad.data.add_(args.lamb_a * torch.sign(model.adj_mask_train.data))
    
    if args.net in ['GCN', 'GAT', 'GIN']:
        for layer in range(args.n_layers):
            model.convs[layer].lin.weight_mask_train.grad.data.add_(args.lamb_w * torch.sign(model.convs[layer].lin.weight_mask_train.data))
            
            if args.net == 'GAT':
                model.convs[layer].att_src_mask_train.grad.data.add_(args.lamb_w * torch.sign(model.convs[layer].att_src_mask_train.data))
                model.convs[layer].att_dst_mask_train.grad.data.add_(args.lamb_w * torch.sign(model.convs[layer].att_dst_mask_train.data))
            
        if args.net == 'GIN':
            model.mlp.weight_mask_train.grad.data.add_(args.lamb_w * torch.sign(model.mlp.weight_mask_train.data))
    
    elif args.net == 'SAGE':
        for layer in range(args.n_layers):
            model.convs[layer].lin_l.weight_mask_train.grad.data.add_(args.lamb_w * torch.sign(model.convs[layer].lin_l.weight_mask_train.data))
            model.convs[layer].lin_r.weight_mask_train.grad.data.add_(args.lamb_w * torch.sign(model.convs[layer].lin_r.weight_mask_train.data))
    
    elif args.net == 'GT':
        for layer in range(args.n_layers):
            model.convs[layer].lin_query.weight_mask_train.grad.data.add_(args.lamb_w * torch.sign(model.convs[layer].lin_query.weight_mask_train.data))
            model.convs[layer].lin_key.weight_mask_train.grad.data.add_(args.lamb_w * torch.sign(model.convs[layer].lin_key.weight_mask_train.data))
            model.convs[layer].lin_value.weight_mask_train.grad.data.add_(args.lamb_w * torch.sign(model.convs[layer].lin_value.weight_mask_train.data))
            model.convs[layer].lin_skip.weight_mask_train.grad.data.add_(args.lamb_w * torch.sign(model.convs[layer].lin_skip.weight_mask_train.data))
    
        
def get_mask_distribution(model, args):
    adj_mask_tensor = model.adj_mask_train.flatten()
    nonzero = adj_mask_tensor.abs() > 0
    adj_mask_tensor = adj_mask_tensor[nonzero].detach().cpu()
    
    weight_mask_tensors = []
    if args.net in ['GCN', 'GAT', 'GIN']:
        for layer in range(args.n_layers):
            weight_mask_tensor = model.convs[layer].lin.weight_mask_train.flatten()
            nonzero = weight_mask_tensor.abs() > 0
            weight_mask_tensor = weight_mask_tensor[nonzero]
            weight_mask_tensors.append(weight_mask_tensor)
            
            if args.net == 'GAT':
                weight_mask_tensor = model.convs[layer].att_src_mask_train.flatten()
                nonzero = weight_mask_tensor.abs() > 0
                weight_mask_tensor = weight_mask_tensor[nonzero]
                weight_mask_tensors.append(weight_mask_tensor)

                weight_mask_tensor = model.convs[layer].att_dst_mask_train.flatten()
                nonzero = weight_mask_tensor.abs() > 0
                weight_mask_tensor = weight_mask_tensor[nonzero]
                weight_mask_tensors.append(weight_mask_tensor)
                
        if args.net == 'GIN':
            weight_mask_tensor = model.mlp.weight_mask_train.flatten()
            nonzero = weight_mask_tensor.abs() > 0
            weight_mask_tensor = weight_mask_tensor[nonzero]
            weight_mask_tensors.append(weight_mask_tensor)
    
    elif args.net == 'SAGE':
        for layer in range(args.n_layers):
            weight_mask_tensor = model.convs[layer].lin_l.weight_mask_train.flatten()
            nonzero = weight_mask_tensor.abs() > 0
            weight_mask_tensor = weight_mask_tensor[nonzero]
            weight_mask_tensors.append(weight_mask_tensor)
            
            weight_mask_tensor = model.convs[layer].lin_r.weight_mask_train.flatten()
            nonzero = weight_mask_tensor.abs() > 0
            weight_mask_tensor = weight_mask_tensor[nonzero]
            weight_mask_tensors.append(weight_mask_tensor)
            
    elif args.net == 'GT':
        for layer in range(args.n_layers):
            weight_mask_tensor = model.convs[layer].lin_query.weight_mask_train.flatten()
            nonzero = weight_mask_tensor.abs() > 0
            weight_mask_tensor = weight_mask_tensor[nonzero]
            weight_mask_tensors.append(weight_mask_tensor)
            
            weight_mask_tensor = model.convs[layer].lin_key.weight_mask_train.flatten()
            nonzero = weight_mask_tensor.abs() > 0
            weight_mask_tensor = weight_mask_tensor[nonzero]
            weight_mask_tensors.append(weight_mask_tensor)
            
            weight_mask_tensor = model.convs[layer].lin_value.weight_mask_train.flatten()
            nonzero = weight_mask_tensor.abs() > 0
            weight_mask_tensor = weight_mask_tensor[nonzero]
            weight_mask_tensors.append(weight_mask_tensor)
            
            weight_mask_tensor = model.convs[layer].lin_skip.weight_mask_train.flatten()
            nonzero = weight_mask_tensor.abs() > 0
            weight_mask_tensor = weight_mask_tensor[nonzero]
            weight_mask_tensors.append(weight_mask_tensor)
            
    weight_mask_tensors = torch.cat(weight_mask_tensors)
    
    return adj_mask_tensor, weight_mask_tensors.detach().cpu()
    

def prune_mask(mask_weight_tensor, threshold):
    ones  = torch.ones_like(mask_weight_tensor)
    zeros = torch.zeros_like(mask_weight_tensor) 
    mask = torch.where(mask_weight_tensor.abs() > threshold, ones, zeros)
    
    return mask


def prune_adj(edge_index, score, adj_percent):
    # We do not consider self-loops when comparing degree information.
    # Hence, in TEDDY, self-loops should not the subject for pruning.
    row, col = edge_index
    loops = row == col
    score[loops] = score.max().item()
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


def get_best_epoch_mask(model, args, adj_percent, wei_percent):
    mask_dict = {}
    adj_mask, wei_mask = get_mask_distribution(model, args) # , wei_mask
    adj_total = adj_mask.shape[0]
    adj_y, adj_i = torch.sort(adj_mask.abs())
    adj_thre_index = int(adj_total * adj_percent)
    adj_thre = adj_y[adj_thre_index]
    
    mask_dict['adj_mask'] = prune_mask(model.adj_mask_train.detach().cpu(), adj_thre)
    pruned_indices = (mask_dict['adj_mask']==0).nonzero().flatten()
    
    wei_total = wei_mask.shape[0]
    wei_y, wei_i = torch.sort(wei_mask.abs())
    wei_thre_index = int(wei_total * wei_percent)
    wei_thre = wei_y[wei_thre_index]
    
    if args.net in ['GCN', 'GAT', 'GIN']:
        for layer in range(args.n_layers):
            mask_dict[f"weight{layer}_mask"] = prune_mask(model.convs[layer].lin.weight_mask_train, wei_thre)
            
            if args.net == 'GAT':
                mask_dict[f"attn_src{layer}_mask"] = prune_mask(model.convs[layer].att_src_mask_train, wei_thre)
                mask_dict[f"attn_dst{layer}_mask"] = prune_mask(model.convs[layer].att_dst_mask_train, wei_thre)
                
        if args.net == 'GIN':
            mask_dict[f"mlp_mask"] = prune_mask(model.mlp.weight_mask_train, wei_thre)
    
    elif args.net == 'SAGE':
        for layer in range(args.n_layers):
            mask_dict[f"weight_l{layer}_mask"] = prune_mask(model.convs[layer].lin_l.weight_mask_train, wei_thre)
            mask_dict[f"weight_r{layer}_mask"] = prune_mask(model.convs[layer].lin_r.weight_mask_train, wei_thre)
            
    elif args.net == 'GT':
        for layer in range(args.n_layers):
            mask_dict[f"weight_query{layer}_mask"] = prune_mask(model.convs[layer].lin_query.weight_mask_train, wei_thre)
            mask_dict[f"weight_key{layer}_mask"] = prune_mask(model.convs[layer].lin_key.weight_mask_train, wei_thre)
            mask_dict[f"weight_value{layer}_mask"] = prune_mask(model.convs[layer].lin_value.weight_mask_train, wei_thre)
            mask_dict[f"weight_skip{layer}_mask"] = prune_mask(model.convs[layer].lin_skip.weight_mask_train, wei_thre)
           
    return mask_dict, pruned_indices


def random_pruning(model, args, adj_percent, wei_percent):
    mask_dict = {}
    adj_mask, wei_mask = get_mask_distribution(model, args) # gather all nonzero elements
    
    adj_total = adj_mask.numel()
    adj_pruned_num = int(adj_total * adj_percent)
    adj_index = random.sample([i for i in range(adj_total)], adj_pruned_num)
    
    prune_mask = torch.ones_like(adj_mask)
    prune_mask[adj_index] = 0
    mask_dict['adj_mask'] = prune_mask.clone()
    
    wei_total = wei_mask.numel()
    wei_pruned_num = int(wei_total * wei_percent)
    wei_index = random.sample([i for i in range(wei_total)], wei_pruned_num)
    
    prune_mask = torch.ones_like(wei_mask)
    prune_mask[wei_index] = 0
    
    start = 0
    if args.net in ['GCN', 'GAT', 'GIN']:
        for layer in range(args.n_layers):
            end = start + model.convs[layer].lin.weight_mask_train.numel()
            mask_dict[f"weight{layer}_mask"] = prune_mask[start:end].view_as(model.convs[layer].lin.weight_mask_train)
            start = end
            if args.net == 'GAT':
                end += model.convs[layer].att_src_mask_train.numel()
                mask_dict[f"attn_src{layer}_mask"] = prune_mask[start:end].view_as(model.convs[layer].att_src_mask_train)
                start = end
                end += model.convs[layer].att_dst_mask_train.numel()
                mask_dict[f"attn_dst{layer}_mask"] = prune_mask[start:end].view_as(model.convs[layer].att_dst_mask_train)
                start = end
        
        if args.net == 'GIN':
            end += model.mlp.weight_mask_train.numel()
            mask_dict[f"mlp_mask"] = prune_mask[start:end].view_as(model.mlp.weight_mask_train)
            start = end
    
    elif args.net == 'SAGE':
        for layer in range(args.n_layers):
            end = start + model.convs[layer].lin_l.weight_mask_train.numel()
            mask_dict[f"weight_l{layer}_mask"] = prune_mask[start:end].view_as(model.convs[layer].lin_l.weight_mask_train)
            start = end
            end += model.convs[layer].lin_r.weight_mask_train.numel()
            mask_dict[f"weight_r{layer}_mask"] = prune_mask[start:end].view_as(model.convs[layer].lin_r.weight_mask_train)
            start = end
            
    elif args.net == 'GT':
        for layer in range(args.n_layers):
            end = start + model.convs[layer].lin_query.weight_mask_train.numel()
            mask_dict[f"weight_query{layer}_mask"] = prune_mask[start:end].view_as(model.convs[layer].lin_query.weight_mask_train)
            start = end
            end += model.convs[layer].lin_key.weight_mask_train.numel()
            mask_dict[f"weight_key{layer}_mask"] = prune_mask[start:end].view_as(model.convs[layer].lin_key.weight_mask_train)
            start = end
            end += model.convs[layer].lin_value.weight_mask_train.numel()
            mask_dict[f"weight_value{layer}_mask"] = prune_mask[start:end].view_as(model.convs[layer].lin_value.weight_mask_train)
            start = end
            end += model.convs[layer].lin_skip.weight_mask_train.numel()
            mask_dict[f"weight_skip{layer}_mask"] = prune_mask[start:end].view_as(model.convs[layer].lin_skip.weight_mask_train)
            start = end
    
    return mask_dict

   
def print_sparsity(model, args):
    adj_mask_nonzero, wei_mask_nonzero = get_mask_distribution(model, args)

    adj_total = model.adj_mask_fixed.numel()
    adj_nonzero = adj_mask_nonzero.numel()
    adj_spar = adj_nonzero * 100 / adj_total # unpruned edges ratio
    
    wei_total = 0
    for name, param in model.named_parameters():
        if 'mask' in name and 'fixed' in name:
            if 'adj' not in name:
                wei_total += param.numel()
    wei_nonzero = wei_mask_nonzero.numel()
    
    wei_spar = wei_nonzero * 100 / wei_total # unpruned weights ratio

    print("-" * 100)
    print("Sparsity: Adj:[{:.2f}%] Wei:[{:.2f}%]"
    .format(adj_spar, wei_spar))
    print("-" * 100)
    
    return adj_spar, wei_spar


def print_sparsity_random(model, values_ori, values_pruned, args):
    _, wei_mask_nonzero = get_mask_distribution(model, args)

    adj_spar = values_pruned.numel() * 100 / values_ori.numel()
    
    wei_total = 0
    for name, param in model.named_parameters():
        if 'mask' in name and 'fixed' in name:
            if 'adj' not in name:
                wei_total += param.numel()
    wei_nonzero = wei_mask_nonzero.numel()
    
    wei_spar = wei_nonzero * 100 / wei_total # unpruned weights ratio

    print("-" * 100)
    print("Sparsity: Adj:[{:.2f}%] Wei:[{:.2f}%]"
    .format(adj_spar, wei_spar))
    print("-" * 100)
    
    return adj_spar, wei_spar


def add_trainable_mask_noise(model, c=1e-5):
    for name, param in model.named_parameters():
        if 'mask' in name and 'train' in name:
            param.requires_grad = False
            rand = (2 * torch.rand(param.shape) - 1) * c
            rand = rand.to(device)
            rand = rand * param
            param.add_(rand)
            param.requires_grad = True


def update_rewind_weight(rewind_weight, final_mask_dict, args):
    ### adj. mask update
    rewind_weight['adj_mask_train'] = final_mask_dict['adj_mask']
    rewind_weight['adj_mask_fixed'] = final_mask_dict['adj_mask']
    
    ### weight mask update
    if args.net in ['GCN', 'GAT', 'GIN']:
        for layer in range(args.n_layers):
            rewind_weight[f"convs.{layer}.lin.weight_mask_train"] = final_mask_dict[f"weight{layer}_mask"]
            rewind_weight[f"convs.{layer}.lin.weight_mask_fixed"] = final_mask_dict[f"weight{layer}_mask"]
            
            if args.net == 'GAT':
                rewind_weight[f"convs.{layer}.att_src_mask_train"] = final_mask_dict[f"attn_src{layer}_mask"]
                rewind_weight[f"convs.{layer}.att_src_mask_fixed"] = final_mask_dict[f"attn_src{layer}_mask"]
                
                rewind_weight[f"convs.{layer}.att_dst_mask_train"] = final_mask_dict[f"attn_dst{layer}_mask"]
                rewind_weight[f"convs.{layer}.att_dst_mask_fixed"] = final_mask_dict[f"attn_dst{layer}_mask"]
        
        if args.net == 'GIN':
            rewind_weight[f"mlp.weight_mask_train"] = final_mask_dict[f"mlp_mask"]
            rewind_weight[f"mlp.weight_mask_fixed"] = final_mask_dict[f"mlp_mask"]

    elif args.net == 'SAGE':
        for layer in range(args.n_layers):
            rewind_weight[f"convs.{layer}.lin_l.weight_mask_train"] = final_mask_dict[f"weight_l{layer}_mask"]
            rewind_weight[f"convs.{layer}.lin_l.weight_mask_fixed"] = final_mask_dict[f"weight_l{layer}_mask"]
            
            rewind_weight[f"convs.{layer}.lin_r.weight_mask_train"] = final_mask_dict[f"weight_r{layer}_mask"]
            rewind_weight[f"convs.{layer}.lin_r.weight_mask_fixed"] = final_mask_dict[f"weight_r{layer}_mask"]
            
    elif args.net == 'GT':
        for layer in range(args.n_layers):
            rewind_weight[f"convs.{layer}.lin_query.weight_mask_train"] = final_mask_dict[f"weight_query{layer}_mask"]
            rewind_weight[f"convs.{layer}.lin_query.weight_mask_fixed"] = final_mask_dict[f"weight_query{layer}_mask"]
            
            rewind_weight[f"convs.{layer}.lin_key.weight_mask_train"] = final_mask_dict[f"weight_key{layer}_mask"]
            rewind_weight[f"convs.{layer}.lin_key.weight_mask_fixed"] = final_mask_dict[f"weight_key{layer}_mask"]

            rewind_weight[f"convs.{layer}.lin_value.weight_mask_train"] = final_mask_dict[f"weight_value{layer}_mask"]
            rewind_weight[f"convs.{layer}.lin_value.weight_mask_fixed"] = final_mask_dict[f"weight_value{layer}_mask"]
            
            rewind_weight[f"convs.{layer}.lin_skip.weight_mask_train"] = final_mask_dict[f"weight_skip{layer}_mask"]
            rewind_weight[f"convs.{layer}.lin_skip.weight_mask_fixed"] = final_mask_dict[f"weight_skip{layer}_mask"]
         
    return rewind_weight


