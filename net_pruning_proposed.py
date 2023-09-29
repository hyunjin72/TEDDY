import torch
import torch.nn as nn
import pdb
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_sparse
from torch_sparse import SparseTensor
import torch.nn.functional as F

from typing import Union, Tuple, Optional, Callable
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, 
                                    OptTensor)
from math import log
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag, matmul
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import reset, glorot, zeros
import pdb, pickle, os
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils.num_nodes import maybe_num_nodes
import math
from torch_geometric.utils import softmax
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GINConv(MessagePassing):
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GINConv, self).__init__(**kwargs)
        self.lin = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
            
        self.reset_parameters()
            
    def reset_parameters(self):
        reset(self.lin)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1] # x_self (original input `x`)
        if x_r is not None:
            out += (1 + self.eps) * x_r # x_aggr + (1 + eps) * x_self

        return self.lin(out)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor):
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}(lin={})'.format(self.__class__.__name__, self.lin)


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers, args):
        super().__init__()
        self.args = args
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers): # num_layers = 2
            if layer == 0:
                mlp = nn.Linear(in_channels, hidden_channels, bias=False)
            else:
                mlp = nn.Linear(hidden_channels, out_channels, bias=False)
            self.convs.append(GINConv(nn=mlp, train_eps=True))

        self.mlp = nn.Linear(hidden_channels, out_channels, bias=False)
    
    def forward(self, x, edge_index, pruned_values, val_test=False, compute_macs=False):
        mask = ~(pruned_values==0)
        edge_index = edge_index[:, mask]
        pruned_values = pruned_values[mask]
        adj_pruned = SparseTensor.from_edge_index(edge_index, pruned_values, 
                                        sparse_sizes=(self.args.n_nodes, self.args.n_nodes))
        if compute_macs:
            macs = gin_macs(self, adj_pruned, self.in_channels, self.hidden_channels, 
                            self.out_channels, self.num_layers, self.args)
        else:
            macs = None
            
        ts = time.time()
        xs = []
        for conv in self.convs:
            x = conv(x, adj_pruned)
            x = F.relu(x)
            xs.append(x)
        x = (self.mlp(xs[0]) + xs[1]) / 2
        tt = time.time()
        duration = tt - ts
        return x, macs, duration


