import torch
import torch.nn as nn
import pdb
import copy
import utils
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_sparse
from torch_sparse import SparseTensor
import torch.nn.functional as F

from typing import Union, Tuple, Optional, Callable
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, 
                                    OptTensor, PairTensor)
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
import torch_geometric
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn import JumpingKnowledge
import math, time

from torch_geometric.utils import softmax

from typing import Any, Dict, Optional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCNConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        if self.normalize:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim),
                    self.improved, self.add_self_loops)

            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim),
                    self.improved, self.add_self_loops)

        x = self.lin(x)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 adj_t, dropout, args):
        super().__init__()
        self.args = args
        self.edge_num = adj_t.nnz()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, bias=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, bias=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, bias=False))

        self.dropout = dropout

    def forward(self, x, adj_t, val_test=False):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            if val_test:
                continue
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class GATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = False, bias: bool = True, 
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.lin = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
        else:
            self.lin = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')

        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None):
        H, C = self.heads, self.out_channels
        
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin(x_src).view(-1, H, C)

        x = (x_src, x_dst)

        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)
        
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads, adj_t, args=None):
        super().__init__()
        self.args = args
        self.edge_num = adj_t.nnz()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.heads = heads

        self.adj_nonzero = self.edge_num
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads, bias=False, 
                                  dropout=0.6))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(heads * hidden_channels, hidden_channels, bias=False, 
                                      dropout=0.6))
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, 
                             concat=False, bias=False, dropout=0.))

    def forward(self, x, adj_t, val_test=False):
        x = F.dropout(x, p=0.6, training=self.training)
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.elu(x)
            x = F.dropout(x, p=0.6, training=self.training)
            
        x = self.convs[-1](x, adj_t)
        return x


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

        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.lin(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor):
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}(lin={})'.format(self.__class__.__name__, self.lin)


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, adj_t, args):
        super().__init__()
        self.args = args
        self.edge_num = adj_t.nnz()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                mlp = nn.Linear(in_channels, hidden_channels, bias=False)
            else:
                mlp = nn.Linear(hidden_channels, out_channels, bias=False)
            self.convs.append(GINConv(nn=mlp, train_eps=True))

        self.mlp = nn.Linear(hidden_channels, out_channels, bias=False)

    def forward(self, x, adj_t, val_test=False):
        xs = []
        for conv in self.convs:
            x = conv(x, adj_t)
            x = F.relu(x)
            xs.append(x)
        x = (self.mlp(xs[0]) + xs[1]) / 2
        return x
    

class SAGEConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                    size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                                x: OptPairTensor) -> Tensor:
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                    self.out_channels)
        

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 adj_t, dropout, args):
        super().__init__()
        self.args = args
        self.edge_num = adj_t.nnz()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, bias=False))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, bias=False))
        self.convs.append(SAGEConv(hidden_channels, out_channels, bias=False))

        self.dropout = dropout

    def forward(self, x, adj_t, val_test=False):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            if val_test:
                continue
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x
    
    
class TransformerConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(TransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_skip.reset_parameters()
        

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            out += x_r

        return out


    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        query = self.lin_query(x_i).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x_j).view(-1, self.heads, self.out_channels)

        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.lin_value(x_j).view(-1, self.heads, self.out_channels)

        out *= alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 adj_t, dropout, args):
        super(GraphTransformer, self).__init__()
        self.args = args
        self.edge_num = adj_t.nnz()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        self.num_heads = self.args.n_heads
        
        self.adj_nonzero = self.edge_num

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                TransformerConv(self.hidden_channels * self.num_heads, self.hidden_channels,
                            heads = self.num_heads,
                            concat = True,
                            dropout = self.args.dropout,
                            beta = False
                            )
                )

        self.convs[0] = TransformerConv(self.in_channels, self.hidden_channels,
                            heads = self.num_heads,
                            concat = True,
                            dropout = self.args.dropout,
                            beta = False
                            )

        self.convs[-1] = TransformerConv(self.hidden_channels * self.num_heads, self.out_channels,
                            heads = self.num_heads,
                            concat = False, # mean representation
                            dropout = self.args.dropout,
                            beta = False
                            )

        self.dropout = nn.Dropout(self.args.dropout)
        self.elu = F.elu

    def forward(self, x,  adj_t, val_test=False):
        x = self.dropout(x)
        for i in range(self.args.n_layers - 1):
            x = self.convs[i](x, adj_t)
            x = self.elu(x)
            x = self.dropout(x)
        
        x = self.convs[-1](x, adj_t)
        return x
    