import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter

import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree
from typing import Tuple


class GeneralizedRelationalConv(MessagePassing):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    # TODO for compile() - doesn't work currently
    # propagate_type = {"edge_index": torch.LongTensor, "size": Tuple[int, int]}

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, conv_emb_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", project_conv_emb=False,):
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.project_conv_emb = project_conv_emb

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)

        if self.project_conv_emb:
            self.proj_conv_emb = nn.Sequential(
                    nn.Linear(conv_emb_dim, conv_emb_dim),
                    nn.ReLU(),
                    nn.Linear(conv_emb_dim, conv_emb_dim)
                )
  
           

    def forward(self, input, query, boundary, edge_index, edge_type, conv_edge_embedding, size, edge_weight=None):
        batch_size = len(query)
        #print (f"conv_edge_embedding.shape: {conv_edge_embedding.shape}")
        #raise NotImplementedError
        if self.project_conv_emb:
            conv_edge_embedding = self.proj_conv_emb(conv_edge_embedding)

        if edge_weight is None:
            edge_weight = torch.ones(len(edge_type), device=input.device)
        #todo Edge_attr
        # note that we send the initial boundary condition (node states at layer0) to the message passing
        output = self.propagate(input=input, boundary=boundary, edge_index=edge_index,
                                edge_type=edge_type, size=size, edge_weight=edge_weight, edge_attr = conv_edge_embedding)
        return output

    def propagate(self, edge_index, size=None, **kwargs):
        
       
        if kwargs["edge_weight"].requires_grad or self.message_func == "rotate":
            # the rspmm cuda kernel only works for TransE and DistMult message functions
            # otherwise we invoke separate message & aggregate functions
            return super(GeneralizedRelationalConv, self).propagate(edge_index, size, **kwargs)

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        # in newer PyG, 
        # __check_input__ -> _check_input()
        # __collect__ -> _collect()
        # __fused_user_args__ -> _fuser_user_args
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._fused_user_args, edge_index, size, kwargs)

        # TODO: use from packaging.version import parse as parse_version as by default 2.4 > 2.14 which is wrong
        # Let's collectively hope there will be PyG 3.0 after 2.9 and not 2.10
        pyg_version = [int(i) for i in torch_geometric.__version__.split(".")]
        col_fn = self.inspector.distribute if pyg_version[1] <= 4 else self.inspector.collect_param_data

        msg_aggr_kwargs = col_fn("message_and_aggregate", coll_dict)
        
        for hook in self._message_and_aggregate_forward_pre_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs))
            if res is not None:
                edge_index, msg_aggr_kwargs = res

        out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
        for hook in self._message_and_aggregate_forward_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs), out)
            if res is not None:
                out = res

        # PyG 2.5+ distribute -> collect_param_data
        update_kwargs = col_fn("update", coll_dict)
        out = self.update(out, **update_kwargs)

        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, size, kwargs), out)
            if res is not None:
                out = res

        return out

    def message(self, input_j, relation, boundary, edge_type):

        relation_j = relation.index_select(self.node_dim, edge_type)

        if self.message_func == "transe":
            message = input_j + relation_j
        elif self.message_func == "distmult":
            message = input_j * relation_j
        elif self.message_func == "rotate":
            x_j_re, x_j_im = input_j.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            message_re = x_j_re * r_j_re - x_j_im * r_j_im
            message_im = x_j_re * r_j_im + x_j_im * r_j_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

        # augment messages with the boundary condition
        message = torch.cat([message, boundary], dim=self.node_dim)  # (num_edges + num_nodes, batch_size, input_dim)

        return message

    def aggregate(self, input, edge_weight, index, dim_size):
        # augment aggregation index with self-loops for the boundary condition
        index = torch.cat([index, torch.arange(dim_size, device=input.device)]) # (num_edges + num_nodes,)
        edge_weight = torch.cat([edge_weight, torch.ones(dim_size, device=input.device)])
        shape = [1] * input.ndim
        shape[self.node_dim] = -1
        edge_weight = edge_weight.view(shape)

        if self.aggregate_func == "pna":
            mean = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            sq_mean = scatter(input ** 2 * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            max = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="max")
            min = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="min")
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            degree_out = degree(index, dim_size).unsqueeze(0).unsqueeze(-1)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            output = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            output = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size,
                             reduce=self.aggregate_func)

        return output

    def message_and_aggregate(self, edge_index, input, boundary, edge_type, edge_weight, edge_attr, index, dim_size):
        
        # fused computation of message and aggregate steps with the custom rspmm cuda kernel
        # speed up computation by several times
        # reduce memory complexity from O(|E|d) to O(|V|d), so we can apply it to larger graphs
        from .rspmm import generalized_rspmm



        batch_size, num_node = input.shape[:2]
        input = input.transpose(0, 1).flatten(1)
        boundary = boundary.transpose(0, 1).flatten(1)
        degree_out = degree(index, dim_size).unsqueeze(-1) + 1

        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        if self.aggregate_func == "sum":
            update = generalized_rspmm(edge_index, edge_type, edge_weight, edge_attr, input, sum="add", mul=mul)
            update = update + boundary
        elif self.aggregate_func == "mean":
            update = generalized_rspmm(edge_index, edge_type, edge_weight, edge_attr, input, sum="add", mul=mul)
            update = (update + boundary) / degree_out
        elif self.aggregate_func == "max":
            update = generalized_rspmm(edge_index, edge_type, edge_weight, edge_attr, input, sum="max", mul=mul)
            update = torch.max(update, boundary)
        elif self.aggregate_func == "pna":
            raise NotImplementedError
            # we use PNA with 4 aggregators (mean / max / min / std)
            # and 3 scalars (identity / log degree / reciprocal of log degree)
            sum = generalized_rspmm(edge_index, edge_type, edge_weight, input, sum="add", mul=mul)
            sq_sum = generalized_rspmm(edge_index, edge_type, edge_weight, input ** 2, sum="add",
                                       mul=mul)
            max = generalized_rspmm(edge_index, edge_type, edge_weight, input, sum="max", mul=mul)
            min = generalized_rspmm(edge_index, edge_type, edge_weight, input, sum="min", mul=mul)
            mean = (sum + boundary) / degree_out
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            max = torch.max(max, boundary)
            min = torch.min(min, boundary) # (node, batch_size * input_dim)
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2) # (node, batch_size * input_dim * 4)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1) # (node, 3)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2) # (node, batch_size * input_dim * 4 * 3)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        update = update.view(num_node, batch_size, -1).transpose(0, 1)
        return update

    def update(self, update, input):
        # node update as a function of old states (input) and this layer output (update)
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
    