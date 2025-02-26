import torch
from torch import nn
import torch.nn.functional as F

from . import tasks, layers
from ultra.base_nbfnet import BaseNBFNet
from torch_geometric.nn.models import LightGCN

class Ultra1(nn.Module):

    def __init__(self, rel_model_cfg, entity_model_cfg, embedding_user_cfg, embedding_item_cfg):
        # kept that because super Ultra sounds cool
        super(Ultra, self).__init__()
        
        # MLP's for obtaining item/user emb.
        self.item_mlp = MLP(**embedding_item_cfg)
        self.user_mlp = MLP(**embedding_user_cfg)
        # adding a bit more flexibility to initializing proper rel/ent classes from the configs
        # globals() contains all global class variable 
        # rel_model_cfg.pop('class') pops the class name from the cfg thus combined it returns the class
        # **rel_model_cfg contains dict of cfg file
        self.relation_model = globals()[rel_model_cfg.pop('class')](**rel_model_cfg)
        self.entity_model = globals()[entity_model_cfg.pop('class')](**entity_model_cfg)

        
    def forward(self, data, batch):
        # calculate embeddings
        user_embedding = self.user_mlp(data.x_user)  # shape: (num_users, 16)
        item_embedding = self.item_mlp(data.x_item)
        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs
        query_rels = batch[:, 0, 2]
        relation_representations = self.relation_model(data.relation_graph, query=query_rels)
        score = self.entity_model(data, relation_representations, batch, user_embedding, item_embedding)
        # what does score look like? score (batch_size, 1 + num negatives)
        
        return score


# NBFNet to work on the graph of relations with 4 fundamental interactions
# Doesn't have the final projection MLP from hidden dim -> 1, returns all node representations 
# of shape [bs, num_rel, hidden]
class RelNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=4, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False)
                )

        if self.concat_hidden:
            feature_dim = sum(hidden_dims) + input_dim
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, input_dim)
            )

    
    def bellmanford(self, data, h_index, separate_grad=False):
        batch_size = len(h_index)

        # initialize initial nodes (relations of interest in the batcj) with all ones
        query = torch.ones(h_index.shape[0], self.dims[0], device=h_index.device, dtype=torch.float)
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        #boundary = torch.zeros(data.num_nodes, *query.shape, device=h_index.device)
        # Indicator function: by the scatter operation we put ones as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, rel_graph, query):

        # message passing and updated node representations (that are in fact relations)
        output = self.bellmanford(rel_graph, h_index=query)["node_feature"]  # (batch_size, num_nodes, hidden_dim）
        
        return output
    

class EntityNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, relation_input_dim, num_relation=1, **kwargs):

        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)
        self.concat_hidden = True
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True, relation_input_dim = relation_input_dim)
            )

        feature_dim = (sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]) + input_dim
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    
    def bellmanford(self, data, h_index, r_index, user_embedding, item_embedding, h_embeddings , separate_grad=False):
        batch_size = len(r_index)

        # initialize queries (relation types of the given triples)
        # Must adjust size of queries since we add the 16 bit embeddings 
        # in the hidden layers we project (expected input dim) the queries for further calculations
        input_dim = self.dims[0]
        query_temp = self.query[torch.arange(batch_size, device=r_index.device), r_index]  # (8, 16)
        query_size = query_temp.size(1)
        zeros = torch.zeros(batch_size, input_dim - query_size, dtype=query_temp.dtype, device=r_index.device)  # Create zeros on the same device as r_index
        

        
        # Compress h_embeddings to (batch_size, input_dim - query_size) by taking the first column since all head_embeddings are consistent in each batch
        compressed_h_embeddings = h_embeddings[:, 0, :]
        
        query = torch.cat([query_temp, compressed_h_embeddings], dim=1)

        query_zero = torch.cat([query_temp, zeros], dim=1)  # Concatenate along the second dimension
        
        index = h_index.unsqueeze(-1).expand_as(query)


        # ORIGINAL ULTRA CODE 
        # initial (boundary) condition - initialize all node states as zeros
        # boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        # boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        # Initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, input_dim, device=h_index.device)  # size is 32 (16 + 16)
        
        # Append node embeddings for all nodes (user and item)
        embedding_index= query_size # TODO
        all_embeddings = torch.cat([user_embedding, item_embedding], dim=0)  # Combine user and item embeddings (dim: num_nodes x 16)
        boundary[:, :, embedding_index:] = all_embeddings  # Fill the last 16 dimensions with node embeddings
        
        boundary.scatter_add_(1, index.unsqueeze(1), query_zero.unsqueeze(1))  # Add relation embeddings to the first 16 entries
        
        
        
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:

            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, relation_representations, batch, user_embedding, item_embedding):
        h_index, t_index, r_index = batch.unbind(-1)
        batch_size = batch.shape[0]
        num_users = user_embedding.shape[0]
        embedding_dim = user_embedding.shape[-1]

        # initial query representations are those from the relation graph
        self.query = relation_representations # (bs, num_relations ,input_dim)


        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        
        # gather the head/tail_node embeddings such that h_embeddings.shape = (bs, 1 + num_neg, emb_dim)
        # note orignally we only have user item edges but due to corruption we have may have user-user, item-item edges
        # we dont care about those since we are only interested in the incorrupted embeddings
        # some corrupted h_index may be > num_user - 1
        index_temp = h_index.clamp(max= num_users - 1).unsqueeze(-1).expand(-1,-1, embedding_dim)
        h_embeddings = user_embedding.unsqueeze(0).expand(batch_size,-1,-1).gather(1,index_temp)
        
        # some corrupted nodes may be smaller than num_users
        index_temp = (t_index - num_users).clamp(min=0).unsqueeze(-1).expand(-1,-1, embedding_dim)
        t_embeddings = item_embedding.unsqueeze(0).expand(batch_size,-1,-1).gather(1,index_temp)

        
        
        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index, h_embeddings   = self.negative_sample_to_tail(h_index, t_index, r_index,
                                                                                               num_direct_rel=data.num_relations // 2,
                                                                                               h_embeddings=h_embeddings,
                                                                                               t_embeddings=t_embeddings)
        
        # I really dont understand how we pass this check if we check that every batch row has identical head node ?!!?!
        # Answer: Every batch row has either corrupted head node or tail node not both in the negative samples
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0], user_embedding, item_embedding, h_embeddings)  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])  #unsequeeze adds dimensions on top leve x^2 to x^3 expand changes how many rows
        # extract representations of tail entities from the updated node states
        feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)