import torch
from torch import nn
import torch.nn.functional as F
import wandb
from . import tasks, layers
from ultra.base_nbfnet import BaseNBFNet
from torch_geometric.nn.models import LightGCN
from ultra import util


        
class Gru_Ultra(nn.Module):
    def __init__(self, cfg, ultra_ref = None, log = False):
        # kept that because super Ultra sounds cool
        super(Gru_Ultra, self).__init__()
        
        # Dataset-specific projection layerscfg.backbone_model
        self.edge_features = cfg["edge_features"]
        if self.edge_features:
            self.edge_projection =  MLP(cfg.edge_projection)
        else:
            raise NotImplementedError
        # Shared backbone
        if ultra_ref is not None:
            self.ultra = ultra_ref
        else:
            self.ultra = Ultra(cfg, log)
        self.log = log
        
    def forward(self, data, batch, target_edge_attr):
        num_users = data.num_users
        num_items = data.num_items
            
        if self.edge_features:
            conv_edge_projection = self.edge_projection(data.edge_attr)
            target_edge_projections= self.edge_projection(target_edge_attr)
        else:
            raise NotImplementedError
        score = self.ultra(data, batch, conv_edge_projection, target_edge_projections)
        # what does score look like? score (batch_size, 1 + num negatives)
        
        return score
    
class Ultra(nn.Module):

    def __init__(self, cfg, log = False):
        # kept that because super Ultra sounds cool
        super(Ultra, self).__init__()
        self.edge_features = cfg["edge_features"]
        if self.edge_features:
            self.edge_mlp = MLP(cfg.backbone_model.embedding_edge)
        else:   
            raise NotImplementedError
            
        self.log = log
        edge_emb_dim = cfg.backbone_model.embedding_edge["hidden_dims"][0]
        # adding a bit more flexibility to initializing proper rel/ent classes from the configs
        # globals() contains all global class variable 
        # rel_model_cfg.pop('class') pops the class name from the cfg thus combined it returns the class
        # **rel_model_cfg contains dict of cfg file
        simple_model_cfg= cfg.backbone_model.simple_model
        self.simple_model = globals()[simple_model_cfg.pop('class')](edge_emb_dim, **simple_model_cfg)

        
    def forward(self, data, batch, conv_edge_projection, target_edge_projections):
        # calculate embeddings
        num_users = data.num_users
        num_items = data.num_items

        edge_attr= data.edge_attr
        if self.edge_features:
            conv_edge_projection_mean = conv_edge_projection.mean().item()
            conv_edge_projection_std = conv_edge_projection.std().item()
            conv_edge_projection_var = conv_edge_projection.var().item()
    
            # Log these metrics under the key "debug/edge_attr_metric"
            #wandb.log({
             #   "debug/conv_edge_projection_mean": conv_edge_projection_mean,
              #  "debug/conv_edge_projection_std": conv_edge_projection_std,
               # "debug/conv_edge_projection_var": conv_edge_projection_var
            #})
            
            conv_edge_embedding = self.edge_mlp(conv_edge_projection) 
            # Compute statistics for conv_edge_embedding
            conv_edge_embedding_mean = conv_edge_embedding.mean().item()
            conv_edge_embedding_std = conv_edge_embedding.std().item()
            conv_edge_embedding_var = conv_edge_embedding.var().item()
    
            # Log these metrics under the key "debug/edge_attr_metric"
            #wandb.log({
             #   "debug/conv_edge_embedding_mean": conv_edge_embedding_mean,
              #  "debug/conv_edge_embedding_std": conv_edge_embedding_std,
               # "debug/conv_edge_embedding_var": conv_edge_embedding_var
            #})
            target_edge_embedding = self.edge_mlp(target_edge_projections) 
        
        score = self.simple_model(data, batch, conv_edge_embedding, target_edge_embedding)
        # score (batch_size, 1 + num negatives)
        
        return score



class SimpleNBFNet(BaseNBFNet):

    def __init__(self, edge_emb_dim, input_dim, hidden_dims, num_relation=2, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        # maybe remove activation function
        self.activation = None
        self.concat_hidden = True
        self.project_conv_emb = kwargs.get('project_conv_emb', False)
       # print(f"self.project_conv_emb {self.project_conv_emb}")
        
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], edge_emb_dim, self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, project_conv_emb = self.project_conv_emb)
                )

        feature_dim = (sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]) + input_dim
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    
    def bellmanford(self, data, batch, conv_edge_embedding, target_edge_embedding, h_index, separate_grad=False):
        
        # initialize queries with target_edge_embedding repeated to match the boundary_dim
        batch_size = len(h_index)
        input_dim = self.dims[0]
        edge_embedding_dim = target_edge_embedding.size(1)
        # Repeat and concatenate target_edge_embedding to match input_dim
        repeat_factor = input_dim // edge_embedding_dim
        query = target_edge_embedding.repeat(1, repeat_factor)  # (batch_size, input_dim)
        #query = torch.ones(batch_size, input_dim, device=h_index.device)  
        #print(f"edge_embedding_dim {edge_embedding_dim}")
        #print(f"query.shape {query.shape}")
        #raise ValueError("until here only")

    


        # Initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, input_dim, device=h_index.device)  
        # Set the boundary condition for query head nodes
        index = h_index.unsqueeze(-1).expand(-1, input_dim)
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        
        
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, conv_edge_embedding,  size, edge_weight)
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

    def forward(self, data, batch, conv_edge_embedding, target_edge_embedding):
        h_index, t_index, r_index = batch.unbind(-1)
        batch_size = batch.shape[0]





        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            #print (f"before removal conv_edge_embedding.shape: {conv_edge_embedding.shape}")
            data, conv_edge_embedding= self.remove_easy_edges(data, conv_edge_embedding, h_index, t_index, r_index)
            #print (f"after removal conv_edge_embedding.shape: {conv_edge_embedding.shape}")
            #raise NotImplementedError
        

        
        
        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index   = self.negative_sample_to_tail(h_index, t_index, r_index,
                                                                                               num_direct_rel=data.num_relations // 2)
        
        # I really dont understand how we pass this check if we check that every batch row has identical head node ?!!?!
        # Answer: Every batch row has either corrupted head node or tail node not both in the negative samples
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations
        output = self.bellmanford(data, batch, conv_edge_embedding, target_edge_embedding, h_index[:, 0])  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])  #unsequeeze adds dimensions on top leve x^2 to x^3 expand changes how many rows
        # extract representations of tail entities from the updated node states
        feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)
    
# Define a standard MLP class
class MLP(nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()

        hidden_dims = cfg["hidden_dims"]
        layers = []
        in_dim = hidden_dims[0]
        if 'input_dim' in cfg.keys():
             in_dim = cfg['input_dim']

        use_dropout = cfg["use_dropout"]
        dropout_rate = cfg["dropout_rate"]
        use_layer_norm = cfg["use_layer_norm"]

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))  # Add LayerNorm
            layers.append(nn.ReLU())  # Add ReLU activation
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))  # Add Dropout if enabled
            in_dim = h_dim  # Update input dimension for the next layer

        # Store layers in a ModuleList for registration and forward pass
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class My_LightGCN(nn.Module):
    def __init__(self, num_nodes):
        super(My_LightGCN, self).__init__()
        self.gcn = LightGCN(num_nodes, 64, 3)
    def forward(self, data, batch):
        
        """
        Args:
            data: A PyG data object containing edge_index, x_user, x_item, etc.
            batch (Tensor): A tensor of shape (batch_size, 1 + num_neg_samples, 3),
                            where batch[:, :, 0] are user indices,
                            batch[:, :, 1] are item indices.
        
        Returns:
            Tensor: Logit scores of shape (batch_size, 1 + num_neg_samples).
        """
        # Get LightGCN embeddings for the nodes in the graph
        lgn_embeddings = self.gcn.get_embedding(data.edge_index)

        # Extract user and item embeddings from LightGCN using the batch indices
        user_indices = batch[:, :, 0].view(-1)  # Flatten user indices
        item_indices = batch[:, :, 1].view(-1)  # Flatten item indices

        # Get the embeddings for the specified users and items
        user_emb_from_lgn = lgn_embeddings[user_indices]
        item_emb_from_lgn = lgn_embeddings[item_indices]

        # Compute scores as the dot product between user and item embeddings
        logits = (user_emb_from_lgn * item_emb_from_lgn).sum(dim=-1)

        # Reshape logits to (batch_size, 1 + num_neg_samples)
        logits = logits.view(batch.size(0), -1)

        #return logits.sigmoid()  # Return scores between 0 and 1
        return logits # no sigmoids
    


