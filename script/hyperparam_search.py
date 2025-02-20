import wandb
import datetime
import os
import sys
import copy
import math
import pprint
from itertools import islice
from functools import partial

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data
import optuna

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import tasks, util, test_functions
from ultra.models import Gru_Ultra, Ultra
from ultra.datasets import Yelp18

separator = ">" * 30
line = "-" * 30
k = 20 # ndcg@k
nr_eval_negs = -1 #  == -1 evaluation on all negatives or nr_eval_negs otherwise

def multigraph_collator(batch, train_graphs):
    num_graphs = len(train_graphs)
    probs = torch.tensor([graph.edge_index.shape[1] for graph in train_graphs]).float()
    probs /= probs.sum()
    graph_id = torch.multinomial(probs, 1, replacement=False).item()

    graph = train_graphs[graph_id]
    bs = len(batch)
    edge_mask = torch.randperm(graph.target_edge_index.shape[1])[:bs]

    # Batch combines edges (u, v), edge type, and the edge indices
    batch = torch.cat([graph.target_edge_index[:, edge_mask], graph.target_edge_type[edge_mask].unsqueeze(0)]).t()
    return graph, batch, edge_mask



# here we assume that train_data and valid_data are tuples of datasets
def train_and_validate(cfg, trial, models, train_data, valid_data, filtered_data=None, batch_per_epoch=None, context = 0):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()
    wandb_on = cfg.train["wandb"]
    node_features= cfg.model["node_features"]
    edge_features= cfg.model["edge_features"]
    parallel_models = []
    # Combine parameters from all models
    all_params = []
    param_groups = []

    # this loop is pretty ugly. It does add the parameter groups based on node features and wraps model in DistributedDataParallel
    for i, model in enumerate(models):
        model.to(device)
        
        # Add backbone convolution parameters for the first model
        if i == 0:
            param_groups.append({"params": model.ultra.simple_model.parameters(), "lr": cfg.optimizer["backbone_conv_lr"]})
            if node_features:
                param_groups.append({"params": model.ultra.user_mlp.parameters(), "lr": cfg.optimizer["backbone_mlp_user_lr"]})
                param_groups.append({"params": model.ultra.item_mlp.parameters(), "lr": cfg.optimizer["backbone_mlp_item_lr"]})
            if edge_features:
                param_groups.append({"params": model.ultra.edge_mlp.parameters(), "lr": cfg.optimizer["backbone_mlp_edge_lr"]})
    
        if node_features:
           # Add node-projection MLP parameters for each model
            param_groups.append({"params": model.user_projection.parameters(), "lr": cfg.optimizer["projection_user_lr"]})
            param_groups.append({"params": model.item_projection.parameters(), "lr": cfg.optimizer["projection_item_lr"]})

        if edge_features:
            # Add edge-projection MLP parameters for each model
            param_groups.append({"params": model.edge_projection.parameters(), "lr": cfg.optimizer["projection_edge_lr"]})
        
        # Wrap model in DistributedDataParallel if needed
        if world_size > 1:
            parallel_models.append(nn.parallel.DistributedDataParallel(model, device_ids=[device]))
        else:
            parallel_models.append(model)
    
    # Initialize a single optimizer with unique parameter groups
    optimizer_cls_name = cfg.optimizer.pop("class")
    optimizer_cls = getattr(optim, optimizer_cls_name)
    optimizer = optimizer_cls(param_groups)
    #optimizer = torch.optim.RMSprop(all_params, lr=1.0e-3, alpha=0.99)

    # Prepare graph-to-model mapping for efficient lookup       
    graph_to_model_map = {id(dataset): idx for idx, dataset in enumerate(train_data)}
    
    train_triplets = torch.cat([
        torch.cat([g.target_edge_index, g.target_edge_type.unsqueeze(0)]).t()
        for g in train_data
    ])
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(train_triplets, cfg.train.batch_size, sampler=sampler, collate_fn=partial(multigraph_collator, train_graphs=train_data))

    batch_per_epoch = batch_per_epoch or len(train_loader)

    

    num_params = sum(p.numel() for p in all_params)
    logger.warning(line)
    logger.warning(f"Number of parameters: {num_params}")

    

    step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1

    batch_id = 0
    

    for i in range(0, cfg.train.num_epoch, step):

        for parallel_model in parallel_models:
            parallel_model.train()
            
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            losses = []
            sampler.set_epoch(epoch)
            for batch in islice(train_loader, batch_per_epoch):
                # now at each step we sample a new graph and edges from it
                train_graph, batch, edge_indices = batch
                target_edge_attr = train_graph.target_edge_attr[edge_indices,:]
                
                #print (f"batch_with_attr.shape: {batch_with_attr.shape}")
                #print (f"target_edge.shape: {target_edge.shape}")
                #print (f"target_edge_attr.shape: {target_edge_attr.shape}")
                #test_functions.debug_edge_attr_alignment(train_graph, torch.cat([batch,target_edge_attr], dim = 1))
                #raise ValueError("until here only")
                # based on the train_graph choose the appropriate model and optimizer
                graph_idx = graph_to_model_map[id(train_graph)]
                parallel_model = parallel_models[graph_idx]
                
                batch = tasks.negative_sampling(train_graph, batch, cfg.task.num_negative,
                                                strict=cfg.task.strict_negative)
                pred = parallel_model(train_graph, batch, target_edge_attr)
                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                neg_weight = torch.ones_like(pred)
                if cfg.task.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1)
                else:
                    neg_weight[:, 1:] = 1 / cfg.task.num_negative
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                loss = loss.mean()

                loss.backward()
                            
                optimizer.step()
                optimizer.zero_grad()

               

                if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("binary cross entropy: %g" % loss)


                losses.append(loss.item())
                batch_id += 1

            if util.get_rank() == 0:
                avg_loss = sum(losses) / len(losses)
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning("average binary cross entropy: %g" % avg_loss)

        epoch = min(cfg.train.num_epoch, i + step)
        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {
                "model": models[0].ultra.state_dict(),
                "optimizer": optimizer.state_dict()  
            }
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")

        result = test(cfg, models, valid_data, filtered_data=filtered_data, context = 0)
        #result = 0
        if context == 0:
            # Decide wether we prune
            trial.report(result, step=epoch)
                
        if result > best_result:
            best_result = result
            best_epoch = epoch
            
        if wandb_on and util.get_rank() == 0:
            if context == 0:
                wandb.log({f"{context}/training/epoch": epoch})
                
            wandb.log({f"{context}/training/val_metric": result})
            wandb.log({f"{context}/training/trial": trial.number})

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)

    for model in models:  
        model.ultra.load_state_dict(state["model"])
    util.synchronize()
    return best_result,state
    


    


@torch.no_grad()
def test(cfg, models, test_data, filtered_data=None, context = 0):
    # context is used for determining the calling context of test (train, valid, test)
    
    world_size = util.get_world_size()
    rank = util.get_rank()
    
    # test_data is a tuple of validation/test datasets
    # process sequentially
    # we target the ndcg@k metric
    all_metrics = []
    dataset_nr = 0
    for model, test_graph, filters in zip(models, test_data, filtered_data):
        num_users = test_graph.num_users

        # add edge_indices to the edges so we can provide the edge_attr to the model
        num_edges = test_graph.target_edge_index.size(1)  # Number of edges
        edge_indices = torch.arange(num_edges,dtype=torch.int64, device=test_graph.target_edge_index.device).unsqueeze(1)  # Shape: (num_edges, 1)
    
        test_triplets_with_idx = torch.cat([
            test_graph.target_edge_index.t(),  # Shape: (num_edges, 2)
            test_graph.target_edge_type.unsqueeze(1),   # Shape: (num_edges, 1)
            edge_indices    # Shape: (num_edges, attr_dim)
        ], dim=1)  # Final Shape: (num_edges, 2 + 1 + attr_dim)
    
        sampler = torch_data.DistributedSampler(test_triplets_with_idx, world_size, rank)
        test_loader = torch_data.DataLoader(test_triplets_with_idx, cfg.train.batch_size, sampler=sampler)

        model.eval()
        rankings = []
        num_negatives = []
        ndcgs = []
        for batch_with_idx in test_loader:
            batch = batch_with_idx[:,:3]
            edge_indices = batch_with_idx[:, 3].long()
            target_edge_attr = test_graph.target_edge_attr[edge_indices,:]
            #test_functions.debug_edge_attr_alignment(test_data, torch.cat([batch,target_edge_attr], dim = 1))
            #raise ValueError("until here only")
            if nr_eval_negs == -1:
                t_batch, h_batch = tasks.all_negative(test_graph, batch)
                t_pred = model(test_graph, t_batch, target_edge_attr)
                h_pred = model(test_graph, h_batch, target_edge_attr)
                #t_pred= (bs, num_nodes)
            else:
                #print(f"im here in test and we evaluate vs: {nr_eval_negs} ")
                batch_size = batch.size(0)
                # Create tensors filled with -infinity
                t_pred = torch.full((batch_size, test_graph.num_nodes), float('-inf'), device=batch.device)
                h_pred = torch.full((batch_size, test_graph.num_nodes), float('-inf'), device=batch.device)
                
                # Concatenate batch for negative sampling
                batch_concat = torch.cat((batch, batch), dim=0)
                
                # Perform negative sampling
                batch_sampled = tasks.negative_sampling(filters, batch_concat, nr_eval_negs, strict=True)
                
                # Split the batch into t_batch and h_batch
                t_batch = batch_sampled[:batch_size, :, :]
                h_batch = batch_sampled[batch_size:, :, :]
                
                # Get predictions for the sampled negatives
                t_pred_batch = model(test_graph, t_batch, target_edge_attr)  # Shape: (batch_size, nr_eval_negs + 1)
                h_pred_batch = model(test_graph, h_batch, target_edge_attr)  # Shape: (batch_size, nr_eval_negs + 1)
                
                # Use scatter to populate t_pred and h_pred efficiently
                # Extract the tail indices from t_batch and head indices from h_batch
                t_indices = t_batch[:, :, 1]  # Tail node indices, shape: (batch_size, 101)
                h_indices = h_batch[:, :, 0]  # Head node indices, shape: (batch_size, 101)
                
                 # Initialize relevance tensors directly with zeros
                t_relevance = torch.zeros((batch_size, test_graph.num_nodes), device=batch.device)
                h_relevance = torch.zeros((batch_size, test_graph.num_nodes), device=batch.device)
            
                # Set the relevance directly using advanced indexing
                t_relevance[torch.arange(batch_size, device=batch.device), t_indices[:, 0]] = 1
                h_relevance[torch.arange(batch_size, device=batch.device), h_indices[:, 0]] = 1
                # Scatter predictions into the respective tensors
                t_pred = t_pred.scatter(1, t_indices, t_pred_batch)
                h_pred = h_pred.scatter(1, h_indices, h_pred_batch)
                
                # At this point, t_pred and h_pred are populated with the predictions

            if filtered_data is None:
                t_mask, h_mask = tasks.strict_negative_mask(test_graph, batch)
            else:
                t_mask, h_mask = tasks.strict_negative_mask(filters, batch)
            pos_h_index, pos_t_index, pos_r_index = batch.t()
            
            # compute ndcg@20
            # compute t_rel/ h_rel = (bs, num_nodes) all should have 1 that are in the test set:
            #t_relevance_neg, h_relevance_neg = tasks.strict_negative_mask(test_graph, batch, context = 3)
            #t_relevance,h_relevance = tasks.invert_mask(t_relevance_neg, h_relevance_neg, num_users)
            #test_functions.validate_relevance(t_relevance, h_relevance, test_graph, pos_h_index, pos_t_index)
            
            # mask out all scores of known edges. 
            #t_mask_inv, h_mask_inv = tasks.invert_mask(t_mask, h_mask, num_users)
            # mask out pos_t/h_index 
            #t_mask_pred = t_mask_inv.logical_xor(t_relevance)
            #h_mask_pred = h_mask_inv.logical_xor(h_relevance)
            #test_functions.validate_pred_mask(t_mask_pred, h_mask_pred, test_graph, filters, pos_h_index, pos_t_index)
    
            #t_pred[t_mask_pred] = float('-inf')
            #h_pred[h_mask_pred] = float('-inf')
        
            #compute ndcg scores 
            t_ndcg = tasks.compute_ndcg_at_k(t_pred, t_relevance, k)
            h_ndcg = tasks.compute_ndcg_at_k(h_pred, h_relevance, k)
            ndcgs += [t_ndcg, h_ndcg]

    
            
            t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
            h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
            num_t_negative = t_mask.sum(dim=-1)
            num_h_negative = h_mask.sum(dim=-1)

            rankings += [t_ranking, h_ranking]
            num_negatives += [num_t_negative, num_h_negative]

        ranking = torch.cat(rankings)
        ndcg = torch.cat(ndcgs)
        num_negative = torch.cat(num_negatives)
        all_size = torch.zeros(world_size, dtype=torch.long, device=device)
        all_size[rank] = len(ranking)
        if world_size > 1:
            dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        cum_size = all_size.cumsum(0)
        
        all_ndcg = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_ndcg[cum_size[rank] - all_size[rank]: cum_size[rank]] = ndcg
        
        all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
        
        all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative
        if world_size > 1:
            dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_ndcg, op=dist.ReduceOp.SUM)

        metrics = {}
        if rank == 0:
            for metric in cfg.task.metric:
                if metric == "mr":
                    score = all_ranking.float().mean()
                elif metric == "mrr":
                    score = (1 / all_ranking.float()).mean()

                elif metric == "ndcg@20":
                     score = all_ndcg.float().mean().item()
                     all_metrics.append(score)
                elif metric.startswith("hits@"):
                    values = metric[5:].split("_")
                    threshold = int(values[0])
                    if len(values) > 1:
                        num_sample = int(values[1])
                        # unbiased estimation
                        fp_rate = (all_ranking - 1).float() / all_num_negative
                        score = 0
                        for i in range(threshold):
                            # choose i false positive from num_sample - 1 negatives
                            num_comb = math.factorial(num_sample - 1) / \
                                    math.factorial(i) / math.factorial(num_sample - i - 1)
                            score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                        score = score.mean()
                    else:
                        score = (all_ranking <= threshold).float().mean()
                logger.warning("%s: %g" % (metric, score))
                metrics[metric] = score
            
                
        mrr = (1 / all_ranking.float()).mean()
        
        #all_metrics.append(mrr)
        if rank == 0:
            logger.warning(separator)
 
        dataset_nr += 1
            
    # to me it needs this
    util.synchronize()
    avg_metric = sum(all_metrics) / len(all_metrics)
    return avg_metric




if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    
    task_name = cfg.task["name"]
    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)
    
    train_data, valid_data, test_data = dataset._data[0], dataset._data[1], dataset._data[2]
    
    tf_dataset = Yelp18(root = "/itet-stor/trachsele/net_scratch/tl4rec/model_outputs/data")
    
    tf_train_data_single, tf_valid_data_single, tf_test_data_single = tf_dataset[0], tf_dataset[1], tf_dataset[2]
    tf_train_data_single.x_user = None
    tf_train_data_single.x_item = None
    tf_valid_data_single.x_user = None
    tf_valid_data_single.x_item = None
    tf_test_data_single.x_user = None
    tf_test_data_single.x_item = None
    
    tf_train_data = [tf_train_data_single]
    tf_valid_data =[tf_valid_data_single]
    tf_test_data =[ tf_test_data_single]
    
    # check if we are evaluating on amazon
    dataset_name = cfg.dataset['graphs'][0]  # Example: ['Amazon_Beauty', 'Amazon_Games']
    is_amazon = dataset_name.startswith("Amazon")
    if is_amazon:
        print ("We are using a amazon dataset")
        nr_eval_negs = 100
        k = 10
        
    # make datasets smaller if we dont use node_features
    if not cfg.model.get("node_features", True):
        for graph in train_data:
            graph.x_user = None
            graph.x_item = None
        for graph in valid_data:
            graph.x_user = None
            graph.x_item = None
        for graph in test_data:
            graph.x_user = None
            graph.x_item = None
    
    if "fast_test" in cfg.train:
        num_val_edges = cfg.train.fast_test
        if util.get_rank() == 0:
            logger.warning(f"Fast evaluation on {num_val_edges} samples in validation")
        short_valid = [copy.deepcopy(vd) for vd in valid_data]
        for graph in short_valid:
            mask = torch.randperm(graph.target_edge_index.shape[1])[:num_val_edges]
            graph.target_edge_index = graph.target_edge_index[:, mask]
            graph.target_edge_type = graph.target_edge_type[mask]
        short_valid = [sv.to(device) for sv in short_valid]
        
     # ----- Apply "fast_test" truncation to tfvalid set -----
    if "fast_test" in cfg.train:
        num_val_edges = cfg.train.fast_test
        tf_short_valid = [copy.deepcopy(vd) for vd in tf_valid_data]
        for graph in tf_short_valid:
            mask = torch.randperm(graph.target_edge_index.shape[1])[:num_val_edges]
            graph.target_edge_index = graph.target_edge_index[:, mask]
            graph.target_edge_type = graph.target_edge_type[mask]
        tf_short_valid = [sv.to(device) for sv in tf_short_valid]


    train_data = [td.to(device) for td in train_data]
    valid_data = [vd.to(device) for vd in valid_data]
    test_data = [tst.to(device) for tst in test_data]

    tf_train_data = [td.to(device) for td in tf_train_data]
    tf_valid_data = [vd.to(device) for vd in tf_valid_data]
    tf_test_data = [tst.to(device) for tst in tf_test_data]
     # for transductive setting, use the whole graph for filtered ranking
    filtered_data = [
        Data(
            edge_index=torch.cat([trg.target_edge_index, valg.target_edge_index, testg.target_edge_index], dim=1), 
            edge_type=torch.cat([trg.target_edge_type, valg.target_edge_type, testg.target_edge_type,]),
            num_users = trg.num_users,
            num_items = trg.num_items,
            num_nodes=trg.num_nodes).to(device)
            
        for trg, valg, testg in zip(train_data, valid_data, test_data)
    ]
    
    tf_filtered_data = [
        Data(
            edge_index=torch.cat([trg.target_edge_index, valg.target_edge_index, testg.target_edge_index], dim=1), 
            edge_type=torch.cat([trg.target_edge_type, valg.target_edge_type, testg.target_edge_type,]),
            num_users = trg.num_users,
            num_items = trg.num_items,
            num_nodes=trg.num_nodes).to(device)
            
        for trg, valg, testg in zip(tf_train_data, tf_valid_data, tf_test_data)
    ]

    #discrad test data
    test_data = None
    tf_test_data= None
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"hyperparam_search-{current_time}"
    wandb_on = cfg.train["wandb"]
    if wandb_on:
        wandb.init(
            entity = "pitri-eth-z-rich", project="tl4rec", name=run_name, config=cfg)

    def objective(trial):
        try:
            # --- Sample Learning Rate Hyperparameters ---
            proj_edge_lr = trial.suggest_float("projection_edge_lr", 1e-5, 1e-3, log = True)
            bb_conv_lr   = trial.suggest_float("backbone_conv_lr", 1e-5, 1e-3, log = True)
            bb_mlp_edge_lr = trial.suggest_float("backbone_mlp_edge_lr", 1e-5, 1e-3, log = True)

            simple_model_dim = trial.suggest_categorical("simple_model_dim", [32,64])
            simple_model_hidden_dims = []
            simple_model_num_hidden = trial.suggest_int("simple_model_num_hidden", 2, 8)
            for i in range(simple_model_num_hidden):
                simple_model_hidden_dims.append(simple_model_dim)
                
            # --- Sample Edge Projection Hyperparameters ---
            edge_proj_use_dropout    = trial.suggest_categorical("edge_projection_use_dropout", [False, True])
            edge_proj_dropout_rate   = trial.suggest_float("edge_projection_dropout_rate", 0.0, 0.5)
            edge_proj_use_layer_norm = trial.suggest_categorical("edge_projection_use_layer_norm", [False, True])
            
            num_edge_proj_layers = trial.suggest_int("num_edge_proj_layers", 1, 5)
            edge_emb_dim = trial.suggest_categorical("edge_emb_dim", [2, 4, 8, 16, 32])
            edge_projection_hidden_dims  = []
            for i in range(num_edge_proj_layers):
                edge_projection_hidden_dims.append(edge_emb_dim)
                
        
            # --- Sample Embedding Edge Hyperparameters ---
            embedding_edge_use_dropout    = trial.suggest_categorical("embedding_edge_use_dropout", [False, True])
            embedding_edge_dropout_rate   = trial.suggest_float("embedding_edge_dropout_rate", 0.0, 0.5)
            embedding_edge_use_layer_norm = trial.suggest_categorical("embedding_edge_use_layer_norm", [False, True])
        
            num_edge_emb_layers = trial.suggest_int("num_edge_emb_layers", 1, 6)
            embedding_edge_hidden_dims = []
            for i in range(num_edge_emb_layers):
                embedding_edge_hidden_dims.append(edge_emb_dim)
        
            
            
        
            # --- Sample Task Hyperparameters ---
            # Ensure bs * num_negative <= 128. Assuming cfg.train["batch_size"] is already set.
            max_num_negative = 512 // cfg.train["batch_size"]
            num_negative = trial.suggest_int("num_negative", 1, max_num_negative)
            adversarial_temperature = trial.suggest_categorical("adversarial_temperature", [0.0, 0.5, 1.0])
        
            # --- Update the configuration ---
            cfg_trial = copy.deepcopy(cfg)
            
            
            # Optimizer parameters:
            cfg_trial.optimizer["projection_edge_lr"] = proj_edge_lr
            cfg_trial.optimizer["backbone_conv_lr"] = bb_conv_lr
            cfg_trial.optimizer["backbone_mlp_edge_lr"] = bb_mlp_edge_lr
        
            # Update edge_projection parameters:
            cfg_trial.model["edge_projection"]["use_dropout"]    = edge_proj_use_dropout
            cfg_trial.model["edge_projection"]["dropout_rate"]   = edge_proj_dropout_rate
            cfg_trial.model["edge_projection"]["use_layer_norm"] = edge_proj_use_layer_norm
            cfg_trial.model["edge_projection"]["hidden_dims"] = edge_projection_hidden_dims
        
            # Update embedding_edge parameters (assuming they reside under backbone_model):
            cfg_trial.model["backbone_model"]["embedding_edge"]["use_dropout"]    = embedding_edge_use_dropout
            cfg_trial.model["backbone_model"]["embedding_edge"]["dropout_rate"]   = embedding_edge_dropout_rate
            cfg_trial.model["backbone_model"]["embedding_edge"]["use_layer_norm"] = embedding_edge_use_layer_norm
            cfg_trial.model["backbone_model"]["embedding_edge"]["hidden_dims"]    = embedding_edge_hidden_dims
        
            # Update simple_model parameters (assuming they reside under backbone_model):
            cfg_trial.model["backbone_model"]["simple_model"]["input_dim"] = simple_model_dim
            cfg_trial.model["backbone_model"]["simple_model"]["hidden_dims"] = simple_model_hidden_dims
        
            # Update task parameters:
            cfg_trial.task["num_negative"] = num_negative
            cfg_trial.task["adversarial_temperature"] = adversarial_temperature
            tf_cfg = copy.deepcopy(cfg_trial)
        
            # --- Build the models ---
            # Here we assume that train_data, valid_data, test_data, short_valid, and filtered_data 
            # have already been created globally (and moved to device).
            ultra_ref = Ultra(cfg_trial.model, False)
            models = []
            def weights_init(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        
            for td in train_data:
                # Build a copy of the model configuration for each dataset.
                model_cfg = copy.deepcopy(cfg_trial.model)
                # For this version, we assume node features are not used.
                model_cfg["edge_projection"]["input_dim"] = td.edge_attr.size(1)
                model = Gru_Ultra(model_cfg, ultra_ref, wandb_on)
                model.apply(weights_init)
                models.append(model)
        
            # --- Train and Validate ---
            # Use the pre-built short_valid and filtered_data for consistency.
            _, state = train_and_validate(
                cfg_trial,trial, models, train_data,
                valid_data if "fast_test" not in cfg_trial.train else short_valid,
                filtered_data=filtered_data,
                batch_per_epoch=cfg_trial.train["batch_per_epoch"], context = 0
            )
            tf_cfg.train["num_epoch"] =  1
            tf_cfg.model.edge_projection["input_dim"] = tf_train_data[0].edge_attr.size(1)
            tf_model = Gru_Ultra(
                cfg = tf_cfg.model)
            tf_model = tf_model.to(device)
            tf_model.apply(weights_init)
            tf_model.ultra.load_state_dict(state["model"])
            print ("eval on tf")
            val_metric,_ = train_and_validate(
                tf_cfg,trial, [tf_model], tf_train_data,
                tf_short_valid,
                filtered_data= tf_filtered_data,
                batch_per_epoch= 2000, context = 1
            )
                    
            return val_metric
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Trial {trial.number} encountered OOM. Pruning trial...")
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()  # Marks it as pruned
            else:
                raise e  
        
    storage_url = "sqlite:////itet-stor/trachsele/net_scratch/tl4rec/optuna_hyperparam_tuning_v2.db"

    if False:
        study = optuna.create_study(
            direction="maximize", 
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=1)  
        )

    
    study = optuna.create_study(
        direction="maximize",
        storage=storage_url,  # Store trials in an SQLite file
        study_name="my_hyperparam_study_v2",
        load_if_exists=True,  # Resume trials if the file exists
    )
    print("Existing studies:", optuna.get_all_study_names(storage=storage_url))
        
    study.optimize(objective, n_trials=100)  # Adjust n_trials based on your available resources
    best_params = study.best_trial.params
    util.save_cfg_of_best_params(best_params, cfg)
    logger.warning("hyperparam search done and cfg saved")

    

