import os
import sys
import math
import pprint
from itertools import islice
import wandb
import datetime

import torch
import torch_geometric as pyg
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import tasks, util, test_functions
from ultra.models import Gru_Ultra,My_LightGCN


separator = ">" * 30
line = "-" * 30
k = 20 # ndcg@k
fine_tuning = False # wether we are fine-tuning

def train_and_validate(cfg, model, train_data, valid_data, device, logger, filtered_data=None, batch_per_epoch=None):
    if cfg.train.num_epoch == 0:
        return
    world_size = util.get_world_size()
    rank = util.get_rank()
    wandb_on = cfg.train["wandb"]

    num_edges = train_data.target_edge_index.size(1)  # Number of edges
    edge_indices = torch.arange(num_edges,dtype=torch.int64, device=train_data.target_edge_index.device).unsqueeze(1)  # Shape: (num_edges, 1)

    # Concatenate along the second dimension
    target_triplets_with_idx = torch.cat([
        train_data.target_edge_index.t(),  # Shape: (num_edges, 2)
        train_data.target_edge_type.unsqueeze(1),   # Shape: (num_edges, 1)
        edge_indices    # Shape: (num_edges, attr_dim)
    ], dim=1)  # Final Shape: (num_edges, 2 + 1 + attr_dim)

    
    sampler = torch_data.DistributedSampler(target_triplets_with_idx, world_size, rank)
    train_loader = torch_data.DataLoader(target_triplets_with_idx, cfg.train.batch_size, sampler=sampler)

    batch_per_epoch = batch_per_epoch or len(train_loader)
    edge_features = cfg.model.get("edge_features", False)
    
    param_groups = []
    param_groups.append({"params": model.ultra.simple_model.parameters(), "lr": cfg.optimizer["backbone_conv_lr"]})
    # If edge features are used, add the backbone MLP for edges
    if edge_features:
        param_groups.append({
            "params": model.ultra.edge_mlp.parameters(),
            "lr": cfg.optimizer["backbone_mlp_edge_lr"]
        })
        param_groups.append({
            "params": model.edge_projection.parameters(),
            "lr": cfg.optimizer["projection_edge_lr"]
        })

        

    # Create the optimizer with the parameter groups.
    # Pop the optimizer class name from the configuration and initialize it.
    optimizer_cls_name = cfg.optimizer.pop("class")
    optimizer_cls = getattr(optim, optimizer_cls_name)
    optimizer = optimizer_cls(param_groups)
    #scheduler = StepLR(optimizer, step_size= 4, gamma=0.5) 
    #scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=False)
    num_params = sum(p.numel() for p in model.parameters())
    logger.warning(line)
    logger.warning(f"Number of parameters: {num_params}")
    if wandb_on: 
        wandb.config.num_params = num_params

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model
    num_evals = cfg.train["num_evals"]
    step = math.ceil(cfg.train.num_epoch / num_evals)
    best_result = float("-inf")
    best_epoch = -1
    batch_id = 0
    num_freezes = cfg.train.fine_tuning["num_epoch_proj"]
    
    if fine_tuning:
        util.synchronize()
        print ("freeze my backbone")
        util.freeze_backbone(model)
       
    
    for i in range(0, cfg.train.num_epoch, step):
        parallel_model.train()
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            # check if we should fine tune_further
            if fine_tuning and epoch == num_freezes:
                util.synchronize()
                print ("unfreeze my backbone")
                util.unfreeze_backbone(model)
            

            
            
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            losses = []
            sampler.set_epoch(epoch)
            for batch_with_idx in islice(train_loader, batch_per_epoch):
                target_edge = batch_with_idx[:,:3]
                edge_indices = batch_with_idx[:, 3].long()
                target_edge_attr = train_data.target_edge_attr[edge_indices,:]
                #print (f"batch_with_attr.shape: {batch_with_attr.shape}")
                #print (f"target_edge.shape: {target_edge.shape}")
                #print (f"target_edge_attr.shape: {target_edge_attr.shape}")
                #test_functions.debug_edge_attr_alignment(train_data, torch.cat([target_edge,target_edge_attr], dim = 1))
                #raise ValueError("until here only")
                batch = tasks.negative_sampling(train_data, target_edge, cfg.task.num_negative,
                                                strict=cfg.task.strict_negative)
                pred = parallel_model(train_data, batch, target_edge_attr)
                target = torch.zeros_like(pred)
                target[:, 0] = 1
                # loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                neg_weight = torch.ones_like(pred)
                if cfg.task.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1)
                else:
                    neg_weight[:, 1:] = 1 / cfg.task.num_negative

                
                
                    
                if cfg.train["loss"] == "bpr":
                    pos_scores= pred[:, 0]
                    neg_scores= (pred[:, 1:] * neg_weight[:, 1:]).sum(dim=-1)
                    loss = - (F.logsigmoid(pos_scores - neg_scores).mean())
                else:
                    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                    loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                    loss = loss.mean()
                    
                    
                

                loss.backward()
                # --- Log gradient norms for edge-related parameters ---
                if cfg.train["wandb"]:
                    edge_mlp_grad_norm_sum = 0.0
                    edge_mlp_count = 0
                    edge_proj_grad_norm_sum = 0.0
                    edge_proj_count = 0
                    
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if "edge_mlp" in name:
                                edge_mlp_grad_norm_sum += param.grad.norm().item()
                                edge_mlp_count += 1
                            elif "edge_projection" in name:
                                edge_proj_grad_norm_sum += param.grad.norm().item()
                                edge_proj_count += 1
                    
                    if edge_mlp_count > 0:
                        wandb.log({"debug/grad_norm/edge_mlp": edge_mlp_grad_norm_sum / edge_mlp_count})
                    if edge_proj_count > 0:
                        wandb.log({"debug/grad_norm/edge_projection": edge_proj_grad_norm_sum / edge_proj_count})
                # ------------------------------------------------------
                # Apply gradient clipping
                if cfg.train["gradient_clip"]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    
                    if wandb_on:
                        wandb.log({"training/loss": loss})
                    if cfg.train["loss"] == "bpr":
                        logger.warning(f"Mean positive scores: {pos_scores.mean().item()}")
                        logger.warning(f"Mean negative scores: {neg_scores.mean().item()}")
                        logger.warning(f"BPR loss: {loss.item()}")
                    else: 
                        logger.warning("binary cross entropy: %g" % loss)
                    if wandb_on:
                        if cfg.train["loss"] == "bpr":
                            wandb.log({"training/pos_scores": pos_scores.mean().item()})
                            wandb.log({"training/neg_scores": neg_scores.mean().item()})
                            
                        wandb.log({"training/loss": loss})
                            
                losses.append(loss.item())
                batch_id += 1

            if util.get_rank() == 0:
                avg_loss = sum(losses) / len(losses)
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning("average loss: %g" % avg_loss)
                
            

        epoch = min(cfg.train.num_epoch, i + step)
        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
            

        result_dict = {}
        result_dict["ndcg@20"] = 1
        
        #result_dict = test(cfg, model, valid_data, filtered_data=filtered_data, device=device, logger=logger, return_metrics = True, nr_eval_negs = 100)

        # Log each metric with the hierarchical key format "training/performance/{metric}"
        if wandb_on:
            for metric, score in result_dict.items():
                wandb.log({f"training/performance/{metric}": score})

        target_metric = cfg.train["target_metric"]
        #result = result_dict[target_metric]
        result = 1
        #scheduler.step()
        #scheduler.step(result) 
        if result > best_result:
            best_result = result
            best_epoch = epoch

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()
    
    if rank == 0:
        # Extract the last 6 letters of each dataset name
        dataset_names = dataset_name = cfg.dataset["class"]
    
        # Construct the checkpoint filename
        checkpoint_dir = "/itet-stor/trachsele/net_scratch/tl4rec/ckpts/dump"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_name = f"{dataset_names}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        logger.warning(f"Save final_ckpt to {checkpoint_path}")
        torch.save(state, checkpoint_path)
        
        

@torch.no_grad()
def test_per_user(cfg, model, test_data, device, logger, filtered_data=None, return_metrics=False, valid_data=None, nr_eval_negs = 1000):
    """
    Per-user evaluation for candidate-based ranking.
    
    For each unique test user (taken only once regardless of how many test edges they have),
    this method:
      1. Creates a DataLoader over user indices (using DistributedSampler so each user is evaluated once).
      2. For each batch of users, builds a candidate set (cand_size=1000) via tasks.build_candidate_set.
      3. Uses a generic target_edge_attr (all ones) to get predictions from the model.
      4. Scatters the candidate scores into a full prediction vector over all items.
      5. Computes a relevance mask (via tasks.strict_negative_mask and tasks.invert_mask) and then NDGC and ranking.
      6. Aggregates metrics across batches and uses simple all_reduce to get the same results on all ranks.
      7. Returns the final MRR (or full metrics) unconditionally.
      
    Note:
      - The generic target_edge_attr is created as all ones with shape (B, edge_dim), where
        edge_dim = cfg.model.edge_projection["hidden_dims"][0].
      - We assume tasks.build_candidate_set can handle a batch of user indices.
    """
    world_size = util.get_world_size()
    rank = util.get_rank()
    print (f"world_size {world_size}")
    # 1. Obtain unique test users from test_data (assuming test_data.target_edge_index[0] holds user IDs)
    user_ids = torch.unique(test_data.target_edge_index[0], sorted=True)
    
    # 2. Create a DataLoader over user indices.
    # Using a TensorDataset with the user_ids is sufficient.
    dataset = torch_data.TensorDataset(user_ids)
    if world_size > 1:
        sampler = torch_data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        sampler = None
    batch_size = cfg.train.test_batch_size
    user_loader = torch_data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=0)
    
    # Lists to collect per-user metric tensors.
    ndcg_list = []
    rank_list = []
    num_neg_list = []
    
    #  Determine edge_dim from configuration.
    #edge_dim = cfg.model.edge_projection["hidden_dims"][0]
    edge_dim = test_data.edge_attr.size(1)
    num_users = test_data.num_users
    model.eval()
    for batch in user_loader:
         #Original batch: tuple with one tensor of user IDs of shape (B,)
        user_ids = batch[0]  # shape: (B,)
        B = user_ids.size(0)
        # Expand user_ids to shape (B,1)
        user_ids = user_ids.unsqueeze(1)
        
        # Create a tensor of zeros for the last two columns (tail and relation type)
        generic_zeros = torch.zeros(B, 2, device=device, dtype=user_ids.dtype)
        
        # Concatenate along the second dimension to form a (B, 3) tensor.
        # Column 0: user id, Columns 1-2: generic zeros.
        user_batch = torch.cat([user_ids, generic_zeros], dim=1)
                
        # 3. Create a generic target_edge_attr: all ones of shape (B, edge_dim).
        generic_target = torch.ones((B, edge_dim), device=device)
        if nr_eval_negs == -1:
            t_batch, _ = tasks.all_negative(test_data, user_batch)
            t_pred = model(test_data, t_batch, generic_target)
            #t_pred= (bs, num_nodes)
                # compute ndcg:
        
            # compute t_rel/ h_rel = (bs, num_nodes) all should have 1 that are in the test set:
            t_relevance_neg, h_relevance_neg = tasks.strict_negative_mask(test_data, user_batch, context = 3)
            t_relevance,_ = tasks.invert_mask(t_relevance_neg, h_relevance_neg, num_users)
            #test_functions.validate_relevance(t_relevance, h_relevance, test_data, pos_h_index, pos_t_index)
    
            # mask out all scores of known edges. 
            if filtered_data is None:
                t_mask, h_mask = tasks.strict_negative_mask(test_data, user_batch)
            else:
                t_mask, h_mask = tasks.strict_negative_mask(filtered_data, user_batch)
                
            t_mask_inv, _ = tasks.invert_mask(t_mask, h_mask, num_users)
            # mask out pos_t/h_index 
            t_mask_pred = t_mask_inv.logical_xor(t_relevance)
            #test_functions.validate_pred_mask(t_mask_pred, h_mask_pred, test_data, filtered_data, pos_h_index, pos_t_index)
            
            # For tail predictions:
            t_pred[t_mask_pred] = float('-inf')
            
            #compute ndcg scores 
            batch_ndcg = tasks.compute_ndcg_at_k(t_pred, t_relevance, k)  # returns a tensor of shape (B,)
            ndcg_list.append(batch_ndcg) 
            # Instead of multiplying, use where to mask out non-positive scores.

            masked_t_pred = torch.where(t_relevance.bool(), t_pred, torch.tensor(float('-inf'), device=device))
            pos_scores = masked_t_pred.max(dim=1)[0]
            negative_scores = torch.where(t_mask.bool(), t_pred, torch.tensor(float('-inf'), device=device))
            batch_ranks = (negative_scores >= pos_scores.unsqueeze(1)).sum(dim=1) + 1  # (B,)
            
            rank_list.append(batch_ranks)
        else:
            # 4. Build candidate set for these users.
            # We assume tasks.build_candidate_set returns (t_batch, h_batch) and we ignore h_batch.
            t_batch, _ = tasks.build_candidate_set(test_data, filtered_data, user_batch, cand_size=1000, num_users=num_users)
            # t_batch should be of shape (B, cand_size, 2) where the second column contains candidate tail indices.
            
            # 5. Create a full prediction tensor for tails: (B, test_data.num_nodes) filled with -inf.
            t_pred = torch.full((B, test_data.num_nodes), float('-inf'), device=device)
            
            # 6. Get model predictions for the candidate set.
            # Expected output shape: (B, cand_size)
            t_pred_batch = model(test_data, t_batch, generic_target)
            
            # 7. Scatter candidate scores into the full prediction tensor.
            t_indices = t_batch[:, :, 1]  # Candidate tail indices (B, cand_size)
            t_pred = t_pred.scatter(1, t_indices, t_pred_batch)
            # 8. Compute the relevance mask.
            # For per-user evaluation, we assume tasks.strict_negative_mask accepts a batch of user indices.
            t_relevance_neg, h_relevance_neg = tasks.strict_negative_mask(test_data, user_batch, context=3)
            t_relevance, _ = tasks.invert_mask(t_relevance_neg, h_relevance_neg, test_data.num_users)
            # t_batch has shape (B, cand_size, 2), where t_batch[i, :, 1] are candidate item IDs for user i.
            # t_relevance has shape (B, num_items).
            
            # 9. Compute NDGC for this batch.
            batch_ndcg = tasks.compute_ndcg_at_k(t_pred, t_relevance, k)  # returns a tensor of shape (B,)
            ndcg_list.append(batch_ndcg)
            
            # Instead of multiplying, use where to mask out non-positive scores.
            masked_t_pred = torch.where(t_relevance.bool(), t_pred, torch.tensor(float('-inf'), device=device))
            negative_scores = torch.where(~t_relevance.bool(), t_pred, torch.tensor(float('-inf'), device=device))
            pos_scores = masked_t_pred.max(dim=1)[0]
            batch_ranks = (negative_scores >= pos_scores.unsqueeze(1)).sum(dim=1) + 1  # (B,)
            
            rank_list.append(batch_ranks)
            
    # 11. Concatenate batch results locally.
    all_ndcg = torch.cat(ndcg_list, dim=0)   # shape: (num_users_local,)
    all_ranks = torch.cat(rank_list, dim=0)     # shape: (num_users_local,)
    print(f"Local all_ndcg.shape : {all_ndcg.shape}")
    
    # 12. Distributed aggregation using all_gather.
    if world_size > 1:
        # Prepare placeholders for gathered results.
        gathered_ndcg = [torch.zeros_like(all_ndcg) for _ in range(world_size)]
        gathered_ranks = [torch.zeros_like(all_ranks) for _ in range(world_size)]
        gathered_num_neg = [torch.zeros_like(all_num_neg) for _ in range(world_size)]
        
        # All-gather the local results from all ranks.
        dist.all_gather(gathered_ndcg, all_ndcg)
        dist.all_gather(gathered_ranks, all_ranks)
        dist.all_gather(gathered_num_neg, all_num_neg)
        
        # Concatenate along the 0 dimension to form the global tensors.
        all_ndcg = torch.cat(gathered_ndcg, dim=0)
        all_ranks = torch.cat(gathered_ranks, dim=0)
        all_num_neg = torch.cat(gathered_num_neg, dim=0)
        
    print(f"Global all_ndcg.shape : {all_ndcg.shape}")


    if rank == 0:
        # Compute base metrics
        metrics = {
            "mrr": (1.0 / all_ranks.float()).mean().item(),
            f"ndcg@{k}": all_ndcg.mean().item()
        }
    
        # Compute hits@k for each configured cutoff
        for metric in cfg.task.metric:
            if metric.startswith("hits@"):
                try:
                    cutoff = int(metric.split("@")[1])
                except ValueError:
                    cutoff = 10
                metrics[metric] = (all_ranks <= cutoff).float().mean().item()
    
        # Log metrics
        for key, value in metrics.items():
            logger.warning("%s: %g", key, value)
    
    # Return either MRR or the full metrics dictionary
    return metrics if return_metrics else metrics["mrr"]


   


@torch.no_grad()
def test(cfg, model, test_data, device, logger, filtered_data=None, return_metrics=False, valid_data = None, nr_eval_negs = 100):
    if not nr_eval_negs == 100:
        return test_per_user(cfg, model, test_data, device, logger, filtered_data, return_metrics, valid_data, nr_eval_negs)
    world_size = util.get_world_size()
    user_ids = torch.unique(test_data.target_edge_index[0], sorted=True)
    print(f"user_ids.shape : {user_ids.shape}")
    print(f"test_data.target_edge_index.shape : {test_data.target_edge_index.shape}")
    rank = util.get_rank()
    num_users = test_data.num_users
    wandb_on = cfg.train["wandb"]
        
    # Generate a linear index tensor for the edges
    num_edges = test_data.target_edge_index.size(1)  
    edge_indices = torch.arange(num_edges, dtype=torch.int64, device=test_data.target_edge_index.device).unsqueeze(1)  # Shape: (num_edges, 1)

    # Concatenate test triplets with the edge index
    test_triplets_with_index = torch.cat([
        test_data.target_edge_index.t(),  # Shape: (num_edges, 2)
        test_data.target_edge_type.unsqueeze(1),  # Shape: (num_edges, 1)
        edge_indices  # Shape: (num_edges, 1)
    ], dim=1)  # Final Shape: (num_edges, 2 + 1 + 1)
    
    sampler = torch_data.DistributedSampler(test_triplets_with_index, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets_with_index, cfg.train.test_batch_size, sampler=sampler)
    model.eval()
    rankings = []
    num_negatives = []
    ndcgs = []
    tail_ndcgs = []
    tail_rankings, num_tail_negs = [], []  # for explicit tail-only evaluation needed for 5 datasets
    for batch_with_idx in test_loader:

        batch = batch_with_idx[:,:3]
        edge_indices = batch_with_idx[:, 3].long()
        target_edge_attr = test_data.target_edge_attr[edge_indices,:]
        #test_functions.debug_edge_attr_alignment(test_data, torch.cat([batch,target_edge_attr], dim = 1))
        #raise ValueError("until here only")
        #nr_eval_negs = 100
        
        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)


        pos_h_index, pos_t_index, pos_r_index = batch.t()
        if nr_eval_negs == -1:
            t_batch, h_batch = tasks.all_negative(test_data, batch)
            t_pred = model(test_data, t_batch, target_edge_attr)
            if False:
                # Define a tolerance for floating point differences
                tolerance = 1e-6
                
                # Compute the min and max for each row
                row_min = t_pred.min(dim=1).values
                row_max = t_pred.max(dim=1).values
                
                # Check if the difference is below tolerance for each row
                is_constant = (row_max - row_min) < tolerance
                print(f"[DEBUG] is_constant {is_constant}")
                # Print results
                num_constant = is_constant.sum().item()
                print(f"[DEBUG] {num_constant}/{t_pred.shape[0]} rows in t_pred are constant.")
                if False:
                    pos_pred = t_pred.gather(-1, pos_t_index.unsqueeze(-1))
                    print(f"[DEBUG] Positive scores (pos_pred) per row:\n{pos_pred}")
                    # Compute row-wise average of predictions for all entries where t_mask is True.
                    row_avg_neg = []  # to store average negative score per row.
                    for i in range(t_pred.shape[0]):
                        # Select predictions for row i where the mask is True.
                        neg_scores = t_pred[i][t_mask[i]]
                        if neg_scores.numel() > 0:
                            avg_neg = neg_scores.mean().item()
                        else:
                            avg_neg = float('nan')
                        row_avg_neg.append(avg_neg)
                        pos_val = pos_pred[i].item()
                        print(f"[DEBUG] Row {i}: pos_pred = {pos_val:.4f}, avg_neg_score = {avg_neg:.4f}")
                        if abs(avg_neg - pos_val) < 1e-6:
                            print(f"[DEBUG] Row {i}: Negative scores match the positive score.")
            h_pred = model(test_data, h_batch, target_edge_attr)
            #t_pred= (bs, num_nodes)
                # compute ndcg:
        
            # compute t_rel/ h_rel = (bs, num_nodes) all should have 1 that are in the test set:
            t_relevance_neg, h_relevance_neg = tasks.strict_negative_mask(test_data, batch, context = 3)
            t_relevance,h_relevance = tasks.invert_mask(t_relevance_neg, h_relevance_neg, num_users)
            #test_functions.validate_relevance(t_relevance, h_relevance, test_data, pos_h_index, pos_t_index)
    
            # mask out all scores of known edges. 
            t_mask_inv, h_mask_inv = tasks.invert_mask(t_mask, h_mask, num_users)
            # mask out pos_t/h_index 
            t_mask_pred = t_mask_inv.logical_xor(t_relevance)
            h_mask_pred = h_mask_inv.logical_xor(h_relevance)
            #test_functions.validate_pred_mask(t_mask_pred, h_mask_pred, test_data, filtered_data, pos_h_index, pos_t_index)
            
            # For tail predictions:
            if False:
                violation_mask = t_relevance.bool() & t_mask.bool()
                if violation_mask.any():
                    num_violations = violation_mask.sum().item()
                    print("Mismatch found: {} entries where t_relevance == 1 but t_mask == 1.".format(num_violations))
                t_pos_scores = t_pred[t_relevance.bool()]  # scores for positive (relevant) predictions
                t_neg_mask = (~t_relevance.bool()) & (~t_mask_pred.bool())
                t_neg_scores = t_pred[t_neg_mask]
                
                avg_t_pos = t_pos_scores.mean() if t_pos_scores.numel() > 0 else torch.tensor(float('nan'))
                avg_t_neg = t_neg_scores.mean() if t_neg_scores.numel() > 0 else torch.tensor(float('nan'))
                
                print("Tail predictions -- Avg. positive score: {:.4f}, Avg. negative score: {:.4f}"
                      .format(avg_t_pos.item(), avg_t_neg.item()))

            
            #t_pred[t_mask_pred] = float('-inf')
            h_pred[h_mask_pred] = float('-inf')
            
            if False:
               # Debug: Compute per-row averages for positive (relevant) scores and negative scores (where mask is True)
                for i in range(t_pred.shape[0]):
                    # Average score for entries where relevance is 1
                    pos_entries = t_pred[i][t_relevance[i].bool()]
                    avg_pos = pos_entries.mean().item() if pos_entries.numel() > 0 else float('nan')
                    
                    # Average score for entries where t_mask is True (candidates for negatives)
                    neg_entries = t_pred[i][t_mask[i].bool()]
                    avg_neg = neg_entries.mean().item() if neg_entries.numel() > 0 else float('nan')
                    
                    print(f"[DEBUG] Row {i}: avg positive (rel) score: {avg_pos:.4f}, avg negative (mask) score: {avg_neg:.4f}")
            
            #compute ndcg scores 
            t_ndcg = tasks.compute_ndcg_at_k(t_pred, t_relevance, k)
            if False: 
                # Compute the average NDCG over the batch
                avg_t_ndcg = t_ndcg.mean().item()
                print("Average NDCG: {:.4f}".format(avg_t_ndcg))
            h_ndcg = tasks.compute_ndcg_at_k(h_pred, h_relevance, k)
            ndcgs += [t_ndcg, h_ndcg]
            tail_ndcgs +=  [t_ndcg]
            
        elif nr_eval_negs == 1000:
            #print ("im here against 1000")
            # we need to build the cadidate set with all positives per user and the remaining negatives such that set has size 1000
            batch_size = batch.size(0)
            # Create tensors filled with -infinity
            t_pred = torch.full((batch_size, test_data.num_nodes), float('-inf'), device=batch.device)
            h_pred = torch.full((batch_size, test_data.num_nodes), float('-inf'), device=batch.device)
            
            # Split the batch into t_batch and h_batch
            t_batch, h_batch = tasks.build_candidate_set(test_data, filtered_data, batch, cand_size=1000, num_users=test_data.num_users)
            print (f"t_batch.shape :{t_batch.shape}")
            print (f"target_edge_attr.shape: target_edge_attr.shape")
            # Get predictions for the sampled negatives
            t_pred_batch = model(test_data, t_batch, target_edge_attr)  # Shape: (batch_size, nr_eval_negs + 1)
            h_pred_batch = model(test_data, h_batch, target_edge_attr)  # Shape: (batch_size, nr_eval_negs + 1)

            # Use scatter to populate t_pred and h_pred efficiently
            # Extract the tail indices from t_batch and head indices from h_batch
            t_indices = t_batch[:, :, 1]  # Tail node indices, shape: (batch_size, 101)
            h_indices = h_batch[:, :, 0]  # Head node indices, shape: (batch_size, 101)
            
            # Scatter predictions into the respective tensors
            t_pred = t_pred.scatter(1, t_indices, t_pred_batch)
            h_pred = h_pred.scatter(1, h_indices, h_pred_batch)

            # compute t_rel/ h_rel = (bs, num_nodes) all should have 1 that are in the test set:
            t_relevance_neg, h_relevance_neg = tasks.strict_negative_mask(test_data, batch, context = 3)
            t_relevance,h_relevance = tasks.invert_mask(t_relevance_neg, h_relevance_neg, num_users)
            # Note no filtering is needed since we have not computed scores for things which are included in the 

            #compute ndcg scores 
            t_ndcg = tasks.compute_ndcg_at_k(t_pred, t_relevance, k)
            h_ndcg = tasks.compute_ndcg_at_k(h_pred, h_relevance, k)
            ndcgs += [t_ndcg, h_ndcg]
            tail_ndcgs +=  [t_ndcg]
            
        else:
            #print(f"im here in test and we evaluate vs: {nr_eval_negs} ")
            batch_size = batch.size(0)
            # Create tensors filled with -infinity
            t_pred = torch.full((batch_size, test_data.num_nodes), float('-inf'), device=batch.device)
            h_pred = torch.full((batch_size, test_data.num_nodes), float('-inf'), device=batch.device)
            
            # Concatenate batch for negative sampling
            batch_concat = torch.cat((batch, batch), dim=0)
            
            # Perform negative sampling
            batch_sampled = tasks.negative_sampling(filtered_data, batch_concat, nr_eval_negs, strict=True)
            
            # Split the batch into t_batch and h_batch
            t_batch = batch_sampled[:batch_size, :, :]
            h_batch = batch_sampled[batch_size:, :, :]
            
            # Get predictions for the sampled negatives
            t_pred_batch = model(test_data, t_batch, target_edge_attr)  # Shape: (batch_size, nr_eval_negs + 1)
            h_pred_batch = model(test_data, h_batch, target_edge_attr)  # Shape: (batch_size, nr_eval_negs + 1)
            

            # Use scatter to populate t_pred and h_pred efficiently
            # Extract the tail indices from t_batch and head indices from h_batch
            t_indices = t_batch[:, :, 1]  # Tail node indices, shape: (batch_size, 101)
            h_indices = h_batch[:, :, 0]  # Head node indices, shape: (batch_size, 101)
            
            # Scatter predictions into the respective tensors
            t_pred = t_pred.scatter(1, t_indices, t_pred_batch)
            h_pred = h_pred.scatter(1, h_indices, h_pred_batch)

            # Initialize relevance tensors directly with zeros
            t_relevance = torch.zeros((batch_size, test_data.num_nodes), device=batch.device)
            h_relevance = torch.zeros((batch_size, test_data.num_nodes), device=batch.device)
            
            # Set the relevance directly using advanced indexing
            t_relevance[torch.arange(batch_size, device=batch.device), t_indices[:, 0]] = 1
            h_relevance[torch.arange(batch_size, device=batch.device), h_indices[:, 0]] = 1

            #compute ndcg scores 
            t_ndcg = tasks.compute_ndcg_at_k(t_pred, t_relevance, k)
            h_ndcg = tasks.compute_ndcg_at_k(h_pred, h_relevance, k)
            ndcgs += [t_ndcg, h_ndcg]
            tail_ndcgs +=  [t_ndcg]


        
        # t_mask = (bs, num_nodes) = all valid negative tails for the given headnode in bs
        
        
        
        # pos_h_index = (bs)
    

        # the mask has now become irrelevant since the scores are already masked out but this doesnt really matter for now
        #t_mask_1, h_mask_1 = tasks.strict_negative_mask(filtered_data, batch)
        if False:
            # --- Print statistics about t_mask ---
            print("t_mask shape:", t_mask.shape)
            
            # Since t_mask is a boolean tensor, count the number of True values (1's) per batch.
            ones_per_batch = t_mask.sum(dim=1)  
            # Count zeros per batch: total columns minus ones.
            zeros_per_batch = t_mask.shape[1] - ones_per_batch
            
            print("Count of 1's per batch in t_mask:", ones_per_batch)
            print("Count of 0's per batch in t_mask:", zeros_per_batch)
            print("Percentage of 1's per batch:", (ones_per_batch.float() / t_mask.shape[1]).tolist())

        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)

        if False: 
            avg_t_ranking = t_ranking.float().mean().item()
            print("Average tail ranking: {:.4f}".format(avg_t_ranking))
            
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        #t_ranking = tasks.compute_ranking(t_pred, pos_t_index)
        #h_ranking = tasks.compute_ranking(h_pred, pos_h_index)
        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

        tail_rankings += [t_ranking]
        num_tail_negs += [num_t_negative]

    # the 3 code section below mainly ensure correct behaviour in a multi-core environment 
    # ranking is in num_workers
    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)
    ndcg_scores = torch.cat(ndcgs)
    
        
    # ugly repetitive code for tail-only ranks processing
    tail_ranking = torch.cat(tail_rankings)
    tail_ndcgs_scores= torch.cat(tail_ndcgs)
    num_tail_neg = torch.cat(num_tail_negs)
    all_size_t = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_t[rank] = len(tail_ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_t, op=dist.ReduceOp.SUM)
        
    # obtaining all ranks 
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative

    # obtaining all ndcg
    all_ndcgs = torch.zeros(all_size.sum(), dtype=torch.float, device=device)
    all_ndcgs[cum_size[rank] - all_size[rank]: cum_size[rank]] = ndcg_scores

    # the same for tails-only ranks
    cum_size_t = all_size_t.cumsum(0)
    all_ranking_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_ranking_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = tail_ranking
    all_num_negative_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_num_negative_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = num_tail_neg

    # the same for tail-only ncdgs
    all_ndcgs_t = torch.zeros(all_size_t.sum(), dtype=torch.float, device=device)
    all_ndcgs_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = tail_ndcgs_scores
    
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_ranking_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_ndcgs, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_ndcgs_t, op=dist.ReduceOp.SUM)
        
    # print (all_ranking.size())
    # i thinkg all_ranking = (test_size,2) yes but its (test_size*2)
    metrics = {}
    if rank == 0:
        for metric in cfg.task.metric:
            if "-tail" in metric:
                _metric_name, direction = metric.split("-")
                if direction != "tail":
                    raise ValueError("Only tail metric is supported in this mode")
                _ranking = all_ranking_t
                _num_neg = all_num_negative_t
            else:
                #_ranking = all_ranking 
                #_num_neg = all_num_negative 
                 # we are only interested in tail prediction:
                _ranking = all_ranking_t
                _num_neg = all_num_negative_t
                _metric_name = metric
                _ndcgs = all_ndcgs_t
            
            if _metric_name == "mr":
                score = _ranking.float().mean()
            elif _metric_name == "mrr":
                score = (1 / _ranking.float()).mean()
            elif _metric_name == "ndcg@20":
                 score = _ndcgs.mean().item()
            elif _metric_name.startswith("hits@"):
                values = _metric_name[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    print ("im doing biased")
                    num_sample = int(values[1])
                    # unbiased estimation
                    fp_rate = (_ranking - 1).float() / _num_neg
                    score = 0
                    for i in range(threshold):
                        # choose i false positive from num_sample - 1 negatives
                        num_comb = math.factorial(num_sample - 1) / \
                                   math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                else:
                    print (f"_ranking.shape {_ranking.shape}")
                    score = (_ranking <= threshold).float().mean()
            logger.warning("%s: %g" % (metric, score))
            metrics[metric] = score
    
    mrr = (1 / all_ranking.float()).mean()

    return mrr if not return_metrics else metrics


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())
    #torch.manual_seed(42)
    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

        
    # Initialize Weights & Biases run and assign proper name to the run
    dataset_name = cfg.dataset["class"]
    is_amazon = dataset_name.startswith("Amazon")
    nr_eval_negs = util.set_eval_negs(dataset_name)
    if not nr_eval_negs == -1:
        k = 10
    

        
    run_name = util.get_run_name(cfg)
    wandb_on = cfg.train["wandb"]
    cfg["run_name"] = f"{dataset_name}-{run_name}"
    if wandb_on:
        wandb.init(
            entity = "pitri-eth-z-rich", project="tl4rec", name= f"{dataset_name}-{run_name}", config=cfg)
        
    task_name = cfg.task["name"]
    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)
    #test_functions.test_pyG_graph(dataset)
    
    
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    test_functions.test_pyG_graph([train_data, valid_data, test_data])
    # make datasets smaller since we dont use edge_features
    print ("discarded node_features")
    train_data.x_user = None
    train_data.x_item = None
    valid_data.x_user = None
    valid_data.x_item = None
    test_data.x_user = None
    test_data.x_item = None
    
    # set bpe
    num_edges = train_data.edge_index.size(1)
    bpe = util.set_bpe(cfg,num_edges)
    print(f"bpe = {bpe}")
    cfg.train["batch_per_epoch"]= bpe
    #cfg.train["batch_per_epoch"]= 10
    #print ("This needs to be changed")
    #print (f"bpe {bpe}")
  
        
    # print some dataset statistics
    print (f"edge_attr.shape = {train_data.edge_attr.shape}")
    #raise ValueError("until here")
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)
    #print(f"Number of nodes: {train_data.num_nodes}")
    # entity_model needs to know the dimensions of the relation model
    
    

    # adding the input_dims for the projection mlp's
    cfg.model.edge_projection["input_dim"] = train_data.edge_attr.size(1)
    model = Gru_Ultra(
        cfg = cfg.model)

    #model = My_LightGCN(train_data.num_nodes)
    
    #model = Ultra(
    #    rel_model_cfg= rel_model_cfg,
     #   entity_model_cfg= entity_model_cfg,
      #  embedding_user_cfg = cfg.model.embedding_user,
       # embedding_item_cfg = cfg.model.embedding_item
    #)

    
    
    fine_tuning = cfg.train.num_epoch < 4
    #model = pyg.compile(model, dynamic=True)
    model = model.to(device)
    if wandb_on:
        wandb.watch(model, log= None)
    if cfg.train["init_linear_weights"]:
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
        # Apply the weight initialization
        model.apply(weights_init)
        
    if "checkpoint" in cfg and cfg.checkpoint is not None:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.ultra.load_state_dict(state["model"])
        #print ("this needs to be changed")
        #model.load_state_dict(state["model"])
    
    if task_name == "InductiveInference":
        # filtering for inductive datasets
        # Grail, MTDEA, HM datasets have validation sets based off the training graph
        # ILPC, Ingram have validation sets from the inference graph
        # filtering dataset should contain all true edges (base graph + (valid) + test) 
        if "ILPC" in cfg.dataset['class'] or "Ingram" in cfg.dataset['class']:
            # add inference, valid, test as the validation and test filtering graphs
            full_inference_edges = torch.cat([valid_data.edge_index, valid_data.target_edge_index, test_data.target_edge_index], dim=1)
            full_inference_etypes = torch.cat([valid_data.edge_type, valid_data.target_edge_type, test_data.target_edge_type])
            test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)
            val_filtered_data = test_filtered_data
        else:
            # test filtering graph: inference edges + test edges
            full_inference_edges = torch.cat([test_data.edge_index, test_data.target_edge_index], dim=1)
            full_inference_etypes = torch.cat([test_data.edge_type, test_data.target_edge_type])
            test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)

            # validation filtering graph: train edges + validation edges
            val_filtered_data = Data(
                edge_index=torch.cat([train_data.edge_index, valid_data.target_edge_index], dim=1),
                edge_type=torch.cat([train_data.edge_type, valid_data.target_edge_type])
            )
    else:
        # for transductive setting, use the whole graph for filtered ranking
        # train target edges are the directed edges
        # dataset._data.target_edge_index contains all edges of the graph.
        filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=dataset[0].num_nodes, num_relations=dataset[0].num_relations, num_users = dataset[0].num_users, num_items = dataset[0].num_items)
        val_filtered_data = test_filtered_data = filtered_data
    
    val_filtered_data = val_filtered_data.to(device)
    test_filtered_data = test_filtered_data.to(device)

    
    train_and_validate(cfg, model, train_data, valid_data, filtered_data=val_filtered_data, device=device, batch_per_epoch=cfg.train.batch_per_epoch, logger=logger)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")

    if is_amazon:
        util.synchronize()
        if util.get_rank() == 0:
            logger.warning("Amazon dataset detected. Performing 5 evaluations and averaging results.")
        
        test_results = []
        for _ in range(5):
            result = test(cfg, model, valid_data, filtered_data=test_filtered_data, device=device, logger=logger, return_metrics=True, valid_data=valid_data, nr_eval_negs = nr_eval_negs)
            test_results.append(result)
        
        # Synchronize all processes before averaging results
        util.synchronize()
        
        # Compute average performance only on rank 0
        if util.get_rank() == 0:
            result_valid = {metric: sum(r[metric] for r in test_results) / 5 for metric in test_results[0]}
    else:
        pass
        #result_valid = test(cfg, model, valid_data, filtered_data=test_filtered_data, device=device, logger=logger, return_metrics=True, valid_data=valid_data, nr_eval_negs= nr_eval_negs)
    
    # Log metrics only on rank 0
    if util.get_rank() == 0 and wandb_on:
        for metric, score in result_valid.items():
            wandb.summary[f"validation/performance/{metric}"] = score

    
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")

    
    if is_amazon:
        util.synchronize()
        if util.get_rank() == 0:
            logger.warning("Amazon dataset detected. Performing 5 evaluations and averaging results.")
        
        test_results = []
        for _ in range(5):
            result = test(cfg, model, test_data, filtered_data=test_filtered_data, device=device, logger=logger, return_metrics=True, valid_data=valid_data, nr_eval_negs= nr_eval_negs)
            test_results.append(result)
        
        # Synchronize all processes before averaging results
        util.synchronize()
        
        # Compute average performance only on rank 0
        if util.get_rank() == 0:
            result_test = {metric: sum(r[metric] for r in test_results) / 5 for metric in test_results[0]}
    else:
        result_test = test(cfg, model, test_data, filtered_data=test_filtered_data, device=device, logger=logger, return_metrics=True, valid_data=valid_data, nr_eval_negs= nr_eval_negs)

    # Log metrics only on rank 0
    if util.get_rank() == 0 and wandb_on:
        for metric, score in result_test.items():
            wandb.summary[f"test/performance/{metric}"] = score
        
    if util.get_rank() == 0 and cfg.train["save_results_db"]:
        # Define a custom path for the SQLite database
        DB_FILE = "//itet-stor/trachsele/net_scratch/tl4rec/model_outputs/results.db" 
        # Ensure the directory exists before writing
        Path(DB_FILE).parent.mkdir(parents=True, exist_ok=True)
        run_data = util.build_run_data(cfg, dataset_name, result_valid, result_test)
        util.log_results_to_db(run_data, db_path=DB_FILE)
    
 
    
   
    
 
    
   