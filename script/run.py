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

    train_triplets = torch.cat([train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(train_triplets, cfg.train.batch_size, sampler=sampler)

    batch_per_epoch = batch_per_epoch or len(train_loader)
    
    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
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
            for batch in islice(train_loader, batch_per_epoch):
                batch = tasks.negative_sampling(train_data, batch, cfg.task.num_negative,
                                                strict=cfg.task.strict_negative)
                pred = parallel_model(train_data, batch)
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
        #result_dict = {}
        #result_dict["ndcg@20"] = 1
        result_dict = test(cfg, model, valid_data, filtered_data=filtered_data, device=device, logger=logger, return_metrics = True, nr_eval_negs = 100)   
        # Log each metric with the hierarchical key format "training/performance/{metric}"
        if wandb_on:
            for metric, score in result_dict.items():
                wandb.log({f"training/performance/{metric}": score})

        target_metric = cfg.train["target_metric"]
        result = result_dict[target_metric]
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





@torch.no_grad()
def test(cfg, model, test_data, device, logger, filtered_data=None, return_metrics=False, valid_data = None, nr_eval_negs = 100):
 
    world_size = util.get_world_size()
    rank = util.get_rank()
    num_users = test_data.num_users
    wandb_on = cfg.train["wandb"]
        
    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.test_batch_size, sampler=sampler)
    model.eval()
    rankings = []
    num_negatives = []
    ndcgs = []
    tail_ndcgs = []
    tail_rankings, num_tail_negs = [], []  # for explicit tail-only evaluation needed for 5 datasets
    for batch in test_loader:
        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)


        pos_h_index, pos_t_index, pos_r_index = batch.t()
        if nr_eval_negs == -1:
            t_batch, h_batch = tasks.all_negative(test_data, batch)
            t_pred = model(test_data, t_batch)
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
            h_pred = model(test_data, h_batch)
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

            
            # Get predictions for the sampled negatives
            t_pred_batch = model(test_data, t_batch)  # Shape: (batch_size, nr_eval_negs + 1)
            h_pred_batch = model(test_data, h_batch)  # Shape: (batch_size, nr_eval_negs + 1)

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
            t_pred_batch = model(test_data, t_batch)  # Shape: (batch_size, nr_eval_negs + 1)
            h_pred_batch = model(test_data, h_batch)  # Shape: (batch_size, nr_eval_negs + 1)
            

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
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
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
    
    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

        
    # Initialize Weights & Biases run and assign proper name to the run
    dataset_name = cfg.dataset["class"]
    is_amazon = dataset_name.startswith("Amazon")
    nr_eval_negs = util.set_eval_negs(dataset_name)
    
    run_name = util.get_run_name(cfg)
    
    if is_amazon:
        print ("We are using a amazon dataset")
        k = 10
        
    wandb_on = cfg.train["wandb"]
    if wandb_on:
        wandb.init(
            entity = "pitri-eth-z-rich", project="tl4rec", name= f"{dataset_name}-{run_name}", config=cfg)
        
    task_name = cfg.task["name"]
    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)
    
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    # print some dataset statistics
    num_edges = train_data.edge_index.size(1)
    bpe = util.set_bpe(cfg,num_edges)
    print(f"bpe = {bpe}")
    cfg.train["batch_per_epoch"]= bpe
    #raise ValueError("until here")
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)
    #print(f"Number of nodes: {train_data.num_nodes}")
    # entity_model needs to know the dimensions of the relation model
    
    if not cfg.model.get("node_features", True):
        train_data.x_user = None
        train_data.x_item = None
        valid_data.x_user = None
        valid_data.x_item = None
        test_data.x_user = None
        test_data.x_item = None
        print ("discarded node_features")
    else:
        cfg.model.user_projection["input_dim"] = train_data.x_user.size(1)
        cfg.model.item_projection["input_dim"] = train_data.x_item.size(1)
        
    train_data.edge_attr = None
    train_data.target_edge_attr = None
    valid_data.edge_attr = None
    valid_data.target_edge_attr = None
    test_data.edge_attr = None
    test_data.target_edge_attr = None
    print ("discarded edge_features")

    
    model = Gru_Ultra(cfg.model, log = wandb_on)


    if "checkpoint" in cfg and cfg.checkpoint is not None:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.ultra.load_state_dict(state["model"])
        # initialize linear weights:
        def weights_init(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        # Apply the weight initialization
        model.user_projection.apply(weights_init)
        model.item_projection.apply(weights_init)
    

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
            result = {metric: sum(r[metric] for r in test_results) / 5 for metric in test_results[0]}
    else:
        result = test(cfg, model, valid_data, filtered_data=test_filtered_data, device=device, logger=logger, return_metrics=True, valid_data=valid_data, nr_eval_negs = nr_eval_negs)
    
    # Log metrics only on rank 0
    if util.get_rank() == 0 and wandb_on:
        for metric, score in result.items():
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
            result = test(cfg, model, test_data, filtered_data=test_filtered_data, device=device, logger=logger, return_metrics=True, valid_data=valid_data,nr_eval_negs = nr_eval_negs)
            test_results.append(result)
        
        # Synchronize all processes before averaging results
        util.synchronize()
        
        # Compute average performance only on rank 0
        if util.get_rank() == 0:
            result = {metric: sum(r[metric] for r in test_results) / 5 for metric in test_results[0]}
    else:
        result = test(cfg, model, test_data, filtered_data=test_filtered_data, device=device, logger=logger, return_metrics=True, valid_data=valid_data,nr_eval_negs = nr_eval_negs)

    # Log metrics only on rank 0
    if util.get_rank() == 0 and wandb_on:
        for metric, score in result.items():
            wandb.summary[f"test/performance/{metric}"] = score
 
    
   