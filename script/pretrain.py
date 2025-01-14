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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import tasks, util, test_functions
from ultra.models import Gru_Ultra, Ultra


separator = ">" * 30
line = "-" * 30


def multigraph_collator(batch, train_graphs):
    num_graphs = len(train_graphs)
    probs = torch.tensor([graph.edge_index.shape[1] for graph in train_graphs]).float()
    probs /= probs.sum()
    graph_id = torch.multinomial(probs, 1, replacement=False).item()

    graph = train_graphs[graph_id]
    bs = len(batch)
    edge_mask = torch.randperm(graph.target_edge_index.shape[1])[:bs]

    batch = torch.cat([graph.target_edge_index[:, edge_mask], graph.target_edge_type[edge_mask].unsqueeze(0)]).t()
    return graph, batch

# here we assume that train_data and valid_data are tuples of datasets
def train_and_validate(cfg, models, train_data, valid_data, filtered_data=None, batch_per_epoch=None):

    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()
    wandb_on = cfg.train["wandb"]
    parallel_models = []
    # Combine parameters from all models
    all_params = []
    param_groups = []
    
    for i, model in enumerate(models):
        model.to(device)
        if i == 0:
            # Add backbone convolution parameters for the first model
            param_groups.append({"params": model.ultra.simple_model.parameters(), "lr": cfg.optimizer["backbone_conv_lr"]})
            param_groups.append({"params": model.ultra.user_mlp.parameters(), "lr": cfg.optimizer["backbone_mlp_user_lr"]})
            param_groups.append({"params": model.ultra.item_mlp.parameters(), "lr": cfg.optimizer["backbone_mlp_item_lr"]})
    
        if cfg.model["node_features"]:
            # Add projection MLP parameters for each model
            param_groups.append({"params": model.user_projection.parameters(), "lr": cfg.optimizer["projection_user_lr"]})
            param_groups.append({"params": model.item_projection.parameters(), "lr": cfg.optimizer["projection_item_lr"]})
    
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

        # redundant according to gptito
        #for parallel_model in parallel_models:
         #   parallel_model.train()
            
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            losses = []
            sampler.set_epoch(epoch)
            for batch in islice(train_loader, batch_per_epoch):
                # now at each step we sample a new graph and edges from it
                train_graph, batch = batch
                # based on the train_graph choose the appropriate model and optimizer
                graph_idx = graph_to_model_map[id(train_graph)]
                parallel_model = parallel_models[graph_idx]
                
                batch = tasks.negative_sampling(train_graph, batch, cfg.task.num_negative,
                                                strict=cfg.task.strict_negative)
                pred = parallel_model(train_graph, batch)
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

                # Log gradient norms
                if wandb_on:
                    for name, param in parallel_model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            wandb.log({f"gradients/{name}": grad_norm})
                            
                optimizer.step()
                optimizer.zero_grad()

               

                if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("binary cross entropy: %g" % loss)
                    if wandb_on:
                         # Positive and negative scores
                        pos_scores = pred[:, 0].detach()
                        neg_scores = pred[:, 1:].detach()
                        avg_pos_score = pos_scores.mean().item()
                        avg_neg_score = neg_scores.mean().item()
                        wandb.log({f"debug/loss_per_model/{graph_idx}": loss.item(),
                                    "debug/loss_universal": loss.item(),
                                    f"debug/avg_pos_score/{graph_idx}": avg_pos_score,
                                    f"debug/avg_neg_score/{graph_idx}": avg_neg_score})


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
        if result > best_result:
            best_result = result
            best_epoch = epoch

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)

    for model in models:  
        model.ultra.load_state_dict(state["model"])
    util.synchronize()
    
    # save the final model state
    if rank == 0:
        # Extract the last 6 letters of each dataset name
        dataset_names = cfg.dataset['graphs']  # Example: ['Amazon_Beauty', 'Amazon_Games']
        dataset_prefix = ''.join([name[-6:] for name in dataset_names])  # 'AmazoAmazo'
    
        # Get the current timestamp in a readable format
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    
        # Construct the checkpoint filename
        checkpoint_dir = "/itet-stor/trachsele/net_scratch/tl4rec/ckpts"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_name = f"{dataset_prefix}_{current_time}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
        logger.warning(f"Save final_ckpt to {checkpoint_path}")
        torch.save(state, checkpoint_path)


    


@torch.no_grad()
def test(cfg, models, test_data, filtered_data=None, context = 0):
    # context is used for determining the calling context of test (train, valid, test)
    
    k = 20
    world_size = util.get_world_size()
    rank = util.get_rank()
    wandb_on = cfg.train["wandb"]
    
    # test_data is a tuple of validation/test datasets
    # process sequentially
    all_metrics = []
    dataset_nr = 0
    for model, test_graph, filters in zip(models, test_data, filtered_data):
        num_users = test_graph.num_users
        test_triplets = torch.cat([test_graph.target_edge_index, test_graph.target_edge_type.unsqueeze(0)]).t()
        sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
        test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

        model.eval()
        rankings = []
        num_negatives = []
        ndcgs = []
        for batch in test_loader:
            t_batch, h_batch = tasks.all_negative(test_graph, batch)
            t_pred = model(test_graph, t_batch)
            h_pred = model(test_graph, h_batch)

            if filtered_data is None:
                t_mask, h_mask = tasks.strict_negative_mask(test_graph, batch)
            else:
                t_mask, h_mask = tasks.strict_negative_mask(filters, batch)
            pos_h_index, pos_t_index, pos_r_index = batch.t()
            
            # compute ndcg@20
            # compute t_rel/ h_rel = (bs, num_nodes) all should have 1 that are in the test set:
            t_relevance_neg, h_relevance_neg = tasks.strict_negative_mask(test_graph, batch, context = 3)
            t_relevance,h_relevance = tasks.invert_mask(t_relevance_neg, h_relevance_neg, num_users)
            #test_functions.validate_relevance(t_relevance, h_relevance, test_graph, pos_h_index, pos_t_index)
            
            # mask out all scores of known edges. 
            t_mask_inv, h_mask_inv = tasks.invert_mask(t_mask, h_mask, num_users)
            # mask out pos_t/h_index 
            t_mask_pred = t_mask_inv.logical_xor(t_relevance)
            h_mask_pred = h_mask_inv.logical_xor(h_relevance)
            #test_functions.validate_pred_mask(t_mask_pred, h_mask_pred, test_graph, filters, pos_h_index, pos_t_index)
    
            t_pred[t_mask_pred] = float('-inf')
            h_pred[h_mask_pred] = float('-inf')
        
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
        
        all_metrics.append(mrr)
        if rank == 0:
            logger.warning(separator)
            
        if rank == 0 and wandb_on:
            for metric, score in metrics.items():
                if context == 2:
                    wandb.summary[f"test/performance/{dataset_nr}/{metric}"] = score
                elif context == 1:
                    wandb.summary[f"validation/performance/{dataset_nr}/{metric}"] = score
                else:
                    wandb.log({f"train_validation/performance/{dataset_nr}/{metric}": score})
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

    train_data = [td.to(device) for td in train_data]
    valid_data = [vd.to(device) for vd in valid_data]
    test_data = [tst.to(device) for tst in test_data]
    
    #for td in train_data:
        #print (td)
    


    
    

   

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"pretrain-{current_time}"
    wandb_on = cfg.train["wandb"]
    if wandb_on:
        wandb.init(
            entity = "pitri-eth-z-rich", project="tl4rec", name=run_name, config=cfg)

    # initialize the list of Gru-Models sharing the backbone Model
    # Shared Ultra backbone
    ultra_ref = Ultra(cfg.model, wandb_on)
    
    if "checkpoint" in cfg:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        ultra_ref.load_state_dict(state["model"])

    
    # Create a Gru_Ultra model for each dataset
    models = []
    for td in train_data:
        model_cfg = copy.deepcopy(cfg.model)
        model_cfg.user_projection["input_dim"] = td.x_user.size(1)
        model_cfg.item_projection["input_dim"] = td.x_item.size(1)
        models.append(Gru_Ultra(model_cfg, ultra_ref, wandb_on))
        
   # I avoid this for now 
    #model = model.to(device)
    #if wandb_on:
        #wandb.watch(model, log= None)
    
    
    assert task_name == "MultiGraphPretraining", "Only the MultiGraphPretraining task is allowed for this script"

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

    train_and_validate(cfg, models, train_data, valid_data if "fast_test" not in cfg.train else short_valid, filtered_data=filtered_data, batch_per_epoch=cfg.train.batch_per_epoch)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")
    test(cfg, models, valid_data, filtered_data=filtered_data, context = 1)
  
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    test(cfg, models, test_data, filtered_data=filtered_data, context = 2)