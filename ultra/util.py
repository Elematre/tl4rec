import os
import sys
import ast
import copy
import time
import logging
import argparse
import datetime

import yaml
import jinja2
import wandb
from jinja2 import meta
import easydict

import torch
from torch import distributed as dist
from torch_geometric.data import Data
from torch_geometric.datasets import RelLinkPredDataset, WordNet18RR

from ultra import models, datasets


logger = logging.getLogger(__file__)

def set_bpe(cfg, num_edges):
    epochs = cfg.train["num_epoch"]
    bs = cfg.train["batch_size"]
    if epochs < 3:
        bpe = (num_edges / bs) // 13
        
    else:
        bpe = (num_edges / bs) // epochs 
    return int(bpe)
    #bpe = (train_data.edge_index.size(1) / bs) // epochs 
    #cfg.train["batch_per_epoch"]= int(bpe)

def set_eval_negs(name):
    if name.startswith("Yelp") or name.startswith("Gowalla"):
        print ("We will evaluate vs all negatives")
        return -1
    elif name.startswith("Amazon"):
        print ("We will evaluate vs 100 negatives")
        return 100
    else: 
        print ("We will evaluate vs 1000 negatives")
        return 1000
    

def recursive_to_plain(d):
    """
    Recursively convert EasyDict (or dict-like) objects into plain Python dicts.
    """
    if isinstance(d, dict) or isinstance(d, easydict.EasyDict):
        return {k: recursive_to_plain(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [recursive_to_plain(item) for item in d]
    else:
        return d

def save_cfg_of_best_params(best_params, cfg):
    """
    best_params: result of hyperparameter search (a dict)
    cfg: original config (possibly an EasyDict)
    Saves a new YAML configuration file determined by best_params.
    """
    best_config = copy.deepcopy(cfg)
    best_config.optimizer["projection_edge_lr"] = best_params["projection_edge_lr"]
    best_config.optimizer["backbone_conv_lr"] = best_params["backbone_conv_lr"]
    best_config.optimizer["backbone_mlp_edge_lr"] = best_params["backbone_mlp_edge_lr"]

    # Update edge_projection parameters:
    best_config.model["edge_projection"]["use_dropout"] = best_params["edge_projection_use_dropout"]
    best_config.model["edge_projection"]["dropout_rate"]   = best_params["edge_projection_dropout_rate"]
    best_config.model["edge_projection"]["use_layer_norm"] = best_params["edge_projection_use_layer_norm"]
    edge_emb_dim = best_params["edge_emb_dim"]
    num_edge_proj_layers = best_params["num_edge_proj_layers"]
    best_config.model["edge_projection"]["hidden_dims"] = [edge_emb_dim] * num_edge_proj_layers

    # Update embedding_edge parameters:
    best_config.model["backbone_model"]["embedding_edge"]["use_dropout"]    = best_params["embedding_edge_use_dropout"]
    best_config.model["backbone_model"]["embedding_edge"]["dropout_rate"]   = best_params["embedding_edge_dropout_rate"]
    best_config.model["backbone_model"]["embedding_edge"]["use_layer_norm"] = best_params["embedding_edge_use_layer_norm"]
    num_edge_emb_layers = best_params["num_edge_emb_layers"]
    best_config.model["backbone_model"]["embedding_edge"]["hidden_dims"]    = [edge_emb_dim] * num_edge_emb_layers

    # Update simple_model parameters:
    #simple_model_multiplier = best_params["simple_model_multiplier"]
    simple_model_dim =  best_params["simple_model_dim"]
    best_config.model["backbone_model"]["simple_model"]["input_dim"] = simple_model_dim 
    simple_model_num_hidden = best_params["simple_model_num_hidden"]
    best_config.model["backbone_model"]["simple_model"]["hidden_dims"] = [simple_model_dim] * simple_model_num_hidden

    # Update task parameters:
    best_config.task["num_negative"] = best_params["num_negative"]
    best_config.task["adversarial_temperature"] = best_params["adversarial_temperature"]

    # Convert the configuration to a plain dictionary recursively:
    plain_config = recursive_to_plain(best_config)

    # Define the directory and make sure it exists:
    output_dir = "/itet-stor/trachsele/net_scratch/tl4rec/config/recommender"
    os.makedirs(output_dir, exist_ok=True)
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    # Define the full file path:
    file_path = os.path.join(output_dir, f"best_config-{current_time}.yaml")
    
    # Save the configuration to the YAML file using safe_dump:
    with open(file_path, "w") as file:
        yaml.safe_dump(plain_config, file, default_flow_style=False, sort_keys=False)

    print("Best configuration saved to", file_path)

    

    


def get_run_name(cfg):
    num_epoch = cfg.train.num_epoch
    edge_proj_dim = cfg.model.edge_projection["hidden_dims"][0]
    conv_dim = cfg.model.backbone_model.simple_model["input_dim"]
    model_type = f"{edge_proj_dim}/{conv_dim}"
    
    if num_epoch == 0:
        run_type = "0-Shot"
    elif num_epoch <= 4:
        num_epoch_proj_ft = cfg.train.fine_tuning["num_epoch_proj"]
        num_epoch_whole_ft = num_epoch - num_epoch_proj_ft
        run_type = f"FT_{num_epoch_proj_ft}/{num_epoch_whole_ft}"
    else:
        run_type = f"End-to-End"
        
    if cfg.model.backbone_model.simple_model["project_conv_emb"]:
        edge_emb_usage = "edge_proj"
    else:
        edge_emb_usage = "edge_init"
        
    return f"{run_type}-{edge_emb_usage}-{model_type}"
    
def log_node_features(user_projection, item_projection, name):             
            user_mean, user_var = user_projection.mean().item(), user_projection.var().item()
            item_mean, item_var = item_projection.mean().item(), item_projection.var().item()
            print ("hi im inside of log_node_features")
            wandb.log({
                f"debug/user_{name}_mean": user_mean,
                f"debug/user_{name}_variance": user_var,
                f"debug/item_{name}_mean": item_mean,
                f"debug/item_{name}_variance": item_var
            })
                
def freeze_backbone(model):
    for name, param in model.ultra.named_parameters():
        param.requires_grad = False

def unfreeze_backbone(model):
    for name, param in model.ultra.named_parameters():
        param.requires_grad = True
            
def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    tree = env.parse(raw)
    vars = meta.find_undeclared_variables(tree)
    return vars


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg


def literal_eval(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, required=True)
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars


def get_root_logger(file=True):
    format = "%(asctime)-10s %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(format=format, datefmt=datefmt)
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    if file:
        handler = logging.FileHandler("log.txt")
        format = logging.Formatter(format, datefmt)
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def synchronize():
    if get_world_size() > 1:
        dist.barrier()


def get_device(cfg):
    if cfg.train.gpus:
        device = torch.device(cfg.train.gpus[get_rank()])
    else:
        device = torch.device("cpu")
    return device


def create_working_directory(cfg):
    file_name = "working_dir.tmp"
    world_size = get_world_size()
    if cfg.train.gpus is not None and len(cfg.train.gpus) != world_size:
        error_msg = "World size is %d but found %d GPUs in the argument"
        if world_size == 1:
            error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
        raise ValueError(error_msg % (world_size, len(cfg.train.gpus)))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://")

    working_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                               cfg.model["class"], cfg.dataset["class"], time.strftime("%Y-%m-%d-%H-%M-%S"))

    # synchronize working directory
    if get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)
        os.makedirs(working_dir)
    synchronize()
    if get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    synchronize()
    if get_rank() == 0:
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir


def build_dataset(cfg):
    data_config = copy.deepcopy(cfg.dataset)
    cls = data_config.pop("class")

    ds_cls = getattr(datasets, cls)
    dataset = ds_cls(**data_config)

    if get_rank() == 0:
        logger.warning("%s dataset" % (cls if "version" not in cfg.dataset else f'{cls}({cfg.dataset.version})'))
        if cls != "JointDataset":
            logger.warning("#train: %d, #valid: %d, #test: %d" %
                        (dataset[0].target_edge_index.shape[1], dataset[1].target_edge_index.shape[1],
                            dataset[2].target_edge_index.shape[1]))
        else:
            logger.warning("#train: %d, #valid: %d, #test: %d" %
                           (sum(d.target_edge_index.shape[1] for d in dataset._data[0]),
                            sum(d.target_edge_index.shape[1] for d in dataset._data[1]),
                            sum(d.target_edge_index.shape[1] for d in dataset._data[2]),
                            ))

    return dataset

