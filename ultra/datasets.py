import os
import csv
import shutil
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.datasets import RelLinkPredDataset, WordNet18RR, MovieLens100K
from collections import defaultdict
from ultra.tasks import build_relation_graph
import matplotlib.pyplot as plt

#this is a test for pushing to git 2

class GrailInductiveDataset(InMemoryDataset):

    def __init__(self, root, version, transform=None, pre_transform=build_relation_graph, merge_valid_test=True):
        self.version = version
        assert version in ["v1", "v2", "v3", "v4"]

        # by default, most models on Grail datasets merge inductive valid and test splits as the final test split
        # with this choice, the validation set is that of the transductive train (on the seen graph)
        # by default it's turned on but you can experiment with turning this option off
        # you'll need to delete the processed datasets then and re-run to cache a new dataset
        self.merge_valid_test = merge_valid_test
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, "grail", self.name, self.version, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "grail", self.name, self.version, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def raw_file_names(self):
        return [
            "train_ind.txt", "valid_ind.txt", "test_ind.txt", "train.txt", "valid.txt"
        ]

    def download(self):
        for url, path in zip(self.urls, self.raw_paths):
            download_path = download_url(url % self.version, self.raw_dir)
            os.rename(download_path, path)

    def process(self):
        test_files = self.raw_paths[:3]
        train_files = self.raw_paths[3:]

        inv_train_entity_vocab = {}
        inv_test_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        for txt_file in train_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[h_token] = len(inv_train_entity_vocab)
                    h = inv_train_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[t_token] = len(inv_train_entity_vocab)
                    t = inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in test_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[h_token] = len(inv_test_entity_vocab)
                    h = inv_test_entity_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[t_token] = len(inv_test_entity_vocab)
                    t = inv_test_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)
        triplets = torch.tensor(triplets)

        edge_index = triplets[:, :2].t()
        edge_type = triplets[:, 2]
        num_relations = int(edge_type.max()) + 1

        # creating fact graphs - those are graphs sent to a model, based on which we'll predict missing facts
        # also, those fact graphs will be used for filtered evaluation
        train_fact_slice = slice(None, sum(num_samples[:1]))
        test_fact_slice = slice(sum(num_samples[:2]), sum(num_samples[:3]))
        train_fact_index = edge_index[:, train_fact_slice]
        train_fact_type = edge_type[train_fact_slice]
        test_fact_index = edge_index[:, test_fact_slice]
        test_fact_type = edge_type[test_fact_slice]

        # add flipped triplets for the fact graphs
        train_fact_index = torch.cat([train_fact_index, train_fact_index.flip(0)], dim=-1)
        train_fact_type = torch.cat([train_fact_type, train_fact_type + num_relations])
        test_fact_index = torch.cat([test_fact_index, test_fact_index.flip(0)], dim=-1)
        test_fact_type = torch.cat([test_fact_type, test_fact_type + num_relations])

        train_slice = slice(None, sum(num_samples[:1]))
        valid_slice = slice(sum(num_samples[:1]), sum(num_samples[:2]))
        # by default, SOTA models on Grail datasets merge inductive valid and test splits as the final test split
        # with this choice, the validation set is that of the transductive train (on the seen graph)
        # by default it's turned on but you can experiment with turning this option off
        test_slice = slice(sum(num_samples[:3]), sum(num_samples)) if self.merge_valid_test else slice(sum(num_samples[:4]), sum(num_samples))
        
        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, train_slice], target_edge_type=edge_type[train_slice], num_relations=num_relations*2)
        valid_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, valid_slice], target_edge_type=edge_type[valid_slice], num_relations=num_relations*2)
        test_data = Data(edge_index=test_fact_index, edge_type=test_fact_type, num_nodes=len(inv_test_entity_vocab),
                         target_edge_index=edge_index[:, test_slice], target_edge_type=edge_type[test_slice], num_relations=num_relations*2)

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

    def __repr__(self):
        return "%s(%s)" % (self.name, self.version)


class FB15k237Inductive(GrailInductiveDataset):

    urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/test.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt"
    ]

    name = "IndFB15k237"

    def __init__(self, root, version):
        super().__init__(root, version)

class WN18RRInductive(GrailInductiveDataset):

    urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/test.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt"
    ]

    name = "IndWN18RR"

    def __init__(self, root, version):
        super().__init__(root, version)

class NELLInductive(GrailInductiveDataset):
    urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s_ind/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s_ind/test.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s/valid.txt"
    ]
    name = "IndNELL"

    def __init__(self, root, version):
        super().__init__(root, version)


def FB15k237(root):
    dataset = RelLinkPredDataset(name="FB15k-237", root=root+"/fb15k237/")
    data = dataset.data
    train_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                        target_edge_index=data.train_edge_index, target_edge_type=data.train_edge_type,
                        num_relations=dataset.num_relations)
    valid_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                        target_edge_index=data.valid_edge_index, target_edge_type=data.valid_edge_type,
                        num_relations=dataset.num_relations)
    test_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                        target_edge_index=data.test_edge_index, target_edge_type=data.test_edge_type,
                        num_relations=dataset.num_relations)
    
    # build relation graphs
    train_data = build_relation_graph(train_data)
    valid_data = build_relation_graph(valid_data)
    test_data = build_relation_graph(test_data)

    dataset.data, dataset.slices = dataset.collate([train_data, valid_data, test_data])
    return dataset

def stratified_split(edge_index, split_ratios, filter_by='item'):
    """Perform a stratified split based on the frequency distribution of items or users."""
    split_col = 1 if filter_by == 'item' else 0
    value_counts = defaultdict(list)

    # Group indices by their corresponding values (items or users)
    for i in range(edge_index.size(1)):
        value_counts[edge_index[split_col, i].item()].append(i)

    for key in value_counts:
        value_counts[key] = torch.tensor(value_counts[key])
        perm = torch.randperm(value_counts[key].size(0))  # Shuffle within each group
        value_counts[key] = value_counts[key][perm]

    splits = [[], [], []]
    thresholds = [sum(split_ratios[:i + 1]) for i in range(len(split_ratios))]

    # Distribute indices across splits
    for indices in value_counts.values():
        group_size = indices.size(0)
        cumulative = 0
        for i, threshold in enumerate(thresholds):
            split_size = int(threshold * group_size) - cumulative
            splits[i].append(indices[cumulative:cumulative + split_size])
            cumulative += split_size

    splits = [torch.cat(split) for split in splits]
    splits = [split[torch.randperm(split.size(0))] for split in splits]


    return splits



def plot_item_distribution(edge_index, split_indices, labels, filter_by='item'):
    """Plots the distribution of items in the dataset splits."""
    split_col = 1 if filter_by == 'item' else 0

    plt.figure(figsize=(12, 6))
    for i, indices in enumerate(split_indices):
        item_counts = torch.bincount(edge_index[split_col, indices])
        plt.plot(item_counts.numpy(), label=f"{labels[i]} (Mean: {item_counts.float().mean():.2f})")

    plt.title("Item Distribution Across Splits")
    plt.xlabel("Item Index")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

def MovieLens100k(root):
    # Load dataset
    dataset = MovieLens100K(root=root + "/movieLens100k/")
    edge_index = dataset[0]['user', 'rates', 'movie'].edge_index
    ratings = dataset[0]['user', 'rates', 'movie'].rating - 1
    user_features = dataset[0]["user"].x
    item_features = dataset[0]["movie"].x
    num_users = user_features.size(0)

    # Filter items with at least 1 user interaction
    #print(f"pre- Filtered edge index size: {edge_index.size()}")
    item_interactions = torch.bincount(edge_index[1])
    valid_items = torch.nonzero(item_interactions > 0).squeeze()
    valid_mask = torch.isin(edge_index[1], valid_items)
    edge_index = edge_index[:, valid_mask]
    ratings = ratings[valid_mask]
    #print(f"Filtered edge index size: {edge_index.size()}")

    # Update item-related data based on valid items
    valid_item_map = {old: new for new, old in enumerate(valid_items.tolist())}
    edge_index[1] = torch.tensor([valid_item_map[i.item()] for i in edge_index[1]])
    num_items = len(valid_items)
    item_features = item_features[valid_items]
    #print(f"Number of valid items: {num_items}, Edge index after filtering: {edge_index.size()}")

    # Adjust movie IDs to prevent overlap with user IDs
    max_user_id = edge_index[0].max()
    edge_index[1] += max_user_id + 1
    #print(f"Max user ID: {max_user_id}")

    # Perform stratified splitting by items
    split_ratios = [0.8, 0.1, 0.1]  # 80% train, 10% validation, 10% test
    train_idx, valid_idx, test_idx = stratified_split(edge_index, split_ratios, filter_by='item')
    #print(f"Train indices: {train_idx.size(0)}, Valid indices: {valid_idx.size(0)}, Test indices: {test_idx.size(0)}")

    # Debugging: Check overlap between splits
    #print(f"Train/Valid overlap: {len(set(train_idx.tolist()).intersection(valid_idx.tolist()))}")
    #print(f"Train/Test overlap: {len(set(train_idx.tolist()).intersection(test_idx.tolist()))}")
    #print(f"Valid/Test overlap: {len(set(valid_idx.tolist()).intersection(test_idx.tolist()))}")

    # Check item distribution
    #plot_item_distribution(edge_index, [train_idx, valid_idx, test_idx], ['Train', 'Validation', 'Test'])

    # Create split datasets
    train_target_edges = edge_index[:, train_idx]
    train_types = torch.zeros(train_target_edges.size(1), dtype=torch.int64)

    valid_edges = edge_index[:, valid_idx]
    valid_types = torch.zeros(valid_edges.size(1), dtype=torch.int64)

    test_edges = edge_index[:, test_idx]
    test_types = torch.zeros(test_edges.size(1), dtype=torch.int64)

    # Combine train edges with reversed edges
    train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=-1)
    train_edge_types = torch.cat([train_types, train_types + 1], dim=0)

    num_nodes = max_user_id + 1 + num_items
    num_relations = 2  # Bidirectional relations (e.g., user rates movie and movie rated by user)
    #print(f"num_nodes: {num_nodes}")
    
    # Construct Data objects
    train_data = Data(edge_index=train_edges, edge_type=train_edge_types, 
                      num_nodes=num_nodes, target_edge_index=train_target_edges, target_edge_type=train_types, num_relations=num_relations)
    
    valid_data = Data(edge_index=train_edges, edge_type=train_edge_types, num_nodes=num_nodes,
                      target_edge_index=valid_edges, target_edge_type=valid_types, num_relations=num_relations)
    
    test_data = Data(edge_index=train_edges, edge_type=train_edge_types, num_nodes=num_nodes,
                     target_edge_index=test_edges, target_edge_type=test_types, num_relations=num_relations)

    # Build relation graphs
    train_data = build_relation_graph(train_data)
    valid_data = build_relation_graph(valid_data)
    test_data = build_relation_graph(test_data)

    # Add user and item features to each split
    for data in [train_data, valid_data, test_data]:
        data.x_user = user_features
        data.x_item = item_features
        data.num_users = num_users
        data.num_items = num_items

    print("Relational graph is built")
    dataset.data, dataset.slices = dataset.collate([train_data, valid_data, test_data])

    return dataset





def WN18RR(root):
    dataset = WordNet18RR(root=root+"/wn18rr/")
    # convert wn18rr into the same format as fb15k-237
    data = dataset.data
    num_nodes = int(data.edge_index.max()) + 1
    num_relations = int(data.edge_type.max()) + 1
    edge_index = data.edge_index[:, data.train_mask]
    edge_type = data.edge_type[data.train_mask]
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)
    edge_type = torch.cat([edge_type, edge_type + num_relations])
    
    
    train_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                        target_edge_index=data.edge_index[:, data.train_mask],
                        target_edge_type=data.edge_type[data.train_mask],
                        num_relations=num_relations*2)
    valid_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                        target_edge_index=data.edge_index[:, data.val_mask],
                        target_edge_type=data.edge_type[data.val_mask],
                        num_relations=num_relations*2)
    test_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                        target_edge_index=data.edge_index[:, data.test_mask],
                        target_edge_type=data.edge_type[data.test_mask],
                        num_relations=num_relations*2)
    
        
    # build relation graphs
    train_data = build_relation_graph(train_data)
    valid_data = build_relation_graph(valid_data)
    test_data = build_relation_graph(test_data)

    dataset.data, dataset.slices = dataset.collate([train_data, valid_data, test_data])
    dataset.num_relations = num_relations * 2
    
    raise ValueError("abort.")
    
    return dataset



class TransductiveDataset(InMemoryDataset):

    delimiter = None
    
    def __init__(self, root, transform=None, pre_transform=build_relation_graph, **kwargs):

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["train.txt", "valid.txt", "test.txt"]
    
    def download(self):
        for url, path in zip(self.urls, self.raw_paths):
            download_path = download_url(url, self.raw_dir)
            os.rename(download_path, path)
    
    def load_file(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}):

        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(triplet_file, "r", encoding="utf-8") as fin:
            for l in fin:
                u, r, v = l.split() if self.delimiter is None else l.strip().split(self.delimiter)
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))

        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab), #entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }
    
    # default loading procedure: process train/valid/test files, create graphs from them
    def process(self):

        train_files = self.raw_paths[:3]

        train_results = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        valid_results = self.load_file(train_files[1], 
                        train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        test_results = self.load_file(train_files[2],
                        train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        
        # in some datasets, there are several new nodes in the test set, eg 123,143 YAGO train adn 123,182 in YAGO test
        # for consistency with other experimental results, we'll include those in the full vocab and num nodes
        num_node = test_results["num_node"] 
        # the same for rels: in most cases train == test for transductive
        # for AristoV4 train rels 1593, test 1604
        num_relations = test_results["num_relation"]

        train_triplets = train_results["triplets"]
        valid_triplets = valid_results["triplets"]
        test_triplets = test_results["triplets"]

        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_triplets], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_triplets])

        valid_edges = torch.tensor([[t[0], t[1]] for t in valid_triplets], dtype=torch.long).t()
        valid_etypes = torch.tensor([t[2] for t in valid_triplets])

        test_edges = torch.tensor([[t[0], t[1]] for t in test_triplets], dtype=torch.long).t()
        test_etypes = torch.tensor([t[2] for t in test_triplets])

        # train_edges is undirected whil train_target edges is directed why? also doubles edge types
        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat([train_target_etypes, train_target_etypes+num_relations])

        train_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_relations*2)
        valid_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=valid_edges, target_edge_type=valid_etypes, num_relations=num_relations*2)
        test_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                         target_edge_index=test_edges, target_edge_type=test_etypes, num_relations=num_relations*2)

        # build graphs of relations
        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

    def __repr__(self):
        return "%s()" % (self.name)
    
    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"
    


class MovieLens1M_pyG(TransductiveDataset):
    urls = [
        "https://raw.githubusercontent.com/Elematre/tl4rec/refs/heads/main/MovieLenseData/train_full.txt",
        "https://raw.githubusercontent.com/Elematre/tl4rec/refs/heads/main/MovieLenseData/valid_small.txt",
        "https://raw.githubusercontent.com/Elematre/tl4rec/refs/heads/main/MovieLenseData/test_small.txt",
    ]
    name = "movielens1M_pyG"
    delimiter = "\t"
    

class CoDEx(TransductiveDataset):

    name = "codex"
    urls = [
        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/%s/train.txt",
        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/%s/valid.txt",
        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/%s/test.txt",
    ]
    
    def download(self):
        for url, path in zip(self.urls, self.raw_paths):
            download_path = download_url(url % self.name, self.raw_dir)
            os.rename(download_path, path)


class CoDExSmall(CoDEx):
    """
    #node: 2034
    #edge: 36543
    #relation: 42
    """
    url = "https://zenodo.org/record/4281094/files/codex-s.tar.gz"
    md5 = "63cd8186fc2aeddc154e20cf4a10087e"
    name = "codex-s"

    def __init__(self, root):
        super(CoDExSmall, self).__init__(root=root, size='s')


class CoDExMedium(CoDEx):
    """
    #node: 17050
    #edge: 206205
    #relation: 51
    """
    url = "https://zenodo.org/record/4281094/files/codex-m.tar.gz"
    md5 = "43e561cfdca1c6ad9cc2f5b1ca4add76"
    name = "codex-m"
    def __init__(self, root):
        super(CoDExMedium, self).__init__(root=root, size='m')


class CoDExLarge(CoDEx):
    """
    #node: 77951
    #edge: 612437
    #relation: 69
    """
    url = "https://zenodo.org/record/4281094/files/codex-l.tar.gz"
    md5 = "9a10f4458c4bd2b16ef9b92b677e0d71"
    name = "codex-l"
    def __init__(self, root):
        super(CoDExLarge, self).__init__(root=root, size='l')


class NELL995(TransductiveDataset):

    # from the RED-GNN paper https://github.com/LARS-research/RED-GNN/tree/main/transductive/data/nell
    # the OG dumps were found to have test set leakages
    # training set is made out of facts+train files, so we sum up their samples to build one training graph

    urls = [
        "https://raw.githubusercontent.com/LARS-research/RED-GNN/main/transductive/data/nell/facts.txt",
        "https://raw.githubusercontent.com/LARS-research/RED-GNN/main/transductive/data/nell/train.txt",
        "https://raw.githubusercontent.com/LARS-research/RED-GNN/main/transductive/data/nell/valid.txt",
        "https://raw.githubusercontent.com/LARS-research/RED-GNN/main/transductive/data/nell/test.txt",
    ]
    name = "nell995"

    @property
    def raw_file_names(self):
        return ["facts.txt", "train.txt", "valid.txt", "test.txt"]
    

    def process(self):
        train_files = self.raw_paths[:4]

        facts_results = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        train_results = self.load_file(train_files[1], facts_results["inv_entity_vocab"], facts_results["inv_rel_vocab"])
        valid_results = self.load_file(train_files[2], train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        test_results = self.load_file(train_files[3], train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        
        num_node = valid_results["num_node"]
        num_relations = train_results["num_relation"]

        train_triplets = facts_results["triplets"] + train_results["triplets"]
        valid_triplets = valid_results["triplets"]
        test_triplets = test_results["triplets"]

        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_triplets], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_triplets])

        valid_edges = torch.tensor([[t[0], t[1]] for t in valid_triplets], dtype=torch.long).t()
        valid_etypes = torch.tensor([t[2] for t in valid_triplets])

        test_edges = torch.tensor([[t[0], t[1]] for t in test_triplets], dtype=torch.long).t()
        test_etypes = torch.tensor([t[2] for t in test_triplets])

        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat([train_target_etypes, train_target_etypes+num_relations])

        train_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_relations*2)
        valid_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=valid_edges, target_edge_type=valid_etypes, num_relations=num_relations*2)
        test_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                         target_edge_index=test_edges, target_edge_type=test_etypes, num_relations=num_relations*2)

        # build graphs of relations
        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])


class ConceptNet100k(TransductiveDataset):

    urls = [
        "https://raw.githubusercontent.com/guojiapub/BiQUE/master/src_data/conceptnet-100k/train",
        "https://raw.githubusercontent.com/guojiapub/BiQUE/master/src_data/conceptnet-100k/valid",
        "https://raw.githubusercontent.com/guojiapub/BiQUE/master/src_data/conceptnet-100k/test",
    ]
    name = "cnet100k"
    delimiter = "\t"


class DBpedia100k(TransductiveDataset):
    urls = [
        "https://raw.githubusercontent.com/iieir-km/ComplEx-NNE_AER/master/datasets/DB100K/_train.txt",
        "https://raw.githubusercontent.com/iieir-km/ComplEx-NNE_AER/master/datasets/DB100K/_valid.txt",
        "https://raw.githubusercontent.com/iieir-km/ComplEx-NNE_AER/master/datasets/DB100K/_test.txt",
        ]
    name = "dbp100k"


class YAGO310(TransductiveDataset):

    urls = [
        "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/datI implemented stratified splits but the performance of my model detriorated from mrr of 0.1 to 0.08 is this possible or thus suggest that my implementation of stratified splits might be wrong?ma/YAGO3-10/train.txt",
        "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/valid.txt",
        "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/test.txt",
        ]
    name = "yago310"


class Hetionet(TransductiveDataset):

    urls = [
        "https://www.dropbox.com/s/y47bt9oq57h6l5k/train.txt?dl=1",
        "https://www.dropbox.com/s/a0pbrx9tz3dgsff/valid.txt?dl=1",
        "https://www.dropbox.com/s/4dhrvg3fyq5tnu4/test.txt?dl=1",
        ]
    name = "hetionet"


class AristoV4(TransductiveDataset):

    url = "https://zenodo.org/record/5942560/files/aristo-v4.zip"

    name = "aristov4"
    delimiter = "\t"

    def download(self):
        download_path = download_url(self.url, self.raw_dir)
        extract_zip(download_path, self.raw_dir)
        os.unlink(download_path)
        for oldname, newname in zip(['train', 'valid', 'test'], self.raw_paths):
            os.rename(os.path.join(self.raw_dir, oldname), newname)


class SparserKG(TransductiveDataset):

    # 5 datasets based on FB/NELL/WD, introduced in https://github.com/THU-KEG/DacKGR
    # re-writing the loading function because dumps are in the format (h, t, r) while the standard is (h, r, t)

    url = "https://raw.githubusercontent.com/THU-KEG/DacKGR/master/data.zip"
    delimiter = "\t"
    base_name = "SparseKG"

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.base_name, self.name, "raw")
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, self.base_name, self.name, "processed")

    def download(self):
        base_path = os.path.join(self.root, self.base_name)
        download_path = download_url(self.url, base_path)
        extract_zip(download_path, base_path)
        for dsname in ['NELL23K', 'WD-singer', 'FB15K-237-10', 'FB15K-237-20', 'FB15K-237-50']:
            for oldname, newname in zip(['train.triples', 'dev.triples', 'test.triples'], self.raw_file_names):
                os.renames(os.path.join(base_path, "data", dsname, oldname), os.path.join(base_path, dsname, "raw", newname))
        shutil.rmtree(os.path.join(base_path, "data"))
    
    def load_file(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}):

        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(triplet_file, "r", encoding="utf-8") as fin:
            for l in fin:
                u, v, r = l.split() if self.delimiter is None else l.strip().split(self.delimiter)
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))

        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab), #entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }
    
class WDsinger(SparserKG):   
    name = "WD-singer"

class NELL23k(SparserKG):   
    name = "NELL23K"

class FB15k237_10(SparserKG):   
    name = "FB15K-237-10"

class FB15k237_20(SparserKG):   
    name = "FB15K-237-20"

class FB15k237_50(SparserKG):   
    name = "FB15K-237-50"


class InductiveDataset(InMemoryDataset):

    delimiter = None
    # some datasets (4 from Hamaguchi et al and Indigo) have validation set based off the train graph, not inference
    valid_on_inf = True  # 
    
    def __init__(self, root, version, transform=None, pre_transform=build_relation_graph, **kwargs):

        self.version = str(version)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self):
        for url, path in zip(self.urls, self.raw_paths):
            download_path = download_url(url % self.version, self.raw_dir)
            os.rename(download_path, path)
    
    def load_file(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}):

        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(triplet_file, "r", encoding="utf-8") as fin:
            for l in fin:
                u, r, v = l.split() if self.delimiter is None else l.strip().split(self.delimiter)
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))

        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab), #entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }
    
    def process(self):
        
        train_files = self.raw_paths[:4]

        train_res = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        inference_res = self.load_file(train_files[1], inv_entity_vocab={}, inv_rel_vocab={})
        valid_res = self.load_file(
            train_files[2], 
            inference_res["inv_entity_vocab"] if self.valid_on_inf else train_res["inv_entity_vocab"], 
            inference_res["inv_rel_vocab"] if self.valid_on_inf else train_res["inv_rel_vocab"]
        )
        test_res = self.load_file(train_files[3], inference_res["inv_entity_vocab"], inference_res["inv_rel_vocab"])

        num_train_nodes, num_train_rels = train_res["num_node"], train_res["num_relation"]
        inference_num_nodes, inference_num_rels = test_res["num_node"], test_res["num_relation"]

        train_edges, inf_graph, inf_valid_edges, inf_test_edges = train_res["triplets"], inference_res["triplets"], valid_res["triplets"], test_res["triplets"]
        
        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_edges], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_edges])

        train_fact_index = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_fact_type = torch.cat([train_target_etypes, train_target_etypes + num_train_rels])

        inf_edges = torch.tensor([[t[0], t[1]] for t in inf_graph], dtype=torch.long).t()
        inf_edges = torch.cat([inf_edges, inf_edges.flip(0)], dim=1)
        inf_etypes = torch.tensor([t[2] for t in inf_graph])
        inf_etypes = torch.cat([inf_etypes, inf_etypes + inference_num_rels])
        
        inf_valid_edges = torch.tensor(inf_valid_edges, dtype=torch.long)
        inf_test_edges = torch.tensor(inf_test_edges, dtype=torch.long)

        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=num_train_nodes,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_train_rels*2)
        valid_data = Data(edge_index=inf_edges if self.valid_on_inf else train_fact_index, 
                          edge_type=inf_etypes if self.valid_on_inf else train_fact_type, 
                          num_nodes=inference_num_nodes if self.valid_on_inf else num_train_nodes,
                          target_edge_index=inf_valid_edges[:, :2].T, 
                          target_edge_type=inf_valid_edges[:, 2], 
                          num_relations=inference_num_rels*2 if self.valid_on_inf else num_train_rels*2)
        test_data = Data(edge_index=inf_edges, edge_type=inf_etypes, num_nodes=inference_num_nodes,
                         target_edge_index=inf_test_edges[:, :2].T, target_edge_type=inf_test_edges[:, 2], num_relations=inference_num_rels*2)

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])
    
    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, self.version, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, self.version, "processed")
    
    @property
    def raw_file_names(self):
        return [
            "transductive_train.txt", "inference_graph.txt", "inf_valid.txt", "inf_test.txt"
        ]

    @property
    def processed_file_names(self):
        return "data.pt"

    def __repr__(self):
        return "%s(%s)" % (self.name, self.version)


class IngramInductive(InductiveDataset):

    @property
    def raw_dir(self):
        return os.path.join(self.root, "ingram", self.name, self.version, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "ingram", self.name, self.version, "processed")
    

class FBIngram(IngramInductive):

    urls = [
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/FB-%s/train.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/FB-%s/msg.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/FB-%s/valid.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/FB-%s/test.txt",
    ]
    name = "fb"


class WKIngram(IngramInductive):

    urls = [
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/WK-%s/train.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/WK-%s/msg.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/WK-%s/valid.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/WK-%s/test.txt",
    ]
    name = "wk"

class NLIngram(IngramInductive):

    urls = [
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/NL-%s/train.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/NL-%s/msg.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/NL-%s/valid.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/NL-%s/test.txt",
    ]
    name = "nl"


class ILPC2022(InductiveDataset):

    urls = [
        "https://raw.githubusercontent.com/pykeen/ilpc2022/master/data/%s/train.txt",
        "https://raw.githubusercontent.com/pykeen/ilpc2022/master/data/%s/inference.txt",
        "https://raw.githubusercontent.com/pykeen/ilpc2022/master/data/%s/inference_validation.txt",
        "https://raw.githubusercontent.com/pykeen/ilpc2022/master/data/%s/inference_test.txt",
    ]

    name = "ilpc2022"
    

class HM(InductiveDataset):
    # benchmarks from Hamaguchi et al and Indigo BM

    urls = [
        "https://raw.githubusercontent.com/shuwen-liu-ox/INDIGO/master/data/%s/train/train.txt",
        "https://raw.githubusercontent.com/shuwen-liu-ox/INDIGO/master/data/%s/test/test-graph.txt",
        "https://raw.githubusercontent.com/shuwen-liu-ox/INDIGO/master/data/%s/train/valid.txt",
        "https://raw.githubusercontent.com/shuwen-liu-ox/INDIGO/master/data/%s/test/test-fact.txt",
    ]

    name = "hm"
    versions = {
        '1k': "Hamaguchi-BM_both-1000",
        '3k': "Hamaguchi-BM_both-3000",
        '5k': "Hamaguchi-BM_both-5000",
        'indigo': "INDIGO-BM" 
    }
    # in 4 HM graphs, the validation set is based off the training graph, so we'll adjust the dataset creation accordingly
    valid_on_inf = False 

    def __init__(self, root, version, **kwargs):
        version = self.versions[version]
        super().__init__(root, version, **kwargs)

    # HM datasets are a bit weird: validation set (based off the train graph) has a few hundred new nodes, so we need a custom processing
    def process(self):
        
        train_files = self.raw_paths[:4]

        train_res = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        inference_res = self.load_file(train_files[1], inv_entity_vocab={}, inv_rel_vocab={})
        valid_res = self.load_file(
            train_files[2], 
            inference_res["inv_entity_vocab"] if self.valid_on_inf else train_res["inv_entity_vocab"], 
            inference_res["inv_rel_vocab"] if self.valid_on_inf else train_res["inv_rel_vocab"]
        )
        test_res = self.load_file(train_files[3], inference_res["inv_entity_vocab"], inference_res["inv_rel_vocab"])

        num_train_nodes, num_train_rels = train_res["num_node"], train_res["num_relation"]
        inference_num_nodes, inference_num_rels = test_res["num_node"], test_res["num_relation"]

        train_edges, inf_graph, inf_valid_edges, inf_test_edges = train_res["triplets"], inference_res["triplets"], valid_res["triplets"], test_res["triplets"]
        
        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_edges], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_edges])

        train_fact_index = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_fact_type = torch.cat([train_target_etypes, train_target_etypes + num_train_rels])

        inf_edges = torch.tensor([[t[0], t[1]] for t in inf_graph], dtype=torch.long).t()
        inf_edges = torch.cat([inf_edges, inf_edges.flip(0)], dim=1)
        inf_etypes = torch.tensor([t[2] for t in inf_graph])
        inf_etypes = torch.cat([inf_etypes, inf_etypes + inference_num_rels])
        
        inf_valid_edges = torch.tensor(inf_valid_edges, dtype=torch.long)
        inf_test_edges = torch.tensor(inf_test_edges, dtype=torch.long)

        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=num_train_nodes,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_train_rels*2)
        valid_data = Data(edge_index=train_fact_index, 
                          edge_type=train_fact_type, 
                          num_nodes=valid_res["num_node"],  # the only fix in this function
                          target_edge_index=inf_valid_edges[:, :2].T, 
                          target_edge_type=inf_valid_edges[:, 2], 
                          num_relations=inference_num_rels*2 if self.valid_on_inf else num_train_rels*2)
        test_data = Data(edge_index=inf_edges, edge_type=inf_etypes, num_nodes=inference_num_nodes,
                         target_edge_index=inf_test_edges[:, :2].T, target_edge_type=inf_test_edges[:, 2], num_relations=inference_num_rels*2)

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])


class MTDEAInductive(InductiveDataset):

    valid_on_inf = False
    url = "https://reltrans.s3.us-east-2.amazonaws.com/MTDEA_data.zip"
    base_name = "mtdea"

    def __init__(self, root, version, **kwargs):

        assert version in self.versions, f"unknown version {version} for {self.name}, available: {self.versions}"
        super().__init__(root, version, **kwargs)

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.base_name, self.name, self.version, "raw")
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, self.base_name, self.name, self.version, "processed")
    
    @property
    def raw_file_names(self):
        return [
            "transductive_train.txt", "inference_graph.txt", "transductive_valid.txt", "inf_test.txt"
        ]

    def download(self):
        base_path = os.path.join(self.root, self.base_name)
        download_path = download_url(self.url, base_path)
        extract_zip(download_path, base_path)
        # unzip all datasets at once
        for dsname in ['FBNELL', 'Metafam', 'WikiTopics-MT1', 'WikiTopics-MT2', 'WikiTopics-MT3', 'WikiTopics-MT4']:
            cl = globals()[dsname.replace("-","")]
            versions = cl.versions
            for version in versions:
                for oldname, newname in zip(['train.txt', 'observe.txt', 'valid.txt', 'test.txt'], self.raw_file_names):
                    foldername = cl.prefix % version + "-trans" if "transductive" in newname else cl.prefix % version + "-ind"
                    os.renames(
                        os.path.join(base_path, "MTDEA_datasets", dsname, foldername, oldname), 
                        os.path.join(base_path, dsname, version, "raw", newname)
                    )
        shutil.rmtree(os.path.join(base_path, "MTDEA_datasets"))

    def load_file(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}, limit_vocab=False):

        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        # limit_vocab is for dropping triples with unseen head/tail not seen in the main entity_vocab
        # can be used for FBNELL and MT3:art, other datasets seem to be ok and share num_nodes/num_relations in the train/inference graph  
        with open(triplet_file, "r", encoding="utf-8") as fin:
            for l in fin:
                u, r, v = l.split() if self.delimiter is None else l.strip().split(self.delimiter)
                if u not in inv_entity_vocab:
                    if limit_vocab:
                        continue
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    if limit_vocab:
                        continue
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    if limit_vocab:
                        continue
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))
        
        return {
            "triplets": triplets,
            "num_node": entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }

    # special processes for MTDEA datasets for one particular fix in the validation set loading
    def process(self):
    
        train_files = self.raw_paths[:4]

        train_res = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        inference_res = self.load_file(train_files[1], inv_entity_vocab={}, inv_rel_vocab={})
        valid_res = self.load_file(
            train_files[2], 
            inference_res["inv_entity_vocab"] if self.valid_on_inf else train_res["inv_entity_vocab"], 
            inference_res["inv_rel_vocab"] if self.valid_on_inf else train_res["inv_rel_vocab"],
            limit_vocab=True,  # the 1st fix in this function compared to the superclass processor
        )
        test_res = self.load_file(train_files[3], inference_res["inv_entity_vocab"], inference_res["inv_rel_vocab"])

        num_train_nodes, num_train_rels = train_res["num_node"], train_res["num_relation"]
        inference_num_nodes, inference_num_rels = test_res["num_node"], test_res["num_relation"]

        train_edges, inf_graph, inf_valid_edges, inf_test_edges = train_res["triplets"], inference_res["triplets"], valid_res["triplets"], test_res["triplets"]
        
        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_edges], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_edges])

        train_fact_index = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_fact_type = torch.cat([train_target_etypes, train_target_etypes + num_train_rels])

        inf_edges = torch.tensor([[t[0], t[1]] for t in inf_graph], dtype=torch.long).t()
        inf_edges = torch.cat([inf_edges, inf_edges.flip(0)], dim=1)
        inf_etypes = torch.tensor([t[2] for t in inf_graph])
        inf_etypes = torch.cat([inf_etypes, inf_etypes + inference_num_rels])
        
        inf_valid_edges = torch.tensor(inf_valid_edges, dtype=torch.long)
        inf_test_edges = torch.tensor(inf_test_edges, dtype=torch.long)

        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=num_train_nodes,
                        target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_train_rels*2)
        valid_data = Data(edge_index=train_fact_index, 
                        edge_type=train_fact_type, 
                        num_nodes=valid_res["num_node"],  # the 2nd fix in this function
                        target_edge_index=inf_valid_edges[:, :2].T, 
                        target_edge_type=inf_valid_edges[:, 2], 
                        num_relations=inference_num_rels*2 if self.valid_on_inf else num_train_rels*2)
        test_data = Data(edge_index=inf_edges, edge_type=inf_etypes, num_nodes=inference_num_nodes,
                        target_edge_index=inf_test_edges[:, :2].T, target_edge_type=inf_test_edges[:, 2], num_relations=inference_num_rels*2)

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])


class FBNELL(MTDEAInductive):

    name = "FBNELL"
    prefix = "%s"
    versions = ["FBNELL_v1"]

    def __init__(self, **kwargs):
        kwargs.pop("version")
        kwargs['version'] = self.versions[0]
        super(FBNELL, self).__init__(**kwargs)


class Metafam(MTDEAInductive):

    name = "Metafam"
    prefix = "%s"
    versions = ["Metafam"]

    def __init__(self, **kwargs):
        kwargs.pop("version")
        kwargs['version'] = self.versions[0]
        super(Metafam, self).__init__(**kwargs)


class WikiTopicsMT1(MTDEAInductive):

    name = "WikiTopics-MT1"
    prefix = "wikidata_%sv1"
    versions = ['mt', 'health', 'tax']

    def __init__(self, **kwargs):
        assert kwargs['version'] in self.versions, f"unknown version {kwargs['version']}, available: {self.versions}"
        super(WikiTopicsMT1, self).__init__(**kwargs)


class WikiTopicsMT2(MTDEAInductive):

    name = "WikiTopics-MT2"
    prefix = "wikidata_%sv1"
    versions = ['mt2', 'org', 'sci']

    def __init__(self, **kwargs):
        super(WikiTopicsMT2, self).__init__(**kwargs)


class WikiTopicsMT3(MTDEAInductive):

    name = "WikiTopics-MT3"
    prefix = "wikidata_%sv2"
    versions = ['mt3', 'art', 'infra']

    def __init__(self, **kwargs):
        super(WikiTopicsMT3, self).__init__(**kwargs)


class WikiTopicsMT4(MTDEAInductive):

    name = "WikiTopics-MT4"
    prefix = "wikidata_%sv2"
    versions = ['mt4', 'sci', 'health']

    def __init__(self, **kwargs):
        super(WikiTopicsMT4, self).__init__(**kwargs)


# a joint dataset for pre-training ULTRA on several graphs
class JointDataset(InMemoryDataset):

    datasets_map = {
        'FB15k237': FB15k237,
        'WN18RR': WN18RR,
        'CoDExSmall': CoDExSmall,
        'CoDExMedium': CoDExMedium,
        'CoDExLarge': CoDExLarge,
        'NELL995': NELL995,
        'ConceptNet100k': ConceptNet100k,
        'DBpedia100k': DBpedia100k,
        'YAGO310': YAGO310,
        'AristoV4': AristoV4,
    }

    def __init__(self, root, graphs, transform=None, pre_transform=None):


        self.graphs = [self.datasets_map[ds](root=root) for ds in graphs]
        self.num_graphs = len(graphs)
        super().__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, "joint", f'{self.num_graphs}g', "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "joint", f'{self.num_graphs}g', "processed")

    @property
    def processed_file_names(self):
        return "data.pt"
    
    def process(self):
        
        train_data = [g[0] for g in self.graphs]
        valid_data = [g[1] for g in self.graphs]
        test_data = [g[2] for g in self.graphs]
        # filter_data = [
        #     Data(edge_index=g.data.target_edge_index, edge_type=g.data.target_edge_type, num_nodes=g[0].num_nodes) for g in self.graphs
        # ]

        torch.save((train_data, valid_data, test_data), self.processed_paths[0])