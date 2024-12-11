import os
import csv
import shutil
import torch
import requests
import pandas as pd
import json
import pickle
import difflib
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.datasets import RelLinkPredDataset, WordNet18RR, MovieLens100K
from collections import defaultdict
from ultra.tasks import build_relation_graph
from ultra import preprocess_data, test_functions
import matplotlib.pyplot as plt
from datetime import datetime
import json
import difflib
import pandas as pd
import json
import difflib

    

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





class GowallaDataset(InMemoryDataset):
    """
    Gowalla Dataset for recommender systems.
    Includes edge features from check-ins and friendship metadata.
    """
    urls = {
        "train": "https://huggingface.co/datasets/reczoo/Gowalla_m1/resolve/main/train.txt",
        "test": "https://huggingface.co/datasets/reczoo/Gowalla_m1/resolve/main/test.txt",
        "edges": "https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz",
        "checkins": "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"
    }

    name = "gowalla"

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        return ["train.txt", "test.txt", "loc-gowalla_edges.txt", "loc-gowalla_totalCheckins.txt"]

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        # Download required files
        for name, url in self.urls.items():
            path = os.path.join(self.raw_dir, os.path.basename(url))
            response = requests.get(url)
            response.raise_for_status()
            with open(path, "wb") as f:
                f.write(response.content)

    def process(self):
        # Parse edges
        train_edges = self.parse_edges(os.path.join(self.raw_dir, "train.txt"))
        test_edges = self.parse_edges(os.path.join(self.raw_dir, "test.txt"))

        # Adjust item IDs to prevent overlap with user IDs
        num_users = train_edges[0].max().item() + 1
        train_edges[1] += num_users
        test_edges[1] += num_users

        # Stratified split for validation
        train_indices, valid_indices = stratified_split(train_edges, [0.9, 0.1], filter_by="item")[:2]
        train_target_edges = train_edges[:, train_indices]
        valid_target_edges = train_edges[:, valid_indices]

        # Combine train edges with reversed edges
        train_edges_combined = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_edge_types = torch.cat(
            [torch.zeros(train_target_edges.size(1), dtype=torch.int64),
             torch.ones(train_target_edges.size(1), dtype=torch.int64)], dim=0
        )

        valid_edge_types = torch.zeros(valid_target_edges.size(1), dtype=torch.int64)
        test_edge_types = torch.zeros(test_edges.size(1), dtype=torch.int64)
        train_target_edge_types = torch.zeros(train_target_edges.size(1), dtype=torch.int64)

        # Load friendship data
        friends = self.load_friendship_edges(os.path.join(self.raw_dir, "loc-gowalla_edges.txt"))

        # Load and process check-in metadata
        checkins = self.load_checkins(os.path.join(self.raw_dir, "loc-gowalla_totalCheckins.txt"), num_users)
        edge_features = self.map_checkins_to_edges(train_edges_combined, checkins)

        # Number of nodes and relations
        num_nodes = num_users + train_edges[1].max().item() + 1
        num_relations = 2

        # Construct Data objects
        train_data = Data(
            edge_index=train_edges_combined, edge_type=train_edge_types, num_nodes=num_nodes,
            target_edge_index=train_target_edges, target_edge_type=train_target_edge_types,
            edge_attr=edge_features, num_relations=num_relations
        )

        valid_data = Data(
            edge_index=train_edges_combined, edge_type=train_edge_types, num_nodes=num_nodes,
            target_edge_index=valid_target_edges, target_edge_type=valid_edge_types,
            edge_attr=edge_features, num_relations=num_relations
        )

        test_data = Data(
            edge_index=train_edges, edge_type=test_edge_types, num_nodes=num_nodes,
            target_edge_index=test_edges, target_edge_type=test_edge_types,
            edge_attr=edge_features, num_relations=num_relations
        )

        # Add metadata
        for data in [train_data, valid_data, test_data]:
            data.num_users = num_users
            data.num_items = train_edges[1].max().item() - num_users + 1
            data.friends = friends

        # Pre-transform if provided
        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        # Save processed data
        torch.save(self.collate([train_data, valid_data, test_data]), self.processed_paths[0])

    @staticmethod
    def parse_edges(file_path):
        """
        Parse edges from a file into a tensor.
        """
        edges = []
        with open(file_path, "r") as f:
            for line in f:
                ids = list(map(int, line.strip().split()))
                user, items = ids[0], ids[1:]
                edges.extend([(user, item) for item in items])
        return torch.tensor(edges, dtype=torch.int64).t()

    @staticmethod
    def load_friendship_edges(file_path):
        """
        Load friendship edges into a dictionary.
        """
        friends = defaultdict(list)
        with open(file_path, "r") as f:
            for line in f:
                user, friend = map(int, line.strip().split())
                friends[user].append(friend)
        return friends

    @staticmethod
    def load_checkins(file_path, num_users):
        """
        Load check-in metadata into a DataFrame.
        """
        checkins = pd.read_csv(
            file_path, sep="\t", header=None,
            names=["user", "time", "latitude", "longitude", "location"]
        )
        checkins["user"] -= 1  # Remap user IDs to 0-indexed
        checkins["time"] = pd.to_datetime(checkins["time"], format="%Y-%m-%dT%H:%M:%SZ")
        checkins["time"] = checkins["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return checkins

    @staticmethod
    def map_checkins_to_edges(edge_index, checkins):
        """
        Map check-ins to edge features based on edge_index.
        """
        edge_features = []
        checkin_dict = checkins.set_index(["user", "location"]).to_dict(orient="index")

        for src, tgt in edge_index.t().tolist():
            if (src, tgt - src) in checkin_dict:  # Match user-location pairs
                edge_features.append(list(checkin_dict[(src, tgt - src)].values()))
            else:
                edge_features.append([0.0] * 4)  # Default for missing values

        return torch.tensor(edge_features, dtype=torch.float32)



class AmazonDataset(InMemoryDataset):
    """
    Generic Amazon Dataset (Fashion, Men, Beauty, Games) for recommender systems.
    Supports ID remapping, stratified splitting, and edge processing.
    """
    
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        return [f"{self.name}.txt"]

    @property
    def processed_file_names(self):
        return "data.pt"

    def process(self):
        """
        Processes the dataset to create train, validation, and test splits, 
        along with edge_index tensors for user-item interactions.
        """
        raw_file_path = os.path.join(self.raw_dir, f"{self.name}.txt")
        train_edges, valid_edges, test_edges, num_users, num_items = self.load_edges(raw_file_path)

        # Adjust item IDs to prevent overlap with user IDs
        train_edges[1] += num_users
        valid_edges[1] += num_users
        test_edges[1] += num_users

        # Add reversed edges
        reversed_train_edges = train_edges.flip(0)

        # Combine edges and set edge types
        train_edges_combined = torch.cat([train_edges, reversed_train_edges], dim=1)


        train_edge_types = torch.cat(
            [torch.zeros(train_edges.size(1), dtype=torch.int64), 
             torch.ones(reversed_train_edges.size(1), dtype=torch.int64)], dim=0
        )

        test_edge_types = torch.zeros(test_edges.size(1), dtype=torch.int64)
        valid_edge_types = torch.zeros(valid_edges.size(1), dtype=torch.int64)
        target_train_edges = train_edges
        target_train_edge_types = torch.zeros(train_edges.size(1), dtype=torch.int64)

        # Number of nodes and relations
        num_nodes = num_users + num_items
        num_relations = 2  # Normal and reversed edges

        # Construct Data objects
        train_data = Data(
            edge_index=train_edges_combined, edge_type=train_edge_types, num_nodes=num_nodes,
            target_edge_index=target_train_edges, target_edge_type=target_train_edge_types,
            num_relations=num_relations
        )

        valid_data = Data(
            edge_index=train_edges_combined, edge_type=train_edge_types, num_nodes=num_nodes,
            target_edge_index=valid_edges, target_edge_type=valid_edge_types,
            num_relations=num_relations
        )

        test_data = Data(
            edge_index=train_edges_combined, edge_type=train_edge_types, num_nodes=num_nodes,
            target_edge_index=test_edges, target_edge_type=test_edge_types,
            num_relations=num_relations
        )

        print (f"num_users: {num_users} ")
        print (f"num_items: {num_items} ")
        
         # Load item features
        item_features = self.load_item_features(num_items)

        # Create edge features (context)
        edge_features = self.create_edge_features(train_edges, num_users)

        # Create generic user features
        user_features = torch.ones((num_users, 1), dtype=torch.float32)

        print (f"user_features.shape: {user_features.shape} ")
        print (f"item_features.shape: {item_features.shape} ")
        print (f"edge_features.shape: {edge_features.shape} ")
        # Add metadata
        for data in [train_data, valid_data, test_data]:
            data.num_users = num_users
            data.num_items = num_items
            data.x_user = user_features
            data.x_item = item_features
            data.x_edges = edge_features

        # Pre-transform if provided
        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        # Save processed data
        torch.save(self.collate([train_data, valid_data, test_data]), self.processed_paths[0])
        print("yeiy it worked")

    def load_edges(self, file_path):
        """
        Load edges from the dataset file and perform splits into train, validation, and test sets.

        Args:
            file_path (str): Path to the dataset file.
        
        Returns:
            Tuple[Tensor, Tensor, Tensor, int, int]: Train, validation, test edges, and user/item counts.
        """
        user_data = defaultdict(list)
        num_users, num_items = 0, 0

        # Read interactions from file
        with open(file_path, "r") as f:
            for line in f:
                user, item = map(int, line.strip().split())
                user, item = user - 1, item - 1  # Adjust IDs to start from 0
                user_data[user].append(item)
                num_users = max(num_users, user + 1)
                num_items = max(num_items, item + 1)

        # Create train, validation, and test splits
        train_edges, valid_edges, test_edges = [], [], []
        for user, items in user_data.items():
            if len(items) < 3:
                train_edges.extend([(user, item) for item in items])
            else:
                train_edges.extend([(user, item) for item in items[:-2]])
                valid_edges.append((user, items[-2]))
                test_edges.append((user, items[-1]))

        # Convert to PyTorch tensors
        train_edges = torch.tensor(train_edges, dtype=torch.int64).t()
        valid_edges = torch.tensor(valid_edges, dtype=torch.int64).t()
        test_edges = torch.tensor(test_edges, dtype=torch.int64).t()

        return train_edges, valid_edges, test_edges, num_users, num_items
        
    def load_data(self,filename):
        try:
            with open(filename, "rb") as f:
                x= pickle.load(f)
        except:
            x = []
        return x    
    
    def load_item_features(self, num_items):
        """
        Load item features from the corresponding file.
        If no features are available, initialize zeros.
        """
        feature_file = os.path.join(self.raw_dir, f"{self.name}_feat_cat.dat")
        try:
            item_features = self.load_data(feature_file)  # Assuming load_data returns a NumPy array
            item_features = torch.tensor(item_features, dtype=torch.float32)
        except Exception as e:
            print(f"Failed to load item features: {e}")
            item_features = torch.zeros((num_items, 1), dtype=torch.float32)  # Default to 1D zeros
        return item_features

    def create_edge_features(self, edge_index, num_users):
        """
        Create edge features from the context dictionary.
        """
        ctxt_file = os.path.join(self.raw_dir, f"{self.name}_ctxt.dat")
        cxtdict = self.load_data(ctxt_file)
        edge_features = []
        missing = 0
        for src, tgt in edge_index.t().tolist():  # Iterate over edges
            src, tgt = int(src), int(tgt)
            src += 1
            tgt = (tgt + 1) - num_users
            if (src, tgt) in cxtdict:
                edge_features.append(cxtdict[(src, tgt)])
            else:
                missing += 1
                edge_features.append([0.0] * 6)  # Default zero vector for missing context

        print (f"missing: {missing}")
        return torch.tensor(edge_features, dtype=torch.float32)



class Amazon_Beauty(AmazonDataset):
    name = "amazon_beauty"
    
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, self.name, transform, pre_transform)
         
class Amazon_Fashion(AmazonDataset):
    name = "amazon_fashion"
    
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, self.name, transform, pre_transform)

class Amazon_Men(AmazonDataset):
    name = "amazon_men"
    
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, self.name, transform, pre_transform)
        
# Subclass for Amazon Beauty Dataset
class Amazon_Games(AmazonDataset):
    name = "amazon_games"
    
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, self.name, transform, pre_transform)


class Yelp18(InMemoryDataset):
    """
    Yelp18 Dataset for recommender systems with ID remapping and stratified splitting.
    """

    urls = {
        "train": "https://huggingface.co/datasets/reczoo/Yelp18_m1/raw/main/train.txt",
        "test": "https://huggingface.co/datasets/reczoo/Yelp18_m1/raw/main/test.txt",
        "user_list": "https://huggingface.co/datasets/reczoo/Yelp18_m1/resolve/main/user_list.txt",
        "item_list": "https://huggingface.co/datasets/reczoo/Yelp18_m1/resolve/main/item_list.txt",
    }
    name = "yelp18"
    
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        return ["train.txt", "test.txt", "user_list.txt", "item_list.txt"]

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        # Download required files
        for name, url in self.urls.items():
            path = os.path.join(self.raw_dir, f"{name}.txt")
            response = requests.get(url)
            response.raise_for_status()
            with open(path, "wb") as f:
                f.write(response.content)

    def load_mappings(self):
        """Load user and item mappings."""
        user_map = {}
        item_map = {}
    
        # Read user mappings
        user_path = os.path.join(self.raw_dir, "user_list.txt")
        with open(user_path, "r") as f:
            next(f)  # Skip the header line
            for line in f:
                original_id, remap_id = line.strip().split()
                user_map[original_id] = int(remap_id)
    
        # Read item mappings
        num_users = len(user_map)
        item_path = os.path.join(self.raw_dir, "item_list.txt")
        with open(item_path, "r") as f:
            next(f)  # Skip the header line
            for line in f:
                original_id, remap_id = line.strip().split()
                item_map[original_id] = int(remap_id)
    
        return user_map, item_map
    
    def load_user_features(self, user_json_path, user_map):
        
        """
        Parse and preprocess user features with correct data types for all attributes.
        Ensure every user has an entry in the DataFrame.
        """
        # Initialize the user features dictionary with NaN for unmatched users
        default_user_features = {
            "review_count": float("nan"),
            "average_stars": float("nan"),
            "yelping_since": float("nan"),
            "friends": float("nan"),
            "useful": float("nan"),
            "funny": float("nan"),
            "cool": float("nan"),
            "fans": float("nan"),
            "elite": float("nan"),
            "compliment_hot": float("nan"),
            "compliment_more": float("nan"),
            "compliment_profile": float("nan"),
            "compliment_cute": float("nan"),
            "compliment_list": float("nan"),
            "compliment_note": float("nan"),
            "compliment_plain": float("nan"),
            "compliment_cool": float("nan"),
            "compliment_funny": float("nan"),
            "compliment_writer": float("nan"),
            "compliment_photos": float("nan"),
        }
        user_features = {remapped_id: default_user_features.copy() for remapped_id in range(len(user_map))}
    
        # Parse the JSON file
        with open(user_json_path, "r") as f:
            for line in f:
                user = json.loads(line)
                if user["user_id"] in user_map:
                    user_features[user_map[user["user_id"]]] = {
                        "review_count": int(user["review_count"]),
                        "average_stars": float(user["average_stars"]),
                        "yelping_since": str(user["yelping_since"]),
                        "friends": len(user["friends"]),  # Number of friends
                        "useful": int(user["useful"]),
                        "funny": int(user["funny"]),
                        "cool": int(user["cool"]),
                        "fans": int(user["fans"]),
                        "elite": len(user["elite"]),
                        "compliment_hot": int(user["compliment_hot"]),
                        "compliment_more": int(user["compliment_more"]),
                        "compliment_profile": int(user["compliment_profile"]),
                        "compliment_cute": int(user["compliment_cute"]),
                        "compliment_list": int(user["compliment_list"]),
                        "compliment_note": int(user["compliment_note"]),
                        "compliment_plain": int(user["compliment_plain"]),
                        "compliment_cool": int(user["compliment_cool"]),
                        "compliment_funny": int(user["compliment_funny"]),
                        "compliment_writer": int(user["compliment_writer"]),
                        "compliment_photos": int(user["compliment_photos"]),
                    }
    
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(user_features, orient="index")
        df.index.name = "id"
    
        # Sort DataFrame by ID
        df.sort_index(inplace=True)
    
        # Define meta_info
        meta_info = preprocess_data.get_meta_info()
        meta_info["numerical_cols"] = [
            "review_count", "average_stars", "friends", "useful", "funny", "cool", "fans", "elite",
            "compliment_hot", "compliment_more", "compliment_profile", "compliment_cute", "compliment_list",
            "compliment_note", "compliment_plain", "compliment_cool", "compliment_funny",
            "compliment_writer", "compliment_photos"
        ]
        meta_info["date_cols"] = ["yelping_since"]
    
        return df, meta_info
    
        
    
    def flatten_attributes(self, attributes):
        """Flatten nested business attributes for simpler handling."""
        if attributes is None:
            return attributes
        flattened = {}
        for key, value in attributes.items():
            if isinstance(value, dict):  # Handle nested dictionaries
                for sub_key, sub_value in value.items():
                    flattened[f"{key}_{sub_key}"] = sub_value
            else:
                flattened[key] = value
        return flattened
        
    def load_item_features(self, item_json_path, item_map):
        """
        Parse and preprocess item features with correct data types.
        Ensure every item has an entry in the DataFrame.
        """
        default_item_features = {
            "name": None,
            "address": None,
            "city": None,
            "state": None,
            "postal_code": None,
            "latitude": float("nan"),
            "longitude": float("nan"),
            "stars": float("nan"),
            "review_count": float("nan"),
            "is_open": float("nan"),
            "categories": None,
            "attributes": None,
            "hours": None,
        }
        item_features = {remapped_id: default_item_features.copy() for remapped_id in range(len(item_map))}
    
        # Parse the JSON file
        with open(item_json_path, "r") as f:
            for line in f:
                business = json.loads(line)
                if business["business_id"] in item_map:
                    # Extract and clean attributes
                    attributes = business.get("attributes", {})
                    attributes_cleaned = self.flatten_attributes(attributes)
    
                    # Process categories
                    categories = business.get("categories", "")
                    if isinstance(categories, str):
                        categories = [cat.strip() for cat in categories.split(",")]
    
                    # Add item features
                    item_features[item_map[business["business_id"]]] = {
                        "name": str(business["name"]),
                        "address": str(business["address"]),
                        "city": str(business["city"]),
                        "state": str(business["state"]),
                        "postal_code": str(business["postal_code"]),
                        "latitude": float(business["latitude"]),
                        "longitude": float(business["longitude"]),
                        "stars": float(business["stars"]),
                        "review_count": int(business["review_count"]),
                        "is_open": int(business["is_open"]),
                        "categories": categories,
                        "attributes": attributes_cleaned,
                        "hours": business.get("hours", {}),
                    }
    
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(item_features, orient="index")
        df.index.name = "id"
    
        # Sort DataFrame by ID
        df.sort_index(inplace=True)
    
        # Define meta_info
        meta_info = preprocess_data.get_meta_info()
        meta_info["numerical_cols"] = ["latitude", "longitude", "stars", "review_count", "is_open"]
        meta_info["categorical_cols"] = ["state", "postal_code"]
        #meta_info["str_cols"] = ["name", "city", "address"]
        meta_info["ls_of_cat_string"] = ["categories"]
        meta_info["drop_cols"] = ["name", "city", "address", "hours", "attributes"]
        return df, meta_info



    
  


    def process(self):
        # Load mappings please note that item_map doesnt map to the offset (+ num_numser)
        user_map, item_map = self.load_mappings()
        num_users = len(user_map)
        num_items = len(item_map)

        print(f"num_users: {num_users}")
        print(f"num_items: {num_items}")
        
        # integrating user features
        user_features_path = os.path.join(self.raw_dir, "yelp_academic_dataset_user.json")
        user_features_df_tup = self.load_user_features(user_features_path, user_map)
        user_features = preprocess_data.process_df(user_features_df_tup)

        # integrating item features
        item_features_path = os.path.join(self.raw_dir, "yelp_academic_dataset_business.json")
        item_features_df_tup = self.load_item_features(item_features_path, item_map)
        item_features = preprocess_data.process_df(item_features_df_tup)
        
        #raise ValueError("feature preprocessing sucessful")   
        # DEBUG
        #raw_user_ids = extract_user_ids_from_json(item_features_path)
        #find_mismatched_ids(list(item_map.keys()), raw_user_ids, max_output=20)
        #raise ValueError("asdf")
        
        
        


        def parse_edges(file_path):
            """Parse edges from file into a list of tuples."""
            edges = set()  # Avoid duplicate edges
            with open(file_path, "r") as f:
                for line in f:
                    nodes = list(map(int, line.strip().split()))
                    source, targets = nodes[0], nodes[1:]
                    for target in targets:
                        edges.add((source, target))
            return list(edges)

        # Parse train and test edges
        train_edges = parse_edges(os.path.join(self.raw_dir, "train.txt"))
        test_edges = parse_edges(os.path.join(self.raw_dir, "test.txt"))


        # Convert edges to tensors (ensure type is int64)
        train_edge_index = torch.tensor(train_edges, dtype=torch.int64).t()
        test_edge_index = torch.tensor(test_edges, dtype=torch.int64).t()


        # Adjust item IDs to prevent overlap with user IDs
        train_edge_index[1] += num_users
        test_edge_index[1] += num_users

        # Perform stratified split for validation
        train_indices, valid_indices = preprocess_data.stratified_split(train_edge_index, [0.9, 0.1], filter_by="item")[:2]
        train_target_edges = train_edge_index[:, train_indices]
        valid_target_edges = train_edge_index[:, valid_indices]

        # Combine train edges with reversed edges
        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_edge_types = torch.cat(
            [torch.zeros(train_target_edges.size(1), dtype=torch.int64), 
             torch.ones(train_target_edges.size(1), dtype=torch.int64)], dim=0
        )

        test_edge_types = torch.zeros(test_edge_index.size(1), dtype=torch.int64)
        valid_edge_types = torch.zeros(valid_target_edges.size(1), dtype=torch.int64)
        train_target_edge_types = torch.zeros(train_target_edges.size(1), dtype=torch.int64)


        # Number of nodes and relations
        num_nodes = num_users + num_items
        num_relations = 2

        # Construct Data objects
        train_data = Data(
            edge_index=train_edges, edge_type=train_edge_types, num_nodes=num_nodes,
            target_edge_index=train_target_edges, target_edge_type=train_target_edge_types,
            num_relations=num_relations
        )

        valid_data = Data(
            edge_index=train_edges, edge_type=train_edge_types, num_nodes=num_nodes,
            target_edge_index=valid_target_edges, target_edge_type=valid_edge_types,
            num_relations=num_relations
        )

        test_data = Data(
            edge_index=train_edges, edge_type=train_edge_types, num_nodes=num_nodes,
            target_edge_index=test_edge_index, target_edge_type=test_edge_types,
            num_relations=num_relations
        )

        # Add metadata
        for data in [train_data, valid_data, test_data]:
            data.num_users = num_users
            data.num_items = num_items
            data.x_user = user_features
            data.x_item = item_features


        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        #test_functions.validate_graph(train_data, num_users, num_items)
        #test_functions.validate_graph(valid_data, num_users, num_items)
        #test_functions.validate_graph(test_data, num_users, num_items)

        # Save processed data
        torch.save(self.collate([train_data, valid_data, test_data]), self.processed_paths[0])
        print("yeiy it worked")


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
    train_idx, valid_idx, test_idx = preprocess_data.stratified_split(edge_index, split_ratios, filter_by='item')
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