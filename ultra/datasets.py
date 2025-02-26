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
import random
import difflib
import re

    

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





class BaseRecDataset(InMemoryDataset):
    """
    Base class for recommender system datasets that share a common structure.
    Implements shared operations:
      - Loading a CSV file with a "raw_with_splits" naming convention.
      - Adjusting item IDs to start from num_users.
      - Splitting the data into train/test (using the "split" column) and further
        splitting train into train/validation sets using a stratified split.
      - Processing edge features using preprocess_data.process_df.
      - Building an undirected graph for all datasets (using train edges).
      - Adding generic user/item features.
      
    Subclasses should override:
      - custom_preprocessing(self, df): to handle dataset‐specific modifications.
      - get_meta_info(self): to set the correct meta_info dictionary.
      - (optionally) train_split_ratio() and valid_split_ratio() for different splits.
      - raw_file_names (if the raw file name differs).
    """
    
    def __init__(self, root, transform=None, pre_transform=None):
        # Expect that the subclass sets self.dataset_name (e.g., "ml-1m", "epinions", etc.)
        assert hasattr(self, "dataset_name"), "Subclasses must define self.dataset_name"
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_dir(self):
        return os.path.join(self.root, self.dataset_name, "raw")
        
    @property
    def processed_dir(self):
        return os.path.join(self.root, self.dataset_name, "processed")
        
    @property
    def raw_file_names(self):
        # Assumes the raw file is named like "<dataset_name>_raw_with_splits.csv"
        return [f"{self.dataset_name}_raw_with_splits.csv"]
    
    @property
    def processed_file_names(self):
        return "data.pt"
    
    def process(self):
        # --- 1. Load and (optionally) preprocess the CSV file ---
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        df = pd.read_csv(raw_path)
        df = self.custom_preprocessing(df)  # dataset-specific processing
        
        # --- 2. Adjust item IDs ---
        # Compute the number of users (assumes user IDs start at 0)
        num_users = df['user'].max() + 1
        # At this point, each dataset is expected to have a column named "item"
        # (subclasses should rename "items" to "item" if necessary)
        df['item'] = df['item'] + num_users  # shift item IDs so they do not overlap with user IDs
        num_items = df['item'].max() - num_users + 1
        
        # --- 3. Split data into train and test sets ---
        train_df = df[df['split'] == 'train'].drop(columns=['split'])
        test_df  = df[df['split'] == 'test'].drop(columns=['split'])
        
        # --- 4. Process edge features using meta_info ---
        meta_info = self.get_meta_info()
        
        # Concatenate train and test dataframes to ensure consistent processing
        combined_df = pd.concat([train_df, test_df], axis=0)
        
        # Process the combined DataFrame once.
        combined_edges_features = preprocess_data.process_df((combined_df, meta_info))
        
        # Split the processed features back into train and test parts.
        train_edges_features = combined_edges_features[:len(train_df)]
        test_edges_features  = combined_edges_features[len(train_df):]
        
        # --- 5. Create edge index (user-item pairs) ---
        train_edges = torch.tensor(train_df[['user', 'item']].values.T, dtype=torch.long)
        test_edges  = torch.tensor(test_df[['user', 'item']].values.T, dtype=torch.long)
        
        # --- 6. Split train further into train and validation sets ---
        train_indices, valid_indices = preprocess_data.stratified_split(
            train_edges, [self.train_split_ratio(), self.valid_split_ratio()], filter_by="item"
        )[:2]
        train_target_edges = train_edges[:, train_indices]
        valid_target_edges = train_edges[:, valid_indices]
        train_target_edges_features = train_edges_features[train_indices, :]
        valid_target_edges_features = train_edges_features[valid_indices, :]
        
        # --- 7. Build undirected edge_index from the training target edges ---
        # Create a symmetric graph by concatenating the original and flipped edges.
        train_edges_combined = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_edge_types = torch.cat([
            torch.zeros(train_target_edges.size(1), dtype=torch.int64),
            torch.ones(train_target_edges.size(1), dtype=torch.int64)
        ], dim=0)
        train_edges_combined_features = torch.cat([train_target_edges_features,
                                                     train_target_edges_features], dim=0)
        
        # --- 8. Create edge types for valid and test sets ---
        valid_edge_types = torch.zeros(valid_target_edges.size(1), dtype=torch.int64)
        test_edge_types  = torch.zeros(test_edges.size(1), dtype=torch.int64)
        train_target_edge_types = torch.zeros(train_target_edges.size(1), dtype=torch.int64)
        
        # --- 9. Build Data objects for train, valid, and test ---
        num_nodes = num_users + num_items
        num_relations = 2  # for the two types in train_edges_combined
        
        train_data = Data(
            edge_index=train_edges_combined,
            edge_type=train_edge_types,
            edge_attr=train_edges_combined_features,
            num_nodes=num_nodes,
            target_edge_index=train_target_edges,
            target_edge_type=train_target_edge_types,
            target_edge_attr=train_target_edges_features,
            num_relations=num_relations
        )
        
        valid_data = Data(
            edge_index=train_edges_combined,
            edge_type=train_edge_types,
            edge_attr=train_edges_combined_features,
            num_nodes=num_nodes,
            target_edge_index=valid_target_edges,
            target_edge_type=valid_edge_types,
            target_edge_attr=valid_target_edges_features,
            num_relations=num_relations
        )
        
        test_data = Data(
            edge_index=train_edges_combined,
            edge_type=train_edge_types,
            edge_attr=train_edges_combined_features,
            num_nodes=num_nodes,
            target_edge_index=test_edges,
            target_edge_type=test_edge_types,
            target_edge_attr=test_edges_features,
            num_relations=num_relations
        )
        
        # --- 10. Create generic node features (e.g., all ones) ---
        user_features = torch.ones((num_users, 1), dtype=torch.float32)
        item_features = torch.ones((num_items, 1), dtype=torch.float32)
        for d in [train_data, valid_data, test_data]:
            d.num_users = num_users
            d.num_items = num_items
            d.x_user = user_features
            d.x_item = item_features
            
        # --- 11. Optionally pre-transform ---
        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data  = self.pre_transform(test_data)
            
        # --- 12. (Optional) Test the constructed graph ---
        test_functions.test_pyG_graph([train_data, valid_data, test_data])
        
        # --- 13. Save the processed data ---
        torch.save(self.collate([train_data, valid_data, test_data]), self.processed_paths[0])
        
    def custom_preprocessing(self, df):
        """
        Override this method in subclasses if custom DataFrame processing is needed.
        For example, renaming columns, dropping unnecessary columns, or adjusting ratings.
        By default, we rename "items" to "item" if present.
        """
        if "items" in df.columns:
            df = df.rename(columns={"items": "item"})
        return df
    
    def get_meta_info(self):
        """
        Override this method in subclasses to provide the appropriate meta_info.
        By default, we assume the only numerical column is "rating", and that we drop "user" and "item".
        """
        meta_info = preprocess_data.get_meta_info()
        meta_info["numerical_cols"] = ["rating"]
        meta_info["drop_cols"] = ["user", "item"]
        return meta_info
    
    def train_split_ratio(self):
        """Return the ratio for training data (default 0.8)."""
        return 0.8
    
    def valid_split_ratio(self):
        """Return the ratio for validation data (default 0.2)."""
        return 0.2


# ============================================================================
# Subclass for the Book-x dataset
# ============================================================================
class BookX(BaseRecDataset):
    dataset_name = "book-x"
    
    def custom_preprocessing(self, df):
        # Rename columns if necessary
        df = super().custom_preprocessing(df)
        # Drop the timestamp column since it doesn't contain any information
        if "timestamp" in df.columns:
            df = df.drop(columns=["timestamp"])
        # Adjust rating:
        # Ratings are between 0 and 10, with 0 as a default.
        # Replace all 0 ratings with the average of the non-zero ratings.
        nonzero_ratings = df.loc[df["rating"] != 0, "rating"]
        avg_rating = nonzero_ratings.mean() if not nonzero_ratings.empty else 0.0
        df.loc[df["rating"] == 0, "rating"] = avg_rating
        return df


# ============================================================================
# Subclass for the ml-1m dataset
# ============================================================================
class Ml1m(BaseRecDataset):
    dataset_name = "ml-1m"
    
    def custom_preprocessing(self, df):
        # For ml-1m, we just rename "items" to "item" if needed.
        df = super().custom_preprocessing(df)
        # If you wish to keep the timestamp, do nothing further.
        return df
        
    def get_meta_info(self):
        # In epinions, we process both "rating" and "helpfulness" as numerical,
        # and "category" as a categorical variable.
        meta_info = preprocess_data.get_meta_info()
        meta_info["numerical_cols"] = ["rating", "timestamp"]
        meta_info["drop_cols"] = ["user", "item"]
        return meta_info
  


# ============================================================================
# Subclass for the Epinions dataset
# ============================================================================

# Problems with categorical
class Epinions(BaseRecDataset):
    dataset_name = "epinions"
    
    def custom_preprocessing(self, df):
        # Rename "items" to "item" if needed.
        df = super().custom_preprocessing(df)
        # In epinions, you might wish to keep timestamp.
        # Ensure the 'category' column is of type string since its categorical
        if 'category' in df.columns:
            df['category'] = df['category'].astype(str)
        # and "helpfulness" (numerical). No additional changes here.
        return df

    def get_meta_info(self):
        # In epinions, we process both "rating" and "helpfulness" as numerical,
        # and "category" as a categorical variable.
        meta_info = preprocess_data.get_meta_info()
        meta_info["numerical_cols"] = ["rating", "helpfulness"]
        meta_info["categorical_cols"] = ["category"]
        meta_info["drop_cols"] = ["user", "item"]
        return meta_info

# ============================================================================
# Subclass for the LastFM dataset
# ============================================================================
class LastFM(BaseRecDataset):
    dataset_name = "lastfm"
    
    def custom_preprocessing(self, df):
        # Rename "items" to "item" if needed.
        df = super().custom_preprocessing(df)
         # Drop the timestamp column since it doesn't contain any information
        if "timestamp" in df.columns:
            df = df.drop(columns=["timestamp"])
        return df

  

class Gowalla(InMemoryDataset):
    """
    Gowalla Dataset for recommender systems.
    Includes edge features from check-ins and metadata.
    """
    urls = {
        "train": "https://huggingface.co/datasets/reczoo/Gowalla_m1/raw/main/train.txt",
        "test": "https://huggingface.co/datasets/reczoo/Gowalla_m1/raw/main/test.txt",
        "edges": "https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz",
        "checkins": "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz",
        "user_list": "https://huggingface.co/datasets/reczoo/Gowalla_m1/raw/main/user_list.txt",
        "item_list": "https://huggingface.co/datasets/reczoo/Gowalla_m1/raw/main/item_list.txt",
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
        print (f"size of train: train_edges: {train_edges.size(1)}")
        print (f"size of test: test_edges: {test_edges.size(1)}")

        # Adjust item IDs to prevent overlap with user IDs
        num_users = train_edges[0].max().item() + 1
        train_edges[1] += num_users
        test_edges[1] += num_users
        
        # Load mappings
        user_map = self.load_mapping(os.path.join(self.raw_dir, "user_list.txt"))
        item_map = self.load_mapping(os.path.join(self.raw_dir, "item_list.txt"))
        
        # Load and process check-in metadata
        checkin_dict = self.load_checkins(os.path.join(self.raw_dir, "loc-gowalla_totalCheckins.txt"), user_map, item_map, num_users)

        # Allign checkins with edges
        train_edges_features_df = self.map_checkins_to_edges(train_edges, checkin_dict)
        test_edges_features_df = self.map_checkins_to_edges(test_edges, checkin_dict)
        
        print (f"size of train_edges_features_df: {train_edges_features_df.shape}")
        print (f"size of test_edges_features_df: {test_edges_features_df.shape}")
        
        # debug: check that the edges allign
        # test_functions.test_edge_feature_alignment(train_edges, train_edges_features_df)
        # test_functions.test_edge_feature_alignment(test_edges, test_edges_features_df)
        

        # process the edge_feature df's by using preprocess_data.process_df with appropriate meta_info
        meta_info = preprocess_data.get_meta_info()
        meta_info["numerical_cols"] = ["latitude", "longitude"]
        meta_info["date_cols"] = ["date"]
        meta_info["drop_cols"] = ["item", "user"]
        train_edges_features=  preprocess_data.process_df((train_edges_features_df,meta_info))
        test_edges_features=  preprocess_data.process_df((test_edges_features_df,meta_info))
        #raise ValueError("feature preprocessing sucessful")  

        # Stratified split for validation
        train_indices, valid_indices = preprocess_data.stratified_split(train_edges, [0.9, 0.1], filter_by="item")[:2]
        train_target_edges = train_edges[:, train_indices]
        valid_target_edges = train_edges[:, valid_indices]
        train_target_edges_features = train_edges_features[train_indices ,:]
        valid_target_edges_features = train_edges_features[valid_indices ,:]

        # Combine train edges with reversed edges
        train_edges_combined = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_edge_types = torch.cat(
            [torch.zeros(train_target_edges.size(1), dtype=torch.int64),
             torch.ones(train_target_edges.size(1), dtype=torch.int64)], dim=0
        )
        train_edges_combined_features = torch.cat([train_target_edges_features, train_target_edges_features], dim=0)

        valid_edge_types = torch.zeros(valid_target_edges.size(1), dtype=torch.int64)
        test_edge_types = torch.zeros(test_edges.size(1), dtype=torch.int64)
        train_target_edge_types = torch.zeros(train_target_edges.size(1), dtype=torch.int64)

        # Load friendship data
        #friends = self.load_friendship_edges(os.path.join(self.raw_dir, "loc-gowalla_edges.txt"))
        
        

        # Number of nodes and relations
        num_nodes = num_users + train_edges[1].max().item() + 1
        num_relations = 2
        print (f"size of train_edges_combined: {train_edges_combined.size(1)}")
        print (f"size of train_edges_combined_features: {train_edges_combined_features.size(0)}")
        
        print (f"size of train_target_edges: {train_target_edges.size(1)}")
        print (f"size of train_target_edges_features: {train_target_edges_features.size(0)}")

        print (f"size of valid_target_edges: {valid_target_edges.size(1)}")
        print (f"size of valid_target_edges_features: {valid_target_edges_features.size(0)}")
        
        print (f"size of test_edges: {test_edges.size(1)}")
        print (f"size of test_edges_features: {test_edges_features.size(0)}")

        # Construct Data objects
        train_data = Data(
            edge_index=train_edges_combined, edge_type=train_edge_types, edge_attr = train_edges_combined_features,
            num_nodes=num_nodes,
            target_edge_index=train_target_edges, target_edge_type=train_target_edge_types, target_edge_attr = train_target_edges_features,
            num_relations=num_relations
        )

        valid_data = Data(
            edge_index=train_edges_combined, edge_type=train_edge_types, edge_attr = train_edges_combined_features,
            num_nodes=num_nodes,
            target_edge_index=valid_target_edges, target_edge_type=valid_edge_types, target_edge_attr = valid_target_edges_features,
            num_relations=num_relations
        )

        test_data = Data(
            edge_index=train_edges_combined, edge_type=train_edge_types, edge_attr = train_edges_combined_features, 
            num_nodes=num_nodes, 
            target_edge_index=test_edges, target_edge_type=test_edge_types, target_edge_attr = test_edges_features,
            num_relations=num_relations
        )
         
        
        # Add metadata
        # Create generic user/item features
        num_items = num_nodes - num_users
        user_features = torch.ones((num_users, 1), dtype=torch.float32)
        item_features = torch.ones((num_items, 1), dtype=torch.float32)
        
        for data in [train_data, valid_data, test_data]:
            data.num_users = num_users
            data.num_items = num_items
            data.x_user = user_features
            data.x_item = item_features
           # data.friends = friends
        
        #raise ValueError("feature preprocessing sucessful")  
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

   # @staticmethod
   # def load_friendship_edges(file_path):
    #    """
     #   Load friendship edges into a dictionary.
      #  """
       # friends = defaultdict(list)
        #with open(file_path, "r") as f:
         #   for line in f:
          #      user, friend = map(int, line.strip().split())
           #     friends[user].append(friend)
        #return friends
    
    @staticmethod
    def load_mapping(file_path):
        """
        Load ID mappings from a file.
        """
        mapping = {}
        with open(file_path, "r") as f:
            next(f)  # Skip header
            for line in f:
                original, remapped = map(int, line.strip().split())
                mapping[original] = remapped
        return mapping
        
    @staticmethod
    def load_checkins(file_path, user_map, item_map, num_users):
        """
        Load and remap check-in metadata into a dictionary for quick lookup.
        """
        checkins = pd.read_csv(
            file_path, sep="\t", header=None,
            names=["user", "date", "latitude", "longitude", "item"]
        )
        # Remap IDs
        checkins["user"] = checkins["user"].map(user_map)
        # dont know if this is necessary
        checkins["item"] = checkins["item"].map(item_map) + num_users
        # Drop rows with unmapped IDs
        checkins = checkins.dropna(subset=["user", "item"]).reset_index(drop=True)
        # Format date
        checkins["date"] = pd.to_datetime(checkins["date"], format="%Y-%m-%dT%H:%M:%SZ")
        checkins["date"] = checkins["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Deduplicate check-ins: Group by user and item, taking the first occurrence
        checkins_grouped = checkins.groupby(["user", "item"]).first().reset_index()
    
        # Create a lookup dictionary for check-ins
        checkin_dict = {
            (row["user"], row["item"]): [row["date"], row["latitude"], row["longitude"], row["item"]]
            for _, row in checkins_grouped.iterrows()
        }
        return checkin_dict

    
    @staticmethod
    def map_checkins_to_edges(edge_index, checkin_dict):
        """
        Align check-in metadata with edges
        
        Args:
            edge_index (Tensor): Edge index tensor of shape [2, num_edges].
            checkins (DataFrame): Check-in metadata dictionary.
    
        Returns:
            DataFrame: df aligned with edge_index.
            
        """
    
        unmatched_count = 0  # Count unmatched edges
        matched_count = 0    # Count matched edges
        users  = []
    
        # Initialize lists for train and test features
        edge_features = []
    
        # Align features for each edge
        for edge in edge_index.t().tolist():
            user, item = edge
            users.append(user)
            # Handle both directions
            if (user, item) in checkin_dict:
                edge_features.append(checkin_dict[(user, item)])
                matched_count += 1
            elif (item, user) in checkin_dict:
                edge_features.append(checkin_dict[(item, user)])
                matched_count += 1
            else:
                edge_features.append([float('nan')] * 4)  # Fill with NaN for missing edges
                unmatched_count += 1
    
        print(f"Matched edges: {matched_count}, Unmatched edges: {unmatched_count}")
    
        # Convert edge features to DataFrame
        df = pd.DataFrame(
            edge_features, columns=["date", "latitude", "longitude", "item"]
        )
        df["user"] = users
    
        return df
    




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

        # Create edge features (context)
        train_target_edge_features = self.create_edge_features(train_edges, num_users)
        valid_edge_features = self.create_edge_features(valid_edges, num_users)
        test_edge_features = self.create_edge_features(test_edges, num_users)

        # Add reversed edges
        reversed_train_edges = train_edges.flip(0)

        # Combine edges and set edge types
        train_edges_combined = torch.cat([train_edges, reversed_train_edges], dim=1)


        train_edge_types = torch.cat(
            [torch.zeros(train_edges.size(1), dtype=torch.int64), 
             torch.ones(reversed_train_edges.size(1), dtype=torch.int64)], dim=0
        )
        
        train_edge_features = torch.cat([train_target_edge_features, train_target_edge_features], dim=0)
        print (f"train_edge_features.shape: {train_edge_features.shape} ")
        print (f"train_edges_combined.shape: {train_edges_combined.shape} ")
        
         
        
        test_edge_types = torch.zeros(test_edges.size(1), dtype=torch.int64)
        valid_edge_types = torch.zeros(valid_edges.size(1), dtype=torch.int64)
        target_train_edges = train_edges
        target_train_edge_types = torch.zeros(train_edges.size(1), dtype=torch.int64)

        # Number of nodes and relations
        num_nodes = num_users + num_items
        num_relations = 2  # Normal and reversed edges

        # Construct Data objects
        train_data = Data(
            edge_index=train_edges_combined, edge_type=train_edge_types, edge_attr = train_edge_features,
            num_nodes=num_nodes,
            target_edge_index=target_train_edges, target_edge_type=target_train_edge_types, target_edge_attr = train_target_edge_features,
            num_relations=num_relations
        )

        valid_data = Data(
            edge_index=train_edges_combined, edge_type=train_edge_types, edge_attr = train_edge_features,
            num_nodes=num_nodes,
            target_edge_index=valid_edges, target_edge_type=valid_edge_types, target_edge_attr = valid_edge_features,
            num_relations=num_relations
        )

        test_data = Data(
            edge_index=train_edges_combined, edge_type=train_edge_types, edge_attr = train_edge_features,
            num_nodes=num_nodes,
            target_edge_index=test_edges, target_edge_type=test_edge_types, target_edge_attr = test_edge_features,
            num_relations=num_relations
        )

        print (f"num_users: {num_users} ")
        print (f"num_items: {num_items} ")
        
         # Load item features
        item_features = self.load_item_features(num_items)

        # Create generic user features
        user_features = torch.ones((num_users, 1), dtype=torch.float32)

        print (f"user_features.shape: {user_features.shape} ")
        print (f"item_features.shape: {item_features.shape} ")
        # Add metadata
        for data in [train_data, valid_data, test_data]:
            data.num_users = num_users
            data.num_items = num_items
            data.x_user = user_features
            data.x_item = item_features
            
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
        missing = 0
        matched = 0
        # Parse the JSON file
        with open(user_json_path, "r") as f:
            for line in f:
                user = json.loads(line)
                if user["user_id"] in user_map:
                    # Process the yelping_since date
                    try:
                        yelping_since = datetime.strptime(user["yelping_since"], "%Y-%m-%d").strftime("%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        yelping_since = None
                        print ("partially failed")
                    matched += 1
                    user_features[user_map[user["user_id"]]] = {
                        "review_count": int(user["review_count"]),
                        "average_stars": float(user["average_stars"]),
                        "yelping_since": yelping_since,
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
                else:
                    missing += 1
        print (f"missing user feautres: {missing}")
        print (f"matched user feautres: {matched}")
        print (f"num users: {31668}")
                    
    
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
        matched = 0
        # Parse the JSON file
        with open(item_json_path, "r") as f:
            for line in f:
                business = json.loads(line)
                if business["business_id"] in item_map:
                    matched += 1
                    # Extract and clean attributes
                    attributes = business.get("attributes", {})
                    attributes_cleaned = self.flatten_attributes(attributes)
    
                    # Process categories
                    categories = business.get("categories", "")
                    if isinstance(categories, str):
                        categories = [cat.strip() for cat in categories.split(",")]
                    latitude = float(business["latitude"]) if business["latitude"] is not None else float("nan")
                    longitude = float(business["longitude"]) if business["longitude"] is not None else float("nan")
    

    
                    # Add item features
                    item_features[item_map[business["business_id"]]] = {
                        "name": str(business["name"]),
                        "address": str(business["address"]),
                        "city": str(business["city"]),
                        "state": str(business["state"]),
                        "postal_code": str(business["postal_code"]),
                        "latitude": latitude,
                        "longitude": longitude,
                        "stars": float(business["stars"]),
                        "review_count": int(business["review_count"]),
                        "is_open": int(business["is_open"]),
                        "categories": categories,
                        "attributes": attributes_cleaned,
                        "hours": business.get("hours", {}),
                    }

                    
                    
        print (f"missing item feautres: {38048 - matched}")
    
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



    @staticmethod
    def load_edge_features_dict(file_path, user_map, item_map):
        """
        Load and remap Yelp review data into a dictionary for quick lookup.
        """
        # Read JSON lines into a DataFrame
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f]
        reviews = pd.DataFrame(data)

        
        def validate_ids(reviews):
            """
            Validate that all user_id and business_id match the expected ID format.
            """
            # Define the regex pattern for valid IDs
            id_pattern = re.compile(r"^[a-zA-Z0-9_\-]+$")
        
            # Validate user_id
            invalid_user_ids = reviews[~reviews["user_id"].str.match(id_pattern, na=False)]
            print(f"Invalid user IDs: {len(invalid_user_ids)}")
            if not invalid_user_ids.empty:
                print(invalid_user_ids)
        
            # Validate business_id
            invalid_business_ids = reviews[~reviews["business_id"].str.match(id_pattern, na=False)]
            print(f"Invalid business IDs: {len(invalid_business_ids)}")
            if not invalid_business_ids.empty:
                print(invalid_business_ids)

        #validate_ids(reviews)

        
        # Drop the text/review_id columns
        reviews = reviews.drop(columns=["review_id"])
        reviews = reviews.drop(columns=["text"])
        
        # Strip leading/trailing whitespaces
        reviews["user_id"] = reviews["user_id"].str.strip()
        reviews["business_id"] = reviews["business_id"].str.strip()

        # Remap IDs
        reviews["user_id"] = reviews["user_id"].map(user_map)
        reviews["business_id"] = reviews["business_id"].map(item_map)
        
        # Drop rows with unmapped IDs
        reviews = reviews.dropna(subset=["user_id", "business_id"]).reset_index(drop=True)
        
        # Ensure date format
        reviews["date"] = reviews["date"].apply(
            lambda x: x if len(x.strip()) > 10 else x.strip() + " 00:00:00"
        )
        reviews["date"] = pd.to_datetime(reviews["date"], format="%Y-%m-%d %H:%M:%S")
        reviews["date"] = reviews["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Create a lookup dictionary for reviews
        review_dict = {
            (row["user_id"], row["business_id"]): [row["stars"], row["useful"], row["funny"], row["cool"], row["date"]]
            for _, row in reviews.iterrows()
        }
        return review_dict

  
    @staticmethod
    def map_reviews_to_edges(edge_index, review_dict):
        """
        Align review metadata with edges, respecting edge direction.
        
        Args:
            edge_index (Tensor): Edge index tensor of shape [2, num_edges].
            review_dict (dict): Review metadata dictionary {(user, item): [stars, useful, funny, cool, date]}.
        
        Returns:
            DataFrame: DataFrame aligned with edge_index containing review features.
        """
        unmatched_count = 0  # Count unmatched edges
        matched_count = 0    # Count matched edges
        
        # Initialize lists for edge features and edge source-user mapping
        edge_features = []
        users = []
        items = []
        
        # Align features for each edge
        for edge in edge_index.t().tolist():  # Transpose to iterate over individual edges
            user, item = edge
            users.append(user)
            items.append(item)
            
            # Match reviews for the given edge direction
            if (user, item) in review_dict:
                edge_features.append(review_dict[(user, item)])
                matched_count += 1
            else:
                # Fill with NaN for unmatched edges
                edge_features.append([float('nan')] * 5)  # 5 fields: stars, useful, funny, cool, date
                unmatched_count += 1
        
        print(f"Matched edges: {matched_count}, Unmatched edges: {unmatched_count}")
        
        # Convert edge features to DataFrame
        df = pd.DataFrame(
            edge_features, columns=["stars", "useful", "funny", "cool", "date"]
        )
        df["user"] = users  # Add user ID column
        df["item"] = items
        
        return df


    def process(self):
        # Load mappings please note that item_map doesnt map to the offset (+ num_numser)
        user_map, item_map = self.load_mappings()
        num_users = len(user_map)
        num_items = len(item_map)

        print(f"num_users: {num_users}")
        print(f"num_items: {num_items}")
        
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
        

        # load edge features
        edge_features_path = os.path.join(self.raw_dir, "yelp_academic_dataset_review.json")
        edge_features_dict = self.load_edge_features_dict(edge_features_path, user_map, item_map)
        #print(f"len of edge_dict: {len(edge_features_dict)}")
        train_edge_features_df = self.map_reviews_to_edges(train_edge_index, edge_features_dict)
        test_edge_features_df = self.map_reviews_to_edges(test_edge_index, edge_features_dict)
        #test_functions.test_edge_feature_alignment(train_edge_index, train_edge_features_df)
        #test_functions.test_edge_feature_alignment(test_edge_index, test_edge_features_df)
        
        # process edge feature dfs
        meta_info = preprocess_data.get_meta_info()
        meta_info["numerical_cols"] = ["stars", "useful","funny","cool"]
        meta_info["date_cols"] = ["date"]
        meta_info["drop_cols"] = ["item", "user"]
        raw_train_edges_features=  preprocess_data.process_df((train_edge_features_df,meta_info))
        test_edges_features=  preprocess_data.process_df((test_edge_features_df,meta_info))
        
        
       
        
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

        # Adjust item IDs to prevent overlap with user IDs
        train_edge_index[1] += num_users
        test_edge_index[1] += num_users

        # Perform stratified split for validation
        train_indices, valid_indices = preprocess_data.stratified_split(train_edge_index, [0.9, 0.1], filter_by="item")[:2]
        train_target_edges = train_edge_index[:, train_indices]
        valid_target_edges = train_edge_index[:, valid_indices]
        train_target_edge_features = raw_train_edges_features[train_indices,:]
        valid_edges_features = raw_train_edges_features[valid_indices,:]

        # Combine train edges with reversed edges
        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_edge_types = torch.cat(
            [torch.zeros(train_target_edges.size(1), dtype=torch.int64), 
             torch.ones(train_target_edges.size(1), dtype=torch.int64)], dim=0
        )
        train_edges_features =  torch.cat([train_target_edge_features, train_target_edge_features], dim=0) 

        test_edge_types = torch.zeros(test_edge_index.size(1), dtype=torch.int64)
        valid_edge_types = torch.zeros(valid_target_edges.size(1), dtype=torch.int64)
        train_target_edge_types = torch.zeros(train_target_edges.size(1), dtype=torch.int64)


        # Number of nodes and relations
        num_nodes = num_users + num_items
        num_relations = 2
        print (f"size of edge_index: {train_edges.shape}")
        print (f"size of edge_attr: {train_edges_features.shape}")
        # Construct Data objects
        train_data = Data(
            edge_index=train_edges, edge_type=train_edge_types, edge_attr = train_edges_features,
            num_nodes=num_nodes,
            target_edge_index=train_target_edges, target_edge_type=train_target_edge_types, target_edge_attr = train_target_edge_features,
            num_relations=num_relations
        )

        valid_data = Data(
            edge_index=train_edges, edge_type=train_edge_types, edge_attr = train_edges_features,
            num_nodes=num_nodes,
            target_edge_index=valid_target_edges, target_edge_type=valid_edge_types, target_edge_attr = valid_edges_features,
            num_relations=num_relations
        )

        test_data = Data(
            edge_index=train_edges, edge_type=train_edge_types, edge_attr = train_edges_features,
            num_nodes=num_nodes,
            target_edge_index=test_edge_index, target_edge_type=test_edge_types, target_edge_attr = test_edges_features,
            num_relations=num_relations
        )

        # Add metadata
        for data in [train_data, valid_data, test_data]:
            data.num_users = num_users
            data.num_items = num_items
            data.x_user = user_features
            data.x_item = item_features
            print (f"size of target_edge_index: {data.target_edge_index.shape}")
            print (f"size of target_edge_attr: {data.target_edge_attr.shape}")

        #raise ValueError("everything good")

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



# a joint dataset for pre-training ULTRA on several graphs
class JointDataset(InMemoryDataset):

    datasets_map = {
        'Epinions': Epinions,
        'LastFM': LastFM,
        'BookX': BookX,
        'Ml1m': Ml1m,
        'Gowalla': Gowalla,
        'Amazon_Beauty': Amazon_Beauty,
        'Amazon_Fashion': Amazon_Fashion,
        'Amazon_Men': Amazon_Men,
        'Amazon_Games': Amazon_Games,
        'Yelp18': Yelp18
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