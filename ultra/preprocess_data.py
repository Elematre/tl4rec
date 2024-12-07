import os
import csv
import shutil
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from scipy.sparse import issparse

# maybe add friends as edges
# 
# Example Usage
def print_missing(df,meta_info):
    def count_missing_values(df, meta_info):
        """
        Count missing (None/NaN) values in the DataFrame for different column types.
    
        Args:
            df (pd.DataFrame): The DataFrame to analyze.
            meta_info (dict): Metadata containing column groups (categorical, numerical, date).
    
        Returns:
            dict: Counts of missing values for each column type.
        """
        counts = {"categorical": {}, "numerical": {}, "date": {}}
    
        # Count None/NaN for categorical columns
        for col in meta_info["categorical_cols"]:
            counts["categorical"][col] = df[col].isna().sum() + df[col].eq(None).sum()
    
        # Count NaN for numerical columns
        for col in meta_info["numerical_cols"]:
            counts["numerical"][col] = df[col].isna().sum()
    
        # Count NaN for date columns
        for col in meta_info["date_cols"]:
            counts["date"][col] = df[col].isna().sum()
    
        return counts
    
    missing_counts = count_missing_values(df, meta_info)
    print("Missing Values Count:")
    for col_type, col_counts in missing_counts.items():
        print(f"\n{col_type.capitalize()} Columns:")
        for col, count in col_counts.items():
            print(f"{col}: {count}")



class Debug(BaseEstimator, TransformerMixin):

    def transform(self, X):
        X_dense = X
        if issparse(X_dense):
            X_dense = X.toarray()  # Convert sparse matrix to dense
        print(f"Shape: {X_dense.shape}")
        print(f"Data type: {type(X_dense)}")
        print(f"First few rows:\n{X_dense[:5] if isinstance(X_dense, (np.ndarray, list)) else X_dense.head()}")
        return X

    def fit(self, X, y=None, **fit_params):
        return self


class DateTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer for date columns that
    """
    def __init__(self, fill_value=None, reference_date=None, date_format="%Y-%m-%d %H:%M:%S"):
        self.fill_value = fill_value
        self.reference_date = reference_date
        self.date_format = date_format

    def fit(self, X, y=None):
        # Flatten 2D data into 1D Series
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]  
        X = pd.Series(X).squeeze()

        X_parsed = pd.to_datetime(X, format=self.date_format, errors="coerce")
        if X_parsed.isna().all():
            raise ValueError("All values in the date column are invalid or missing.")

        # Set fill and reference dates
        if self.fill_value is None:
            self.fill_value = X_parsed.median()
        if self.reference_date is None:
            self.reference_date = X_parsed.min()
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        X = pd.Series(X).squeeze()
        X_parsed = pd.to_datetime(X, format=self.date_format, errors="coerce").fillna(self.fill_value)
        X_numeric = (X_parsed - self.reference_date).dt.days
        return X_numeric.values.reshape(-1, 1)


def process_df(df_tup):
    """
    processes the df with the datatypes of the cols indicated in meta_info: It handles: TODO
    df_tup = (df, meta_info)
    """
    df = df_tup[0]
    print(f"df.shape {df.shape}")
    meta_info = df_tup[1]
    
    

    
    # date pipeline
    date_transformer = Pipeline(steps=[
        ("date_conversion", DateTransformer()),
        ("scaler", StandardScaler())  # Treat transformed dates as numerical features
    ])

    # numerical pipeline
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),  
        ("scaler", StandardScaler()),
    ])
    
    # categorical pipeline 
    def lowercase_transform(X):
        """
        Normalize case for categorical columns. Handles multiple columns and preserves input shape.
        """
        if isinstance(X, pd.DataFrame):  # Multi-column case
            return X.apply(lambda col: col.str.lower() if col.dtype == "object" else col)
        elif isinstance(X, np.ndarray):  # Single-column array
            X = pd.DataFrame(X)  
            return X.map(lambda val: val.lower() if isinstance(val, str) else val).values
        else:
            raise ValueError(f"Unexpected input type for lowercase_transform: {type(X)}")



        
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("lowercase", FunctionTransformer(lowercase_transform, validate=False)),  # Normalize case
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    
    
    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, meta_info["numerical_cols"]),
        ("date", date_transformer, meta_info["date_cols"]),
        ("cat", categorical_transformer, meta_info["categorical_cols"])
    ]
)

    # Fit and transform your DataFrame
    processed_features = preprocessor.fit_transform(df)
    #print(processed_features)
    
    raise ValueError("processing was sucessful")
    return processed_features

def get_meta_info():
    meta_info = {}
    meta_info["numerical_cols"] = [] 
    meta_info["date_cols"] = [] # expected in "%Y-%m-%d %H:%M:%S" but might also handle different formats #TODO
    meta_info["categorical_cols"] = [] # low cardinality strings
    meta_info["str_cols"] = [] # high cardinality strings with low semantic info eg. name
    
    #meta_info ["list_of_cat_cols"] = []
    #meta_info ["binary_cols"] = []
    return meta_info
    
    

def stratified_split(edge_index, split_ratios, filter_by="item"):
    """
    Perform a stratified split based on the frequency distribution of items or users.

    Args:
        edge_index (torch.Tensor): Edge index tensor of shape [2, num_edges].
        split_ratios (list of float): Ratios for splitting (e.g., [0.9, 0.1]).
        filter_by (str): 'item' or 'user' to stratify by column 1 or column 0, respectively.

    Returns:
        list of torch.Tensor: Edge indices for each split.
    """
    split_col = 1 if filter_by == "item" else 0
    value_counts = defaultdict(list)

    # Group indices by their corresponding values (items or users)
    for i in range(edge_index.size(1)):
        value_counts[edge_index[split_col, i].item()].append(i)

    # Shuffle and convert to tensors
    for key in value_counts:
        value_counts[key] = torch.tensor(value_counts[key])
        perm = torch.randperm(value_counts[key].size(0))  # Shuffle within each group
        value_counts[key] = value_counts[key][perm]

    # Create split lists dynamically based on split_ratios
    num_splits = len(split_ratios)
    splits = [[] for _ in range(num_splits)]
    thresholds = [sum(split_ratios[:i + 1]) for i in range(num_splits)]

    # Distribute indices across splits
    for indices in value_counts.values():
        group_size = indices.size(0)
        cumulative = 0
        for i, threshold in enumerate(thresholds):
            split_size = int(threshold * group_size) - cumulative
            splits[i].append(indices[cumulative:cumulative + split_size])
            cumulative += split_size

    # Concatenate and shuffle within each split
    splits = [torch.cat(split) for split in splits if split]  # Avoid empty splits
    splits = [split[torch.randperm(split.size(0))] for split in splits]

    return splits




