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
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder, MultiLabelBinarizer
from scipy.sparse import issparse, hstack
from sklearn.feature_extraction.text import HashingVectorizer

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
        counts = {"categorical": {}, "numerical": {}, "date": {}, "str_cols" : {}}
    
        # Count None/NaN for categorical columns
        for col in meta_info["categorical_cols"]:
            counts["categorical"][col] = df[col].isna().sum() + df[col].eq(None).sum()

        for col in meta_info["str_cols"]:
            counts["str_cols"][col] = df[col].isna().sum() + df[col].eq(None).sum()
    
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


# Updated CategoriesTransformer
class CategoriesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.binarizer = MultiLabelBinarizer()

    def _preprocess_categories(self, X):
        processed = []
        for entry in X:
            if isinstance(entry, list):
                processed.append([cat.strip().lower() for cat in entry if isinstance(cat, str)])
            else:
                processed.append([])  # Handle NaN or invalid entries
        return processed

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0].tolist()
        X_processed = self._preprocess_categories(X)
        self.binarizer.fit(X_processed)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0].tolist()
        elif isinstance(X, np.ndarray):
            X = X.ravel()
        X_processed = self._preprocess_categories(X)
        if not X_processed or all(len(entry) == 0 for entry in X_processed):
            return np.zeros((len(X), 0))  # Empty matrix if no categories
        return self.binarizer.transform(X_processed)






class CustomHashingVectorizer(BaseEstimator, TransformerMixin):
    """
    A custom transformer to apply HashingVectorizer column-wise on a 2D array.
    """
    def __init__(self, n_features=32):
        self.n_features = n_features

    def fit(self, X, y=None):
        return self  # No fitting needed for HashingVectorizer

    def transform(self, X):
        def hash_vectorizer_array(X, n_features=32):
            """
            Applies HashingVectorizer column-wise on a numpy.ndarray.
            
            Args:
                X (numpy.ndarray): 2D array of shape (n_samples, n_columns) containing string data.
                n_features (int): Number of features for the HashingVectorizer.
        
            Returns:
                scipy.sparse.csr_matrix: Concatenated sparse matrix with transformed features.
            """
            if not isinstance(X, np.ndarray):
                raise ValueError(f"Expected input of type numpy.ndarray, got {type(X)} instead.")
            if X.ndim != 2:
                raise ValueError(f"Expected a 2D array, got an array with shape {X.shape}.")
            
            hashing_vectorizer = HashingVectorizer(n_features=n_features, norm=None, alternate_sign=False)
            
            # Apply HashingVectorizer to each column and store results
            sparse_matrices = []
            for col_idx in range(X.shape[1]):
                column_data = X[:, col_idx].astype(str)  # Ensure the column is treated as strings
                sparse_matrix = hashing_vectorizer.fit_transform(column_data)
                sparse_matrices.append(sparse_matrix)
            
            # Concatenate sparse matrices horizontally
            return hstack(sparse_matrices)
        return hash_vectorizer_array(X, n_features=self.n_features)


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
    meta_info = df_tup[1]
    print(f"df.shape {df.shape}")
    df = df.drop(columns=meta_info["drop_cols"], errors="ignore")
    print(f"df.shape {df.shape}")

    # DEBUG Extract indices of rows with all null values
    null_row_indices = df.index[df.isnull().all(axis=1)].tolist()
    print(f"Number of null rows: {len(null_row_indices)}")
    
    
    #print_missing(df,meta_info)
    
    

    
    # date pipeline
    date_transformer = Pipeline(steps=[
        ("date_conversion", DateTransformer()),
        ("scaler", StandardScaler())  # Treat transformed dates as numerical features
        #("debug", Debug())
    ])

    # numerical pipeline
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),  
        ("scaler", StandardScaler())
        #("debug", Debug())
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
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
        #("debug", Debug())
    ])

    # string pipeline
    string_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("custom_hashing", CustomHashingVectorizer(n_features=32))
    ])
    
    # ls_of_cats pipeline
    ls_of_cats_transformer = Pipeline(steps=[
        ("categories", CategoriesTransformer())
        #("debug 2", Debug()),
    ])
    
    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, meta_info["numerical_cols"]),
        ("date", date_transformer, meta_info["date_cols"]),
        ("cat", categorical_transformer, meta_info["categorical_cols"]),
        ("str", string_transformer, meta_info["str_cols"]),
        ("ls_of_cats", ls_of_cats_transformer, meta_info["ls_of_cat_string"]) 
    ]
)


    processed_features = preprocessor.fit_transform(df)
    if hasattr(processed_features, "toarray"):
        processed_features = processed_features.toarray()

    # DEBUG check null rows
    null_rows_after_preprocessing = processed_features[null_row_indices, :]
    #for i, row in enumerate(null_rows_after_preprocessing[:5]):  # Print first 5 null rows
        #print(f"Row {null_row_indices[i]} after preprocessing: {row}")
    if not np.all(np.equal(processed_features[null_row_indices], null_rows_after_preprocessing)):
        raise ValueError("Null row values have changed during preprocessing!")

    # Convert to PyTorch tensor
    feature_tensor = torch.tensor(processed_features, dtype=torch.float32)
    print(f"Feature tensor shape: {feature_tensor.shape}")

    # DEBUG: Validate no zero columns in the tensor
    column_sums = feature_tensor.sum(dim=0)
    zero_columns = (column_sums == 0).nonzero(as_tuple=True)[0]
    if len(zero_columns) > 0:
        raise ValueError(f"Zero-information columns detected at indices: {zero_columns.tolist()}")
        
    #raise ValueError("feature preprocessing sucessful")   
    
    
    return feature_tensor

def get_meta_info():
    meta_info = {}
    meta_info["numerical_cols"] = [] 
    meta_info["date_cols"] = [] # expected in "%Y-%m-%d %H:%M:%S" but might also handle different formats #TODO
    meta_info["categorical_cols"] = [] # low cardinality strings
    meta_info["str_cols"] = [] # high cardinality strings with low semantic info eg. name
    meta_info["drop_cols"] = [] # columns that should be dropped and not processed e.g name
    meta_info["ls_of_cat_string"] = [] # columns that consist of lists of categorical strings. E.g mulitple categories per entry
    
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




