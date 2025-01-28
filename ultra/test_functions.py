from functools import reduce
from torch_scatter import scatter_add
from torch_geometric.data import Data
import torch
import random

def test_pyG_graph(datas):
    """
    Checks that the dataset is in correct form
    """
    # Assuming edge_index is a tensor of shape (2, num_edges)
    def has_duplicate_edges(edge_index):
        edge_set = set(map(tuple, edge_index.t().tolist()))  # Convert to set of tuples
        return len(edge_set) < edge_index.shape[1] 
    assert (not has_duplicate_edges(datas[0].edge_index)), "duplicated edges_detected in data."
    for data in datas:
        assert data.edge_index.size(1) == data.edge_attr.size(0), "size mismatch between edge_index and edge_attr"
        assert data.edge_index.size(1) == data.edge_type.size(0), "size mismatch between edge_index and edge_type"
        assert data.target_edge_index.size(1) == data.target_edge_attr.size(0), "size mismatch between target_edge_index and target_edge_attr"
        assert data.target_edge_index.size(1) == data.target_edge_type.size(0), "size mismatch between target_edge_index and target_edge_type"
        assert data.num_users == data.x_user.size(0), "size mismatch between num_users and x_user"
        assert data.num_items == data.x_item.size(0), "size mismatch between num_items and x_item"
        validate_graph (data)
        #assert (not has_duplicate_edges(data.edge_index)), "duplicated edges_detected in data."
        # to do test duplicate
    print ("Graph looks good!")

        

def debug_edge_attr_alignment(train_data, batch_with_attr):
    """
    Verify that edge attributes in batch_with_attr align with train_data.target_edge_attr.
    
    Args:
        train_data: The data object containing target_edge_index and target_edge_attr.
        batch_with_attr: Tensor containing edges and their corresponding attributes from the loader.
    """
    target_edge_index = train_data.target_edge_index.t()  # (num_edges, 2)
    target_edge_attr = train_data.target_edge_attr  # (num_edges, attr_dim)
    
    target_edge = batch_with_attr[:, :3]  # (batch_size, 3) -> [u, v, r]
    batch_attr = batch_with_attr[:, 3:]  # (batch_size, attr_dim)

    for i, edge in enumerate(target_edge):
        u, v, _ = edge.tolist()  # Extract u and v; ignore r
        batch_attr_value = batch_attr[i]  # Corresponding attribute in batch
        
        # Naively search for the edge in target_edge_index
        found_idx = -1
        for idx, target_edge_pair in enumerate(target_edge_index):
            if target_edge_pair[0] == u and target_edge_pair[1] == v:
                found_idx = idx
                break
        
        # If edge is found, compare the attributes
        if found_idx != -1:
            target_attr_value = target_edge_attr[found_idx]
            assert torch.allclose(batch_attr_value, target_attr_value), (
                f"Mismatch at edge {u, v}: "
                f"batch_attr={batch_attr_value}, target_attr={target_attr_value}"
            )
        else:
            print(f"Edge {u, v} not found in target_edge_index.")
    
    print("Debug/test completed: all attributes match.")

def test_edge_feature_alignment(edge_index, df_features):
    """
    Test if the edge identifiers in the edge_index match the order in the feature DataFrame.

    Args:
        edge_index (Tensor): Edge index tensor of shape [2, num_edges].
        df_features (DataFrame): DataFrame containing edge features aligned with edge_index.

    Raises:
        AssertionError: If the order of edge identifiers does not match the feature order.
    """
    for idx, edge in enumerate(edge_index.t().tolist()):
        u, v = edge
        # Check if the edge identifiers match
        assert (df_features.iloc[idx]["user"] == u and df_features.iloc[idx]["item"] == v), \
            f"Mismatch at index {idx}: Edge ({u}, {v}) does not match DataFrame row."
    print("All edge identifiers match the feature DataFrame.")


def extract_user_ids_from_json(user_json_path):
    """
    Extract all user IDs from the raw JSON file.
    
    Args:
        user_json_path (str): Path to the Yelp user JSON file.
    
    Returns:
        set: Set of all user IDs in the JSON file.
    """
    user_ids = set()
    with open(user_json_path, "r") as f:
        for line in f:
            user = json.loads(line)
            user_ids.add(user["business_id"])
    return user_ids

def find_mismatched_ids(user_map_keys, raw_user_ids, max_output=2):
    """
    Find and print mismatched IDs between user_map keys and raw user IDs.
    
    Args:
        user_map_keys (list): List of user IDs from the user_map.
        raw_user_ids (set): Set of user IDs from the raw JSON file.
        max_output (int): Maximum number of mismatched examples to display.
    """
    # Find exact matches
    user_map_set = set(user_map_keys)
    exact_matches = user_map_set & raw_user_ids

    # IDs to check for closeness
    unmatched_ids = user_map_set - exact_matches
    print(f"Total unmatched IDs: {len(unmatched_ids)}")



def validate_graph(data):
    """
    Validate the structure of the graph.
    Ensures:
    - User node IDs are within range [0, num_users - 1].
    - Item node IDs are within range [num_users, num_users + num_items - 1].
    - num_users + num_items == num_nodes.

    Args:
        data (Data): PyG data object containing the graph.
        num_users (int): Number of users.
        num_items (int): Number of items.
    """
    num_items = data.num_items
    num_users = data.num_users
    num_nodes = num_users + num_items

    # Validate num_nodes
    assert data.num_nodes == num_nodes, (
        f"Mismatch in num_nodes: {data.num_nodes} != {num_nodes} "
        f"(num_users + num_items)."
    )

    # Validate edge index ranges
    src, dst = data.target_edge_index
    invalid_users = (src < 0) | (src >= num_users)
    invalid_items = (dst < num_users) | (dst >= num_nodes)
    assert not invalid_users.any(), "Some user node IDs are out of range."
    assert not invalid_items.any(), "Some item node IDs are out of range."


    print("Graph validation passed!")

def validate_pred_mask(t_relevance, h_relevance, test_data, filtered_data, pos_h_index, pos_t_index):
    num_users = test_data.num_users
    device = t_relevance.device
    edge_index = test_data.target_edge_index.to(device)
    filtered_edge_index = filtered_data.edge_index.to(device)
    batch_size, num_nodes = t_relevance.shape
    
    # Validate t_relevance
    for i in range(batch_size):
        h_index = pos_h_index[i]
        for j in range(num_nodes):
            is_in_test_graph = (edge_index[0, :] == h_index).logical_and(edge_index[1, :] == j).any()
            is_in_filtered_graph = (filtered_edge_index[0, :] == h_index).logical_and(filtered_edge_index[1, :] == j).any()
            if t_relevance[i, j].item() == 1:
                if not ((not is_in_test_graph) and is_in_filtered_graph):
                    print(f"is_in_test_graph: {is_in_test_graph} is_in_filtered_graph: {is_in_filtered_graph}")
                    raise ValueError(f"Inconsistency in t_relevance = 1 at ({i}, {j}) - h_index: {h_index}, j: {j}")
            else:
                if ((not is_in_test_graph) and is_in_filtered_graph):
                    print(f"is_in_test_graph: {is_in_test_graph} is_in_filtered_graph: {is_in_filtered_graph}")
                    raise ValueError(f"Inconsistency in t_relevance = 0 at ({i}, {j}) - h_index: {h_index}, j: {j}")            

    # Validate h_relevance
    for i in range(batch_size):
        t_index = pos_t_index[i]
        for j in range(num_nodes):
            is_in_test_graph = (edge_index[1, :] == t_index).logical_and(edge_index[0, :] == j).any()
            is_in_filtered_graph = (filtered_edge_index[1, :] == t_index).logical_and(filtered_edge_index[0, :] == j).any()
            if h_relevance[i, j].item() == 1:
                if not ((not is_in_test_graph) and is_in_filtered_graph):
                    print(f"is_in_test_graph: {is_in_test_graph} is_in_filtered_graph: {is_in_filtered_graph}")
                    raise ValueError(f"Inconsistency in h_relevance = 1 at ({i}, {j}) - t_index: {t_index}, j: {j}")
            else:
                is_in_graph = (edge_index[1, :] == t_index).logical_and(edge_index[0, :] == j).any()
                if ((not is_in_test_graph) and is_in_filtered_graph):
                    print(f"is_in_test_graph: {is_in_test_graph} is_in_filtered_graph: {is_in_filtered_graph}")
                    raise ValueError(f"Inconsistency in h_relevance = 0 at ({i}, {j}) - t_index: {t_index}, j: {j}")
                    

    print("Validation of pred passed: No inconsistencies found.")

def validate_relevance(t_relevance, h_relevance, test_data, pos_h_index, pos_t_index):
    num_users = test_data.num_users
    device = t_relevance.device
    edge_index = test_data.target_edge_index.to(device)

    batch_size, num_nodes = t_relevance.shape



    # Validate t_relevance
    for i in range(batch_size):
        h_index = pos_h_index[i]
        for j in range(num_nodes):
            if t_relevance[i, j].item() == 1:
                # Check edge existence or user-user exclusion
                is_in_graph = (edge_index[0, :] == h_index).logical_and(edge_index[1, :] == j).any()
                if not (is_in_graph):
                    raise ValueError(f"Inconsistency in t_relevance = 1 at ({i}, {j}) - h_index: {h_index}, j: {j}")
            else:
                is_in_graph = (edge_index[0, :] == h_index).logical_and(edge_index[1, :] == j).any()
                if is_in_graph:
                    raise ValueError(f"Inconsistency in t_relevance = 0 at ({i}, {j}) - h_index: {h_index}, j: {j}")            

    # Validate h_relevance
    for i in range(batch_size):
        t_index = pos_t_index[i]
        for j in range(num_nodes):
            if h_relevance[i, j].item() == 1:
                # Check edge existence or item-item exclusion
                is_in_graph = (edge_index[1, :] == t_index).logical_and(edge_index[0, :] == j).any()
                if not (is_in_graph):
                    raise ValueError(f"Inconsistency in h_relevance = 1 at ({i}, {j}) - t_index: {t_index}, j: {j}")
            else:
                is_in_graph = (edge_index[1, :] == t_index).logical_and(edge_index[0, :] == j).any()
                if is_in_graph:
                    raise ValueError(f"Inconsistency in h_relevance = 0 at ({i}, {j}) - t_index: {t_index}, j: {j}")
                    

    print("Validation of relevance passed: No inconsistencies found.")


def validate_mask(t_relevance, h_relevance, test_data, pos_h_index, pos_t_index, num_users, context):
    device = t_relevance.device
    if context == 3:
        edge_index = test_data.target_edge_index.to(device)
    else:
        edge_index = test_data.edge_index.to(device)  # Move edge_index to the correct device
    batch_size, num_nodes = t_relevance.shape



    # Validate t_relevance
    for i in range(batch_size):
        h_index = pos_h_index[i]
        for j in range(num_nodes):
            if t_relevance[i, j].item() == 0:
                # Check edge existence or user-user exclusion
                is_in_graph = (edge_index[0, :] == h_index).logical_and(edge_index[1, :] == j).any()
                is_user_user = j < num_users
                if not (is_in_graph or is_user_user):
                    
                    print(f"is_in_graph is {is_in_graph} and is_user_user is {is_user_user} ")
                    raise ValueError(f"Inconsistency in t_relevance at ({i}, {j}) - h_index: {h_index}, j: {j}")

    # Validate h_relevance
    for i in range(batch_size):
        t_index = pos_t_index[i]
        for j in range(num_nodes):
            if h_relevance[i, j].item() == 0:
                # Check edge existence or item-item exclusion
                is_in_graph = (edge_index[1, :] == t_index).logical_and(edge_index[0, :] == j).any()
                is_item_item = j >= num_users
                if not (is_in_graph or is_item_item):
                    print(f"is_in_graph is {is_in_graph} and is_item_item is {is_item_item} ")
                    raise ValueError(f"Inconsistency in h_relevance at ({i}, {j}) - t_index: {t_index}, j: {j}")

    print("Validation passed: No inconsistencies found.")


def validate_indices(h_index, t_index, num_users, num_items):
    # Ensure h_index is between 0 and num_users - 1 for the second dimension
    h_valid = (h_index[:, 1:] >= 0) & (h_index[:, 1:] < num_users)
    
    # Ensure t_index is between num_users and num_users + num_items - 1 for the second dimension
    t_valid = (t_index[:, 1:] >= num_users) & (t_index[:, 1:] < num_users + num_items)
    
    # Check if all entries in h_index and t_index are valid
    h_valid_check = h_valid.all().item()
    t_valid_check = t_valid.all().item()

    # Print out the results
    if h_valid_check and t_valid_check:
        print("All indices are valid.")
    else:
        if not h_valid_check:
            raise ValueError("Invalid indices found in h_index.")
        if not t_valid_check:
            raise ValueError("Invalid indices found in t_index.")
    
    return h_valid_check, t_valid_check



