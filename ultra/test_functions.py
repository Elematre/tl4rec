from functools import reduce
from torch_scatter import scatter_add
from torch_geometric.data import Data
import torch
import random


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



