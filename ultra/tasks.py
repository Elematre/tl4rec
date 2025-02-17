from functools import reduce
from torch_scatter import scatter_add
from torch_geometric.data import Data
import torch
import random


def build_candidate_set(test_data, filtered_data, batch, cand_size, num_users):
    """
    Build candidate sets for tail and head prediction for evaluation.
    
    For each positive triplet in the batch (of shape (bs, 3), each row is [h, t, r]):
      - Compute the test positives from test_data (using context=3) and use filtered_data (which
        contains all edges: train, valid, test) to determine the valid negatives.
      - For tail prediction, for the given (h, r) pair:
            • pos_t: all candidate tails that are positives in the test set.
            • valid_neg_t: all items that do NOT appear for (h, r) in filtered_data.
            • Then the candidate set is: all pos_t ∪ a random sample from (valid_neg_t minus pos_t)
              so that the total size equals cand_size.
      - Similarly for head prediction.
      - If the available negatives are fewer than required, we “pad” by repeating (tiling) the negatives.
      - Finally, for tail prediction, we form candidate triplets (h, candidate, r) and for head prediction,
        (candidate, t, r).
    
    Returns:
       t_batch: Tensor of shape (bs, cand_size, 3) for tail prediction.
       h_batch: Tensor of shape (bs, cand_size, 3) for head prediction.
       
    Parameters:
      test_data: Data object containing test edges and test-related attributes.
      filtered_data: Data object whose edge_index (and edge_type) includes all edges (train/valid/test);
                     used here to filter out items that should not be recommended.
      batch: Tensor of shape (bs, 3) where each row is a positive triplet [h, t, r].
      cand_size: The desired candidate set size (e.g. 1000).
      num_users: Number of users (used in mask adjustment).
    """
    bs = batch.size(0)
    
    # --- Step 1. Compute Masks ---
    # Use test_data (context=3) to get the test positives.
    t_mask_test, h_mask_test = strict_negative_mask(test_data, batch, context=3)
    t_test_pos, h_test_pos = invert_mask(t_mask_test, h_mask_test, num_users)
    
    # Use filtered_data (context=1) to compute negatives from the full graph.
    t_valid_neg, h_valid_neg = strict_negative_mask(filtered_data, batch, context=1)
    # t_valid_neg is True for nodes that are *not* connected in filtered_data.
    
    t_candidate_list = []
    h_candidate_list = []
    
    # --- Step 2. Build candidate sets per batch sample ---
    for i in range(bs):
        # --- Tail Prediction: For (h, r), predict the tail candidate.
        # Get the test positives: these are items that appear as tails in the test set for (h, r)
        pos_t = torch.where(t_test_pos[i])[0]  # indices where test positive mask is True
        # Valid negatives: items that are not connected in the full graph
        # (Exclude any that are test positives so we do not duplicate them.)
        valid_neg_t = torch.where(t_valid_neg[i])[0]
    
        if pos_t.numel() >= cand_size:
            candidate_t = pos_t[:cand_size]
        else:
            required = cand_size - pos_t.numel()
            if valid_neg_t.numel() > 0:
                if valid_neg_t.numel() >= required:
                    perm = torch.randperm(valid_neg_t.numel(), device=batch.device)
                    sampled_neg_t = valid_neg_t[perm[:required]]
                else:
                    # Not enough negatives: tile them until we have at least 'required' items,
                    # then randomly shuffle and take the first required.
                    reps = (required + valid_neg_t.numel() - 1) // valid_neg_t.numel()  # ceiling division
                    tiled = valid_neg_t.repeat(reps)
                    perm = torch.randperm(tiled.numel(), device=batch.device)
                    sampled_neg_t = tiled[perm[:required]]
            else:
                # If there are no negatives available, repeat the positives.
                reps = (required + pos_t.numel() - 1) // pos_t.numel() if pos_t.numel() > 0 else 1
                sampled_neg_t = pos_t.repeat(reps)[:required]
            candidate_t = torch.cat([pos_t, sampled_neg_t])
        
        # Build the tail candidate triplets: (h, candidate, r)
        h_val = batch[i, 0].expand(cand_size)
        r_val = batch[i, 2].expand(cand_size)
        t_triplets = torch.stack([h_val, candidate_t, r_val], dim=1)
        t_candidate_list.append(t_triplets)
        
        # --- Head Prediction: For (t, r), predict the head candidate.
        pos_h = torch.where(h_test_pos[i])[0]
        valid_neg_h = torch.where(h_valid_neg[i] & (~h_test_pos[i]))[0]
        if pos_h.numel() >= cand_size:
            candidate_h = pos_h[:cand_size]
        else:
            required = cand_size - pos_h.numel()
            if valid_neg_h.numel() > 0:
                if valid_neg_h.numel() >= required:
                    perm = torch.randperm(valid_neg_h.numel(), device=batch.device)
                    sampled_neg_h = valid_neg_h[perm[:required]]
                else:
                    reps = (required + valid_neg_h.numel() - 1) // valid_neg_h.numel()
                    tiled = valid_neg_h.repeat(reps)
                    perm = torch.randperm(tiled.numel(), device=batch.device)
                    sampled_neg_h = tiled[perm[:required]]
            else:
                reps = (required + pos_h.numel() - 1) // pos_h.numel() if pos_h.numel() > 0 else 1
                sampled_neg_h = pos_h.repeat(reps)[:required]
            candidate_h = torch.cat([pos_h, sampled_neg_h])
        
        # Build the head candidate triplets: (candidate, t, r)
        t_val = batch[i, 1].expand(cand_size)
        r_val = batch[i, 2].expand(cand_size)
        h_triplets = torch.stack([candidate_h, t_val, r_val], dim=1)
        h_candidate_list.append(h_triplets)
    
    # --- Step 3. Stack candidate triplets into final tensors ---
    # Each is of shape (bs, cand_size, 3)
    t_batch = torch.stack(t_candidate_list, dim=0)
    h_batch = torch.stack(h_candidate_list, dim=0)
    
    return t_batch, h_batch



def edge_match(edge_index, query_index):
    # O((n + q)logn) time
    # O(n) memory
    # edge_index: big underlying graph
    # query_index: edges to match

    # preparing unique hashing of edges, base: (max_node, max_relation) + 1
    base = edge_index.max(dim=1)[0] + 1
    # we will map edges to long ints, so we need to make sure the maximum product is less than MAX_LONG_INT
    # idea: max number of edges = num_nodes * num_relations
    # e.g. for a graph of 10 nodes / 5 relations, edge IDs 0...9 mean all possible outgoing edge types from node 0
    # given a tuple (h, r), we will search for all other existing edges starting from head h
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)
    scale = scale[-1] // scale

    # hash both the original edge index and the query index to unique integers
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)

    # matched ranges: [start[i], end[i])
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    # num_match shows how many edges satisfy the (h, r) pattern for each query in the batch
    num_match = end - start

    # generate the corresponding ranges
    offset = num_match.cumsum(0) - num_match
    range = torch.arange(num_match.sum(), device=edge_index.device)
    range = range + (start - offset).repeat_interleave(num_match)

    return order[range], num_match




def negative_sampling(data, batch, num_negative, strict=True):
    batch_size = len(batch)
    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # strict negative sampling vs random negative sampling
    if strict:
        t_mask, h_mask = strict_negative_mask(data, batch)
        t_mask = t_mask[:batch_size // 2]
        neg_t_candidate = t_mask.nonzero()[:, 1] # 1 since we are interested in the true COL indices
        num_t_candidate = t_mask.sum(dim=-1)
        # draw samples for negative tails
        
        
        rand = torch.rand(len(t_mask), num_negative, device=batch.device)

        
        index = (rand * num_t_candidate.unsqueeze(-1)).long()
        # now index is (bs//2, num_neg) containing a random index from 0 to number of matches for this batchrow h,r
        # this calculates a correct offset since the are num_t_candidate entries in neg_t_candidate
        index = index + (num_t_candidate.cumsum(0) - num_t_candidate).unsqueeze(-1)

        neg_t_index = neg_t_candidate[index]

        h_mask = h_mask[batch_size // 2:]
        neg_h_candidate = h_mask.nonzero()[:, 1]
        num_h_candidate = h_mask.sum(dim=-1)
        # draw samples for negative heads
        rand = torch.rand(len(h_mask), num_negative, device=batch.device)
        index = (rand * num_h_candidate.unsqueeze(-1)).long()
        index = index + (num_h_candidate.cumsum(0) - num_h_candidate).unsqueeze(-1)
        neg_h_index = neg_h_candidate[index]
    else:
        neg_index = torch.randint(data.num_nodes, (batch_size, num_negative), device=batch.device)
        neg_t_index, neg_h_index = neg_index[:batch_size // 2], neg_index[batch_size // 2:]

    h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
    r_index = pos_r_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index[:batch_size // 2, 1:] = neg_t_index
    h_index[batch_size // 2:, 1:] = neg_h_index
    # validate_indices(h_index,t_index, data.num_users, data.num_items)

    return torch.stack([h_index, t_index, r_index], dim=-1)


def all_negative(data, batch):
    pos_h_index, pos_t_index, pos_r_index = batch.t()
    r_index = pos_r_index.unsqueeze(-1).expand(-1, data.num_nodes)
    # generate all negative tails for this batch
    all_index = torch.arange(data.num_nodes, device=batch.device)
    h_index, t_index = torch.meshgrid(pos_h_index, all_index, indexing="ij")  # indexing "xy" would return transposed
    t_batch = torch.stack([h_index, t_index, r_index], dim=-1)
    # generate all negative heads for this batch
    all_index = torch.arange(data.num_nodes, device=batch.device)
    t_index, h_index = torch.meshgrid(pos_t_index, all_index, indexing="ij")
    h_batch = torch.stack([h_index, t_index, r_index], dim=-1)
    
    # t_batch has all corrupted tail nodes important not only including items but also users. but this doesnt really matter since we
    # mask the users for ranking calculation anyway by strict_negative_mask
    return t_batch, h_batch


def strict_negative_mask(data, batch, context = 1):
    # based on the context where the method is called we want some slightly different behaviour:
    # context 3 (testing) we want to use the target_edge_index instead of the normal edge_index
    # this function makes sure that for a given (h, r) batch we will NOT sample true tails as random negatives
    # similarly, for a given (t, r) we will NOT sample existing true heads as random negatives
    # print("strict_negative_mask data.edge_index shape:", data.edge_index.shape)
    pos_h_index, pos_t_index, pos_r_index = batch.t()
    num_users = data.num_users

    
    # part I: sample hard negative tails
    # edge index of all (head, relation) edges from the underlying graph
    if context == 3:
        edge_index = torch.stack([data.target_edge_index[0], data.target_edge_type]) # (2,num_edges)
    else:
        edge_index = torch.stack([data.edge_index[0], data.edge_type]) # (2,num_edges)
    # edge index of current batch (head, relation) for which we will sample negatives
    query_index = torch.stack([pos_h_index, pos_r_index]) # (2,bs )
    # search for all true tails for the given (h, r) batch
    edge_id, num_t_truth = edge_match(edge_index, query_index)
        
    # edge ids= ids of matched edges 
    # num_t_truth num of matched edges per query size (bs)
    # build an index from the found edges
    if context == 3:
        t_truth_index = data.target_edge_index[1, edge_id]
    else:
        t_truth_index = data.edge_index[1, edge_id]
        
    sample_id = torch.arange(len(num_t_truth), device=batch.device).repeat_interleave(num_t_truth)
    t_mask = torch.ones(len(num_t_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true tails
    t_mask[sample_id, t_truth_index] = 0
    t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)
    # t_mask is (bs, num_nodes) with false at all tail indices which exist in the graph
    # now we want to prevent from sampling users as tails thus 
    t_mask[:, :num_users] = 0
    

    # part II: sample hard negative heads
    # edge_index[1] denotes tails, so the edge index becomes (t, r)
    if context == 3:
        edge_index = torch.stack([data.target_edge_index[1], data.target_edge_type])
    else:
        edge_index = torch.stack([data.edge_index[1], data.edge_type])
    # edge index of current batch (tail, relation) for which we will sample heads
    query_index = torch.stack([pos_t_index, pos_r_index])
    # search for all true heads for the given (t, r) batch
    edge_id, num_h_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    if context == 3:
        h_truth_index = data.target_edge_index[0, edge_id]
    else:
        h_truth_index = data.edge_index[0, edge_id]
        
    sample_id = torch.arange(len(num_h_truth), device=batch.device).repeat_interleave(num_h_truth)
    h_mask = torch.ones(len(num_h_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true heads
    h_mask[sample_id, h_truth_index] = 0
    h_mask.scatter_(1, pos_h_index.unsqueeze(-1), 0)
    h_mask[:, num_users:] = 0   
    return t_mask, h_mask

def invert_mask(t_mask, h_mask, num_users):
    t_mask_inv = (~t_mask.bool())
    h_mask_inv = (~h_mask.bool())
    # adjustment: user -user /item - item edges now have a 1 since we filtered them out in strict_negative_mask
    t_mask_inv[:, :num_users] = 0
    h_mask_inv [:, num_users:] = 0
    return t_mask_inv, h_mask_inv
        
    
# originally method for ultra. Assumes pred is of size (bs, num_nodes) used on most datasets where we evaluate vs all negatives
def compute_ranking(pred, target, mask=None):
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    #pos_pred = (bs,1)
    if mask is not None:
        # filtered ranking
        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    else:
        # unfiltered ranking
        ranking = torch.sum(pos_pred <= pred, dim=-1) + 1
    # ranking = (bs)
    return ranking

def compute_ranking_debug(pred, target, mask=None):
    # Gather the positive predictions for each instance
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    print(f"[DEBUG] pos_pred: {pos_pred}")

    if mask is not None:
         # --- Check that for each sample, the mask at the target index is 0 ---
        target_mask_values = mask.gather(-1, target.unsqueeze(-1))
        if (target_mask_values != 0).any():
            print("[DEBUG] Warning: Found non-zero mask values at target indices:")
            print(target_mask_values)
        else:
            print("[DEBUG] All target indices in mask are 0 as expected.")

        # Create a tensor that indicates where pos_pred < pred AND the mask is True.
        condition_tensor = (pos_pred < pred) & mask
        #print(f"[DEBUG] Condition tensor ((pos_pred < pred) & mask):\n{condition_tensor}")

        # Count the number of True values per batch instance.
        count_true = torch.sum(condition_tensor, dim=-1)
        #print(f"[DEBUG] Count of True values in condition tensor per row: {count_true}")
        
        # Count the number of True values per batch instance.
        count_true_mask = torch.sum(mask, dim=-1)
        print(f"[DEBUG] Count of True values in condition tensor per row: {count_true_mask}")
        
        # --- Calculate the row-wise average of prediction values where mask is True ---
        row_avg_pred = []
        for i in range(pred.shape[0]):
            if mask[i].sum() > 0:
                avg_val = pred[i][mask[i]].mean().item()
            else:
                avg_val = float('nan')
            row_avg_pred.append(avg_val)
        row_avg_pred = torch.tensor(row_avg_pred, device=pred.device)
        print(f"[DEBUG] Row-wise average prediction value for entries where mask is True:\n{row_avg_pred}")
        

        # Optionally, get the maximum prediction score where the mask is True for each row.
        masked_pred = pred.clone()
        masked_pred[~mask] = float('-inf')
        max_masked_pred, _ = masked_pred.max(dim=-1)
        print(f"[DEBUG] Maximum pred values where mask is True per row: {max_masked_pred}")

        ranking = count_true + 1
    else:
        # Unfiltered ranking: count all candidates where pos_pred <= pred.
        condition_tensor = pos_pred <= pred
        print(f"[DEBUG] Condition tensor (pos_pred <= pred):\n{condition_tensor}")

        ranking = torch.sum(condition_tensor, dim=-1) + 1

    print(f"[DEBUG] Final computed ranking: {ranking}")
    return ranking

# adjusted method. Assumes pred is of size (bs, 1 + num_negs) used for evaluation on the amazon datasets (against 100 negative)
def compute_ranking_against_num_negs(pred, target=None, mask=None):
    """
    Computes the rank for each row by counting how many entries in pred are strictly greater
    than the first entry. Assumes pred is of shape (bs, 1 + num_negs) where the first entry 
    is the positive sample.

    Parameters:
        pred (torch.Tensor): The predictions of shape (bs, 1 + num_negs).
        target: (Not used here, included for API compatibility)
        mask: (Optional) A mask to apply; if provided, only entries where mask==True are considered.

    Returns:
        torch.Tensor: The rank for each sample in the batch (1-based indexing).
    """
    # Extract the positive prediction (assumed to be at index 0)
    pos_pred = pred[:, 0:1]  # shape: (bs, 1)
    
    if mask is not None:
        # Apply the mask to the predictions (we assume mask has the same shape as pred)
        # Only consider entries where mask is True.
        # Note: Make sure that the positive sample is always unmasked.
        pred_to_consider = torch.where(mask, pred, torch.tensor(float('-inf'), device=pred.device))
    else:
        pred_to_consider = pred

    # For each row, count how many entries are strictly greater than the positive's score.
    # This yields a 0-based rank (0 means the positive is the highest); add 1 to obtain a 1-based rank.
    rankings = torch.sum(pred_to_consider > pos_pred, dim=-1) + 1
    return rankings

    

def get_relevance_labels(t_mask, h_mask, pred_type="tail"):
    """
    Generate relevance labels for ranking based on masks.

    Args:
        t_mask (Tensor): Tail negative mask, shape (batch_size, num_nodes).
                        t_mask[i, j] = 1 if edge (h(i), j) does not exist in the graph.
        h_mask (Tensor): Head negative mask, shape (batch_size, num_nodes).
                        h_mask[i, j] = 1 if edge (j, t(i)) does not exist in the graph.
        pred_type (str): Either "tail" or "head", determines which mask to use.

    Returns:
        Tensor: Relevance labels, shape (batch_size, num_nodes).
                rel[i, j] = 1 if edge (h(i), j) or (j, t(i)) exists; otherwise 0.
    """
    if pred_type == "tail":
        # Negate t_mask to assign relevance = 1 for edges that exist
        relevance = (~t_mask.bool()).float()
    elif pred_type == "head":
        # Negate h_mask to assign relevance = 1 for edges that exist
        relevance = (~h_mask.bool()).float()
    else:
        raise ValueError(f"Invalid pred_type: {pred_type}. Must be 'tail' or 'head'.")

    return relevance


def compute_ndcg_at_k(pred, target, k):
    """
    Compute NDCG@k for a batch of predictions.

    Args:
        pred (Tensor): Predicted scores, shape (batch_size, num_candidates).
        target (Tensor): Ground truth relevance, shape (batch_size, num_candidates).
        k (int): Cutoff for NDCG computation.

    Returns:
        Tensor: NDCG@k for each batch instance, shape (batch_size,).
    """
    batch_size = pred.size(0)

    # Step 1: Sort predictions and associated relevance scores by descending order of predictions
    _, indices = torch.topk(pred, k, dim=1, largest=True, sorted=True)  # Top-k indices
    sorted_relevance = target.gather(1, indices)  # Relevance of top-k predictions

    # Step 2: Compute DCG@k
    discount = 1.0 / torch.log2(torch.arange(2, k + 2, device=pred.device).float())  # Log discount factors
    dcg = (sorted_relevance * discount).sum(dim=1)  # Sum discounted relevance scores

    # Step 3: Compute IDCG@k (Ideal DCG)
    ideal_relevance, _ = torch.topk(target.float(), k, dim=1, largest=True, sorted=True)  # Top-k ideal relevance
    idcg = (ideal_relevance * discount).sum(dim=1)  # Sum ideal discounted relevance scores

    # Step 4: Compute NDCG@k
    ndcg = dcg / idcg
    if torch.isnan(ndcg).any():
        print("Warning: Found NaN values in NDCG; replacing with 0.0")
    ndcg[torch.isnan(ndcg)] = 0.0  # Handle cases where idcg is 0 (no relevant items)
   
    return ndcg

def build_relation_graph(graph):

    # expect the graph is already with inverse edges
    

    edge_index, edge_type = graph.edge_index, graph.edge_type
    num_nodes, num_rels = graph.num_nodes, graph.num_relations
    device = edge_index.device
    
    Eh = torch.vstack([edge_index[0], edge_type]).T.unique(dim=0)  # (num_edges, 2)
    # this calculate the degree of the heads where dh[i] = is the degree of node i
    Dh = scatter_add(torch.ones_like(Eh[:, 1]), Eh[:, 0])
    
    # this basically creates a spars matrix where first vector has size (2,num_non_zero_elements) for a 2D matrix and the value has size  
    # num_non_zero_elements. Thus here a sparse matrix where for each h,r we have M(r,h) = 1 / deg h
    EhT = torch.sparse_coo_tensor(
        torch.flip(Eh, dims=[1]).T, 
        torch.ones(Eh.shape[0], device=device) / Dh[Eh[:, 0]], 
        (num_rels, num_nodes)
    )

    Eh = torch.sparse_coo_tensor(
        Eh.T, 
        torch.ones(Eh.shape[0], device=device), 
        (num_nodes, num_rels)
    )

    
    

    Et = torch.vstack([edge_index[1], edge_type]).T.unique(dim=0)  # (num_edges, 2)


    

    Dt = scatter_add(torch.ones_like(Et[:, 1]), Et[:, 0])
    assert not (Dt[Et[:, 0]] == 0).any()

    EtT = torch.sparse_coo_tensor(
        torch.flip(Et, dims=[1]).T, 
        torch.ones(Et.shape[0], device=device) / Dt[Et[:, 0]], 
        (num_rels, num_nodes)
    )

    Et = torch.sparse_coo_tensor(
        Et.T, 
        torch.ones(Et.shape[0], device=device), 
        (num_nodes, num_rels)
    )

    Ahh = torch.sparse.mm(EhT, Eh).coalesce()
    Att = torch.sparse.mm(EtT, Et).coalesce()
    Aht = torch.sparse.mm(EhT, Et).coalesce()
    Ath = torch.sparse.mm(EtT, Eh).coalesce()

    hh_edges = torch.cat([Ahh.indices().T, torch.zeros(Ahh.indices().T.shape[0], 1, dtype=torch.long).fill_(0)], dim=1)  # head to head
    tt_edges = torch.cat([Att.indices().T, torch.zeros(Att.indices().T.shape[0], 1, dtype=torch.long).fill_(1)], dim=1)  # tail to tail
    ht_edges = torch.cat([Aht.indices().T, torch.zeros(Aht.indices().T.shape[0], 1, dtype=torch.long).fill_(2)], dim=1)  # head to tail
    th_edges = torch.cat([Ath.indices().T, torch.zeros(Ath.indices().T.shape[0], 1, dtype=torch.long).fill_(3)], dim=1)  # tail to head

    rel_graph = Data(
        edge_index=torch.cat([hh_edges[:, [0, 1]].T, tt_edges[:, [0, 1]].T, ht_edges[:, [0, 1]].T, th_edges[:, [0, 1]].T], dim=1), 
        edge_type=torch.cat([hh_edges[:, 2], tt_edges[:, 2], ht_edges[:, 2], th_edges[:, 2]], dim=0),
        num_nodes=num_rels, 
        num_relations=4
    )

    graph.relation_graph = rel_graph
    return graph


