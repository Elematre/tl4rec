from functools import reduce
from torch_scatter import scatter_add
from torch_geometric.data import Data
import torch
import random



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



# needs to be optimized
def negative_sampling1(data, batch, num_negative, strict=True):
    """
    Generate negative samples for a batch of triples, avoiding edges that already exist in the graph.

    Returns:
        Tensor: A tensor of shape [batch_size, num_negative + 1, 3] with positive and negative samples.
    """
    
    batch_size = len(batch)
    pos_h_index, pos_t_index, pos_r_index = batch.t()
    num_items = data.num_items
    num_users = data.num_users

    # If strict sampling is enabled, avoid sampling edges already present in the graph
    if strict:
        # Convert existing edges to a set for fast lookups
        existing_edges = set((data.edge_index[0, i].item(), data.edge_index[1, i].item())
                             for i in range(data.edge_index.size(1)))

        neg_t_index = []
        neg_h_index = []

        # Generate negative tail samples for the first half of the batch
        for h in pos_h_index[:batch_size // 2]:
            h = h.item()
            tail_negatives = set()  

            while len(tail_negatives) < num_negative:
                # Randomly sample a tail node and check if (h, t) exists
                t_neg = random.randint(num_users, num_users + num_items - 1)
                if (h, t_neg) not in existing_edges:
                    tail_negatives.add(t_neg)
            neg_t_index.append(list(tail_negatives))

        # Generate negative head samples for the second half of the batch
        for t in pos_t_index[batch_size // 2:]:
            t = t.item()
            head_negatives = set()

            while len(head_negatives) < num_negative:
                h_neg = random.randint(0, data.num_users - 1)
                if (h_neg, t) not in existing_edges:
                    head_negatives.add(h_neg)
            neg_h_index.append(list(head_negatives))

        # Convert to tensors
        neg_t_index = torch.tensor(neg_t_index, device=batch.device)
        neg_h_index = torch.tensor(neg_h_index, device=batch.device)
        neg_t_index = neg_t_index.view(batch_size // 2, num_negative)
        neg_h_index = neg_h_index.view(batch_size - (batch_size // 2), num_negative)

    else:
        # Random negative sampling without checking for existence in graph
        neg_index = torch.randint(data.num_nodes, (batch_size, num_negative), device=batch.device)
        neg_t_index, neg_h_index = neg_index[:batch_size // 2], neg_index[batch_size // 2:]

    h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
    r_index = pos_r_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index[:batch_size // 2, 1:] = neg_t_index
    h_index[batch_size // 2:, 1:] = neg_h_index

    return torch.stack([h_index, t_index, r_index], dim=-1)



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
    
# adjusted method. Assumes pred is of size (bs, 1 + num_negs) used for evaluation on the amazon datasets (against 100 negative)
def compute_ranking_against_num_negs(pred, target, mask=None):
    
     # Sort predictions along the last dimension in descending order
    sorted_pred, sorted_indices = pred.sort(dim=-1, descending=True)
    
    # Find where the positive sample (index 0 in `pred`) is located in the sorted predictions
    positive_indices = (sorted_indices == 0).nonzero(as_tuple=True)
    
    # The second value in `positive_indices` gives the rank of the positive sample within each batch
    rankings = positive_indices[1] + 1  # Convert 0-based to 1-based ranking
    
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


