import numpy as np
import torch
from torch import nn, optim, Tensor

from torch_sparse import SparseTensor, matmul

from torch_geometric.utils import structured_negative_sampling
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj

# defines LightGCN model
class LightGCN(MessagePassing):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126
    """

    def __init__(self, num_users, num_items, embedding_dim=64, K=3, add_self_loops=False):
        """Initializes LightGCN Model

        Args:
            num_users (int): Number of users
            num_items (int): Number of items
            embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 8.
            K (int, optional): Number of message passing layers. Defaults to 3.
            add_self_loops (bool, optional): Whether to add self loops for message passing. Defaults to False.
        """
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.embedding_dim, self.K = embedding_dim, K
        self.add_self_loops = add_self_loops

        self.users_emb = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embedding_dim) # e_u^0
        self.items_emb = nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.embedding_dim) # e_i^0

        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    
    def forward_old(self, edge_index: SparseTensor):
        """Forward propagation of LightGCN Model.

        Args:
            edge_index (SparseTensor): adjacency matrix

        Returns:
            tuple (Tensor): e_u_k, e_u_0, e_i_k, e_i_0
        """
        # compute \tilde{A}: symmetrically normalized adjacency matrix
        edge_index_norm = gcn_norm(
            edge_index, add_self_loops=self.add_self_loops)

        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight]) # E^0
        embs = [emb_0]
        emb_k = emb_0

        # multi-scale diffusion
        for i in range(self.K):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1) # E^K

        users_emb_final, items_emb_final = torch.split(
            emb_final, [self.num_users, self.num_items]) # splits into e_u^K and e_i^K

        # returns e_u^K, e_u^0, e_i^K, e_i^0
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight
    
    def forward(self, edge_index: SparseTensor):
        edge_index_norm = gcn_norm(edge_index, add_self_loops=self.add_self_loops)
        
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        emb_final = emb_0 / (self.K + 1) # Start with base layer contribution
        emb_k = emb_0

        for i in range(self.K):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            emb_final = emb_final + (emb_k / (self.K + 1)) # Add layer contribution

        users_emb_final, items_emb_final = torch.split(
            emb_final, [self.num_users, self.num_items])
        
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        # computes \tilde{A} @ x
        return matmul(adj_t, x)
    

def bpr_loss_old(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, lambda_val):
    """Bayesian Personalized Ranking Loss as described in https://arxiv.org/abs/1205.2618

    Args:
        users_emb_final (torch.Tensor): e_u_k
        users_emb_0 (torch.Tensor): e_u_0
        pos_items_emb_final (torch.Tensor): positive e_i_k
        pos_items_emb_0 (torch.Tensor): positive e_i_0
        neg_items_emb_final (torch.Tensor): negative e_i_k
        neg_items_emb_0 (torch.Tensor): negative e_i_0
        lambda_val (float): lambda value for regularization loss term

    Returns:
        torch.Tensor: scalar bpr loss value
    """
    reg_loss = lambda_val * (users_emb_0.norm(2).pow(2) +
                             pos_items_emb_0.norm(2).pow(2) +
                             neg_items_emb_0.norm(2).pow(2)) # L2 loss

    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1) # predicted scores of positive samples
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1) # predicted scores of negative samples

    loss = -torch.mean(torch.nn.functional.softplus(-(pos_scores - neg_scores))) + reg_loss #softplus

    return loss

def bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, lambda_val):
    """
    Optimized Bayesian Personalized Ranking Loss.
    """
    batch_size = users_emb_final.size(0)

    # 1. Faster Regularization: sum of squares avoids unnecessary square roots
    # We divide by batch_size to keep LAMBDA consistent across experiments
    reg_loss = lambda_val * (
        users_emb_0.pow(2).sum() + 
        pos_items_emb_0.pow(2).sum() + 
        neg_items_emb_0.pow(2).sum()
    ) / batch_size

    # 2. Optimized Scoring: Dot product via element-wise mul and sum
    pos_scores = (users_emb_final * pos_items_emb_final).sum(dim=-1)
    neg_scores = (users_emb_final * neg_items_emb_final).sum(dim=-1)

    # 3. Stable BPR Term: logsigmoid is more numerically stable than softplus
    # The objective is to maximize log(sigmoid(pos - neg))
    # PyTorch minimizes, so we take the negative mean.
    #bpr_loss_term = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
    bpr_loss_term = -torch.mean(torch.nn.functional.softplus(-(pos_scores - neg_scores)))

    return bpr_loss_term + reg_loss


# helper function to get N_u
def get_user_positive_items(edge_index):
    """Generates dictionary of positive items for each user

    Args:
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        dict: dictionary of positive items for each user
    """
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
    return user_pos_items


# computes recall@K and precision@K
def RecallPrecision_ATk(groundTruth, r, k):
    """Computers recall @ k and precision @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (intg): determines the top k items to compute precision and recall on

    Returns:
        tuple: recall @ k, precision @ k
    """
    num_correct_pred = torch.sum(r, dim=-1)  # number of correctly predicted items per user
    # number of items liked by each user in the test set
    user_num_liked = torch.Tensor([len(groundTruth[i])
                                  for i in range(len(groundTruth))])
    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred) / k
    return recall.item(), precision.item()


# computes NDCG@K
def NDCGatK_r(groundTruth, r, k):
    """Computes Normalized Discounted Cumulative Gain (NDCG) @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (int): determines the top k items to compute ndcg on

    Returns:
        float: ndcg @ k
    """
    assert len(r) == len(groundTruth)

    test_matrix = torch.zeros((len(r), k))

    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1. / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.mean(ndcg).item()


# wrapper function to get evaluation metrics
def get_metrics_old(model, edge_index, exclude_edge_indices, k):
    """Computes the evaluation metrics: recall, precision, and ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        k (int): determines the top k items to compute metrics on

    Returns:
        tuple: recall @ k, precision @ k, ndcg @ k
    """
    user_embedding = model.users_emb.weight
    item_embedding = model.items_emb.weight

    # get ratings between every user and item - shape is num users x num movies
    rating = torch.matmul(user_embedding, item_embedding.T)

    for exclude_edge_index in exclude_edge_indices:
        # gets all the positive items for each user from the edge index
        user_pos_items = get_user_positive_items(exclude_edge_index)
        # get coordinates of all edges to exclude
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)

        # set ratings of excluded edges to large negative value
        rating[exclude_users, exclude_items] = -(1 << 10)

    # get the top k recommended items for each user
    _, top_K_items = torch.topk(rating, k=k)

    # get all unique users in evaluated split
    users = edge_index[0].unique()

    test_user_pos_items = get_user_positive_items(edge_index)

    # convert test user pos items dictionary into a list
    test_user_pos_items_list = [
        test_user_pos_items[user.item()] for user in users]

    # determine the correctness of topk predictions
    r = []
    for user in users:
        ground_truth_items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
        r.append(label)
    r = torch.Tensor(np.array(r).astype('float'))

    recall, precision = RecallPrecision_ATk(test_user_pos_items_list, r, k)
    ndcg = NDCGatK_r(test_user_pos_items_list, r, k)

    return recall, precision, ndcg

@torch.no_grad()
def get_metrics(model, edge_index, exclude_edge_indices, k, batch_size=512):
    """
    Computes recall, precision, and ndcg @ k using a memory-efficient 
    batching approach to avoid dense matrix explosion on limited RAM.
    """
    model.eval()
    
    # 1. Prepare embeddings
    # We use the raw weights here to avoid re-running the full GCN 
    # if you've already called model.forward() recently, but 
    # ideally you pass the final embeddings into this function.
    user_emb = model.users_emb.weight
    item_emb = model.items_emb.weight
    num_users = user_emb.size(0)
    device = user_emb.device

    # 2. Pre-process ground truth for fast lookup
    # We use a dictionary of tensors to avoid the slow 'item()' calls in loops
    # This maps: user_id -> tensor([pos_item_1, pos_item_2, ...])
    ground_truth = {}
    u_idx_gt = edge_index[0]
    i_idx_gt = edge_index[1]
    for u in u_idx_gt.unique():
        ground_truth[u.item()] = i_idx_gt[u_idx_gt == u]

    recalls, precisions, ndcgs = [], [], []

    # 3. Batch processing loop
    for i in range(0, num_users, batch_size):
        end_idx = min(i + batch_size, num_users)
        user_batch = user_emb[i:end_idx]
        
        # Calculate rating matrix for this batch: (batch_size x num_items)
        # On Nano, this is the most memory-intensive line.
        rating = torch.matmul(user_batch, item_emb.T)

        # 4. Mask already seen items (Training data)
        for ex_index in exclude_edge_indices:
            # Find edges belonging to users in the current batch
            mask = (ex_index[0] >= i) & (ex_index[0] < end_idx)
            if mask.any():
                ex_u = ex_index[0][mask] - i
                ex_i = ex_index[1][mask]
                rating[ex_u, ex_i] = -1e10

        # 5. Get Top-K recommendations
        _, top_K_items = torch.topk(rating, k=k)
        top_K_items = top_K_items.cpu() # Move to CPU for final metric collection

        # 6. Calculate metrics for each user in batch
        for j, user_id_offset in enumerate(range(i, end_idx)):
            if user_id_offset not in ground_truth:
                continue
            
            gt_items = ground_truth[user_id_offset].cpu()
            recommended = top_K_items[j]
            
            # Check hits (using torch.isin avoids the NumPy bridge)
            hits = torch.isin(recommended, gt_items).float()
            
            # Recall & Precision
            num_pos = len(gt_items)
            num_hits = hits.sum().item()
            
            recalls.append(num_hits / num_pos if num_pos > 0 else 0)
            precisions.append(num_hits / k)
            
            # NDCG Calculation
            if num_hits > 0:
                # DCG: sum of 1/log2(rank + 1) for hits
                ranks = torch.where(hits == 1)[0] + 1
                dcg = torch.sum(1.0 / torch.log2(ranks + 1)).item()
                
                # IDCG: Perfect ranking
                idcg_count = min(num_pos, k)
                idcg = torch.sum(1.0 / torch.log2(torch.arange(1, idcg_count + 1) + 1)).item()
                
                ndcgs.append(dcg / idcg)
            else:
                ndcgs.append(0.0)

    # 7. Aggregate results
    return (
        np.mean(recalls) if recalls else 0.0,
        np.mean(precisions) if precisions else 0.0,
        np.mean(ndcgs) if ndcgs else 0.0
    )


def structured_negative_sampling_torch(edge_index, num_nodes):
    row, col = edge_index[0], edge_index[1]
    num_queries = row.size(0)
    
    # Positive edge indices encoded as single integers
    pos_idx = row * num_nodes + col
    
    # Sample random negative items
    neg_col = torch.randint(num_nodes, (num_queries,), device=row.device)
    neg_idx = row * num_nodes + neg_col
    
    # Use a pure torch version of 'isin'
    # We check if neg_idx is in pos_idx
    mask = torch.isin(neg_idx, pos_idx)
    
    # For the indices that were actually positive, we re-sample (simple version)
    # On a Jetson, we want to avoid complex loops to save CPU cycles
    while mask.any():
        n_mask = mask.sum().item()
        new_samples = torch.randint(num_nodes, (n_mask,), device=row.device)
        neg_col[mask] = new_samples
        neg_idx[mask] = row[mask] * num_nodes + new_samples
        mask = torch.isin(neg_idx, pos_idx)
        
    return row, col, neg_col

# wrapper function to evaluate model
def evaluation(model, edge_index, sparse_edge_index, exclude_edge_indices, k, lambda_val):
    """Evaluates model loss and metrics including recall, precision, ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        sparse_edge_index (sparseTensor): sparse adjacency matrix for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        k (int): determines the top k items to compute metrics on
        lambda_val (float): determines lambda for bpr loss

    Returns:
        tuple: bpr loss, recall @ k, precision @ k, ndcg @ k
    """
    # get embeddings
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
        sparse_edge_index)
    
    """
    edges = structured_negative_sampling(
        edge_index, num_nodes=model.num_items, contains_neg_self_loops=False)
    user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]
    """

    user_indices, pos_item_indices, neg_item_indices = structured_negative_sampling_torch(
        edge_index, num_nodes=model.num_items
    )
    
    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[
        pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[
        neg_item_indices], items_emb_0[neg_item_indices]

    loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0,
                    neg_items_emb_final, neg_items_emb_0, lambda_val).item()

    recall, precision, ndcg = get_metrics(
        model, edge_index, exclude_edge_indices, k)

    return loss, recall, precision, ndcg