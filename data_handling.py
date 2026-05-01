import random
import pandas as pd
import torch
from torch_geometric.utils import structured_negative_sampling

# load user and movie nodes
def load_node_csv(path, index_col, header=0, delimiter=',', col_names=None, index_name=None):
    """Loads csv containing node information

    Args:
        path (str): path to csv file
        index_col (str): column name of index column

    Returns:
        dict: mapping of csv row to node id
    """
    df = pd.read_csv(path, index_col=index_col, delimiter=delimiter, header=header)
    if col_names: df.columns = col_names
    if index_name: df = df.rename_axis(index_name, axis='index')
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    return mapping


def dominating_set_mapping(dominating_set, usr_mapping, itm_mapping):
  users, items = [], []

  for element in dominating_set:
    if element.startswith('u'):
      users.append(usr_mapping[int(element[1:])])
    elif element.startswith('i'):
      items.append(itm_mapping[int(element[1:])])

  return users, items

# load edges between users and movies
def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping, link_index_col, rating_threshold=4):
    """Loads csv containing edges between users and items

    Args:
        path (str): path to csv file
        src_index_col (str): column name of users
        src_mapping (dict): mapping between row number and user id
        dst_index_col (str): column name of items
        dst_mapping (dict): mapping between row number and item id
        link_index_col (str): column name of user item interaction
        rating_threshold (int, optional): Threshold to determine positivity of edge. Defaults to 4.

    Returns:
        torch.Tensor: 2 by N matrix containing the node ids of N user-item edges
    """
    if path.endswith('.dat'): 
       df = pd.read_csv(path, delimiter='::', header=None, names=['userId', 'movieId', 'rating', 'timestamp'])
    else:
       df = pd.read_csv(path)
    edge_index = None
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]

    raw_values = df[link_index_col].to_numpy(copy=True)
    edge_attr = torch.as_tensor(raw_values, dtype=torch.long).view(-1, 1) >= rating_threshold

    #edge_attr = torch.from_numpy(df[link_index_col].values).view(-1, 1).to(torch.long) >= rating_threshold


    edge_index = [[], []]
    for i in range(edge_attr.shape[0]):
        if edge_attr[i]:
            edge_index[0].append(src[i])
            edge_index[1].append(dst[i])

    return torch.tensor(edge_index)


# function which random samples a mini-batch of positive and negative samples
def sample_mini_batch_old(batch_size, edge_index):
    """Randomly samples indices of a minibatch given an adjacency matrix

    Args:
        batch_size (int): minibatch size
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        tuple: user indices, positive item indices, negative item indices
    """
    edges = structured_negative_sampling(edge_index, num_nodes=edge_index[1].max().item())
    edges = torch.stack(edges, dim=0)
    indices = random.choices(
        [i for i in range(edges[0].shape[0])], k=batch_size)
    batch = edges[:, indices]
    user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]
    return user_indices, pos_item_indices, neg_item_indices


def sample_mini_batch(batch_size, edge_index):
    # 1. Get unique users and the total number of items
    num_edges = edge_index.shape[1]
    num_items = edge_index[1].max().item() + 1
    
    # 2. Randomly sample 'batch_size' indices from existing edges (Positive samples)
    indices = torch.randint(0, num_edges, (batch_size,))
    user_indices = edge_index[0, indices]
    pos_item_indices = edge_index[1, indices]
    
    # 3. Sample random items (Negative samples)
    # Note: For strict PhD-level accuracy, you'd check if these are truly negative,
    # but for a quick fix, random sampling is usually sufficient for training.
    neg_item_indices = torch.randint(0, num_items, (batch_size,))
    
    return user_indices, pos_item_indices, neg_item_indices

import torch

