from data_handling import *
from torch_sparse import SparseTensor
from sklearn.model_selection import train_test_split
import networkx as nx
import logging
import numpy as np
import torch
import os

# This creates a child logger that inherits from the root you configured above
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(movie_path, rating_path, rating_threshold=3):
    if movie_path.endswith('.dat'): 
        user_mapping = load_node_csv(rating_path, index_col=0, header=None, delimiter='::', col_names=['movieId', 'rating',	'timestamp'], index_name='userId')
        movie_mapping = load_node_csv(movie_path, index_col=0, header=None, delimiter='::', col_names=['title', 'genres'], index_name='movieId')
    else:
        user_mapping = load_node_csv(rating_path, index_col='userId')
        movie_mapping = load_node_csv(movie_path, index_col='movieId')

    for key in movie_mapping:
        movie_mapping[key] += len(user_mapping)

    logger.info(f'Users: {len(user_mapping)}, Items: {len(movie_mapping)}')

    edge_index = load_edge_csv(
        rating_path,
        src_index_col='userId',
        src_mapping=user_mapping,
        dst_index_col='movieId',
        dst_mapping=movie_mapping,
        link_index_col='rating',
        rating_threshold=rating_threshold,
    )

    logger.info(f'Edge index: {edge_index.shape}')

    return(user_mapping, movie_mapping, edge_index)

def generate_graph(edge_index):
    users = ['u' + str(u) for u in edge_index.tolist()[0]]
    items = ['i' + str(i) for i in edge_index.tolist()[1]]

    G = nx.Graph()
    G.add_nodes_from(users, bipartite=0)
    G.add_nodes_from(items, bipartite=1)
    G.add_edges_from(zip(users,items))

    logger.info(f'Number of unique users: {len(set(users))}')
    logger.info(f'Number of unique items: {len(set(items))}')
    logger.info(f'Number of nodes: {G.number_of_nodes()}')
    logger.info(f'Number of edges: {G.number_of_edges()}')

    return G

def get_ds_edges(dom_set, edge_index, G, strict=False):
    ds_users = [int(x[1:]) for x in dom_set if x[0]=='u']
    ds_movies = [int(x[1:]) for x in dom_set if x[0]=='i']

    if strict:
        ds_indices = torch.logical_and(torch.isin(edge_index[0], torch.tensor(ds_users)),
                                torch.isin(edge_index[1], torch.tensor(ds_movies))).nonzero().reshape(-1).tolist()
    else:
        ds_indices = torch.logical_or(torch.isin(edge_index[0], torch.tensor(ds_users)),
                                torch.isin(edge_index[1], torch.tensor(ds_movies))).nonzero().reshape(-1).tolist()
    
    logger.info(f'     Number of nodes: {len(dom_set)}')
    logger.info(f'     Ratio: {len(dom_set) / G.number_of_nodes()}')
    logger.info(f'     Number of unique users: {len(ds_users)}')
    logger.info(f'     Number of unique items: {len(ds_movies)}')
    logger.info(f'     Total dominant set edge: {len(ds_indices)}')
    logger.info(f'     Trainable edges: {len(ds_indices)*0.8}')

    return(ds_indices)

def get_users_items(dom_set):
    ds_users = [int(x[1:]) for x in dom_set if x[0]=='u']
    ds_items = [int(x[1:]) for x in dom_set if x[0]=='i']
    
    return {'ds_users':len(ds_users), 'ds_items': len(ds_items)}

def split_data(edge_index):
    num_interactions = edge_index.shape[1]
    all_indices = [i for i in range(num_interactions)]

    train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=1)
    val_indices, test_indices = train_test_split(test_indices, test_size=0.5, random_state=1)

    train_edge_index = edge_index[:, train_indices]
    val_edge_index = edge_index[:, val_indices]
    test_edge_index = edge_index[:, test_indices]

    return train_indices, val_indices, test_indices, train_edge_index, val_edge_index, test_edge_index

def split_data_stratified(edge_index, test_size=0.1, val_size=0.1):
    # Convert to DataFrame for easier grouping
    edges = edge_index.t().numpy()
    df = pd.DataFrame(edges, columns=['user', 'item'])
    
    train_list, val_list, test_list = [], [], []
    
    # Stratify by user
    for user, group in df.groupby('user'):
        indices = group.index.tolist()
        n = len(indices)
        
        if n < 3: 
            # If user has very few edges, keep them in train to maintain connectivity
            train_list.extend(indices)
            continue
            
        # Shuffle user-specific edges
        np.random.shuffle(indices)
        
        n_test = max(1, int(n * test_size))
        n_val = max(1, int(n * val_size))
        
        test_list.extend(indices[:n_test])
        val_list.extend(indices[n_test : n_test + n_val])
        train_list.extend(indices[n_test + n_val:])

    # Convert back to tensors
    train_edge_index = edge_index[:, train_list]
    val_edge_index = edge_index[:, val_list]
    test_edge_index = edge_index[:, test_list]
    
    return train_list, val_list, test_list, train_edge_index, val_edge_index, test_edge_index

def get_sparse_tensor(edge_index, indices, user_mapping, movie_mapping):
    num_users, num_movies = len(user_mapping), len(movie_mapping)

    sub_edge_index = edge_index[:, indices]
    sparse_edge_index = SparseTensor(row=torch.cat([sub_edge_index[0],sub_edge_index[1]]), 
                                    col=torch.cat([sub_edge_index[1],sub_edge_index[0]]), 
                                    sparse_sizes=(num_users + num_movies, num_users + num_movies))
    return sparse_edge_index

def get_initial_sparse_edge_indexes(edge_index, train_indices, val_indices, test_indices, user_mapping, movie_mapping):
    train_sparse_edge_index = get_sparse_tensor(edge_index, train_indices, user_mapping, movie_mapping)
    val_sparse_edge_index = get_sparse_tensor(edge_index, val_indices, user_mapping, movie_mapping)
    test_sparse_edge_index = get_sparse_tensor(edge_index, test_indices, user_mapping, movie_mapping)

    return train_sparse_edge_index, val_sparse_edge_index, test_sparse_edge_index    